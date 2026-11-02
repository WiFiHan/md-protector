"""MD-ProTector model. Labels: 0 = machine, 1 = human."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.encoder import TextEmbeddingModel

# Unused buffers in checkpoints from our original training code; dropped when loading.
LEGACY_STATE_KEYS = ("class_centers", "eval_score_means", "eval_score_stds")


class MDProTector(nn.Module):
    """Text encoder with R prototypes per class.

    pp_ablation: "none", "drop" (w/o L_PP) or "repulsion" (L_PP replaced by prototype repulsion)
    pp_space: "residual" or "raw" (PP without removing the class hub)
    inference: "max" (Eq. 13) or "weighted" (Appendix B.2)
    """

    def __init__(self, model_name, num_prototypes=8, dim=768, tau=0.15,
                 pp_ablation="none", pp_space="residual", inference="max"):
        super().__init__()
        if pp_ablation not in {"none", "drop", "repulsion"}:
            raise ValueError(f"Unknown pp_ablation: {pp_ablation}")
        if pp_space not in {"residual", "raw"}:
            raise ValueError(f"Unknown pp_space: {pp_space}")
        if inference not in {"max", "weighted"}:
            raise ValueError(f"Unknown inference: {inference}")

        self.model = TextEmbeddingModel(model_name)
        self.num_classes = 2
        self.R = num_prototypes
        self.tau = tau
        self.pp_ablation = pp_ablation
        self.pp_space = pp_space
        self.inference = inference

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.R, dim))
        self.register_buffer("proto_inited", torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def initialize_prototypes(self, loader, fabric, method="kmeans"):
        if method == "kmeans":
            if fabric.global_rank == 0:
                print("Running K-Means initialization for prototypes...")
            P = self._kmeans_init(loader, fabric)
        elif method == "random":
            P = torch.randn(self.num_classes, self.R, self.prototypes.size(-1), device=fabric.device)
            P = F.normalize(P, dim=-1)
        else:
            raise ValueError(f"Unknown proto_init: {method}")
        self.prototypes.copy_(F.normalize(P, dim=-1))
        self.proto_inited.fill_(1)

    @torch.no_grad()
    def _kmeans_init(self, loader, fabric):
        """K-Means on the training embeddings, separately for each class."""
        device = fabric.device
        self.model = self.model.to(device)
        self.model.eval()

        all_z0, all_z1 = [], []
        for batch in tqdm(loader, desc="K-Means embeddings", disable=(fabric.global_rank != 0)):
            encoded_batch, label = batch
            encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
            z = F.normalize(self.model(encoded_batch), dim=-1)
            if fabric.world_size > 1:
                gathered_z = fabric.all_gather(z)
                gathered_label = fabric.all_gather(label)
                if gathered_z.dim() > 2:
                    gathered_z = gathered_z.view(-1, gathered_z.size(-1))
                    gathered_label = gathered_label.view(-1)
            else:
                gathered_z, gathered_label = z, label
            all_z0.append(gathered_z[gathered_label == 0].cpu())
            all_z1.append(gathered_z[gathered_label == 1].cpu())

        z0 = torch.cat(all_z0, dim=0).unique(dim=0)
        z1 = torch.cat(all_z1, dim=0).unique(dim=0)

        if fabric.global_rank == 0:
            print(f"K-Means (R={self.R}) - Machine:{z0.size(0)} / Human:{z1.size(0)}")
            centroids_0 = self._torch_kmeans(z0.to(device), self.R).cpu()
            centroids_1 = self._torch_kmeans(z1.to(device), self.R).cpu()
            P = torch.stack([centroids_0, centroids_1], dim=0)
        else:
            P = torch.empty(self.num_classes, self.R, self.prototypes.size(-1), dtype=torch.float32)
        if fabric.world_size > 1:
            fabric.broadcast(P, src=0)
        return P.to(device)

    @staticmethod
    @torch.no_grad()
    def _torch_kmeans(data, R, n_iter=20):
        """Spherical K-Means on unit-norm embeddings."""
        device = data.device
        if data.size(0) == 0:
            return F.normalize(torch.randn(R, data.size(1), device=device), dim=-1)
        if data.size(0) < R:
            indices = torch.randint(0, data.size(0), (R,), device=device)
            return data[indices]

        indices = torch.randperm(data.size(0), device=device)[:R]
        centroids = data[indices]
        for _ in range(n_iter):
            assignments = torch.argmax(data @ centroids.T, dim=1)
            new_centroids = torch.zeros(R, data.size(1), device=device)
            for r in range(R):
                mask = (assignments == r)
                if mask.sum() == 0:
                    new_centroids[r] = data[torch.randint(0, data.size(0), (1,), device=device)]
                else:
                    new_centroids[r] = F.normalize(data[mask].mean(dim=0), dim=-1)
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return centroids

    @torch.no_grad()
    def renorm_prototypes_(self):
        self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))

    def _p2c_and_pp_losses(self, z, y):
        """L_P2C (Eq. 5) and L_PP (Eqs. 8-11)."""
        P = F.normalize(self.prototypes, dim=-1)

        # class hubs (Eq. 2)
        class_hubs = []
        for cls in range(self.num_classes):
            mask = (y == cls)
            if mask.any():
                hub = F.normalize(z[mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            else:
                hub = F.normalize(P[cls].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            class_hubs.append(hub)
        class_hubs = torch.stack(class_hubs, dim=0)

        p2c_logits = torch.einsum("crd,hd->crh", P, class_hubs) / self.tau
        p2c_targets = torch.arange(self.num_classes, device=z.device).view(self.num_classes, 1).expand(self.num_classes, self.R)
        loss_p2c = F.cross_entropy(p2c_logits.reshape(-1, self.num_classes), p2c_targets.reshape(-1))

        if self.pp_ablation == "drop":
            return loss_p2c, z.new_tensor(0.0)
        if self.pp_ablation == "repulsion":
            return loss_p2c, self._prototype_repulsion_loss()

        # remove the hub direction from the prototypes (Eq. 9)
        if self.pp_space == "residual":
            hubs_expanded = class_hubs.unsqueeze(1)
            proj_proto = (P * hubs_expanded).sum(dim=-1, keepdim=True) * hubs_expanded
            P_child = F.normalize(P - proj_proto, dim=-1, eps=1e-8)
        else:
            P_child = P
        P_child_all = P_child.reshape(self.num_classes * self.R, -1)

        pp_terms = []
        for cls in range(self.num_classes):
            mask = (y == cls)
            if not mask.any():
                continue
            zc = z[mask]
            hub = class_hubs[cls]
            q_cls = F.softmax((zc @ P[cls].t()) / self.tau, dim=-1)
            if self.pp_space == "residual":
                zc_basis = zc - (zc @ hub).unsqueeze(1) * hub.unsqueeze(0)
            else:
                zc_basis = zc
            # q-weighted residual aggregate for each prototype (Eq. 10)
            region_sum = q_cls.t() @ zc_basis
            region_norm = region_sum.norm(dim=1)
            valid = region_norm > 1e-8
            if not valid.any():
                continue
            region = region_sum[valid] / region_norm[valid].unsqueeze(1)
            logits = (region @ P_child_all.t()) / self.tau
            targets = cls * self.R + torch.arange(self.R, device=z.device)[valid]
            pp_terms.append(F.cross_entropy(logits, targets))
        loss_pp = torch.stack(pp_terms).mean() if pp_terms else z.new_tensor(0.0)
        return loss_p2c, loss_pp

    def _prototype_repulsion_loss(self):
        """Ablation that replaces L_PP: pushes apart prototypes of the same class."""
        P = F.normalize(self.prototypes, dim=-1)
        if self.R <= 1:
            return P.new_tensor(0.0)
        mask = ~torch.eye(self.R, dtype=torch.bool, device=P.device)
        sims = torch.cat([(P[cls] @ P[cls].t())[mask] for cls in range(self.num_classes)], dim=0)
        return F.relu(sims).pow(2).mean()

    def _s2p_loss(self, z, y):
        """L_S2P (Eq. 7). The assignments q are detached."""
        P = F.normalize(self.prototypes, dim=-1)
        q = F.softmax(torch.einsum("bd,brd->br", z, P[y]) / self.tau, dim=-1).detach()
        logits_all = (z @ P.view(self.num_classes * self.R, -1).t()) / self.tau
        logp_all = F.log_softmax(logits_all, dim=1)
        idx = (y * self.R).view(-1, 1) + torch.arange(self.R, device=z.device).view(1, -1)
        logp_pos = logp_all.gather(1, idx)
        return -(q * logp_pos).sum(dim=1).mean()

    def class_scores(self, z):
        """Returns [s_0(z), s_1(z)], divided by tau."""
        P_all = F.normalize(self.prototypes, dim=-1)
        scores = []
        for cls in range(self.num_classes):
            logits = (z @ P_all[cls].t()) / self.tau
            if self.inference == "max":
                scores.append(logits.max(dim=1).values)
            else:
                w = F.softmax((z @ P_all[cls].t()) / self.tau, dim=-1)
                scores.append((w * logits).sum(dim=1))
        return scores

    def forward(self, batch, label=None):
        """Train mode: (loss, L_P2C, L_S2P, L_PP). Eval mode: S(z) = s_1(z) - s_0(z)."""
        z = F.normalize(self.model(batch), dim=-1)
        if not self.training:
            score0, score1 = self.class_scores(z)
            return score1 - score0

        y = label.to(z.device).view(-1)
        loss_p2c, loss_pp = self._p2c_and_pp_losses(z, y)
        loss_s2p = self._s2p_loss(z, y)
        loss = (loss_p2c + loss_pp) + loss_s2p
        return loss, loss_p2c.detach(), loss_s2p.detach(), loss_pp.detach()
