"""Train and evaluate MD-ProTector.

    python train.py --config configs/mage_cdcm.yaml
    python train.py --config configs/mage_cdcm.yaml --eval_only --model_path runs/mage_cdcm_v0/model_best.pth
"""
import argparse
import json
import os
import random
from functools import partial

import numpy as np
import torch
import torch.optim as optim
import yaml
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import TextDataset, collate_fn, load_splits
from src.md_protector import LEGACY_STATE_KEYS, MDProTector
from src.metrics import compute_metrics

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(base_seed):
    def worker_init_fn(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn


def make_run_dir(opt):
    if opt.eval_only:
        run_dir = os.path.join(opt.savedir, f"{opt.name}_eval")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    for num in range(10000):
        run_dir = os.path.join(opt.savedir, f"{opt.name}_v{num}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
    raise RuntimeError("Could not create a run directory")


def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if k not in LEGACY_STATE_KEYS}
    model.load_state_dict(state_dict, strict=True)
    model.renorm_prototypes_()


@torch.no_grad()
def predict(model, loader, fabric, desc):
    model.eval()
    scores, labels = [], []
    for encoded_batch, label in tqdm(loader, desc=desc, disable=(fabric.global_rank != 0)):
        score = model(encoded_batch)
        if fabric.world_size > 1:
            score = fabric.all_gather(score.contiguous()).reshape(-1)
            label = fabric.all_gather(label.contiguous()).reshape(-1)
        scores.append(score.float().cpu())
        labels.append(label.cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def evaluate(model, valid_loader, test_loader, fabric, opt, run_dir):
    """Pick the threshold on the validation split and report test metrics with it."""
    valid_scores, valid_labels = predict(model, valid_loader, fabric, f"Eval {opt.valid_split}")
    test_scores, test_labels = predict(model, test_loader, fabric, f"Eval {opt.test_split}")
    if fabric.global_rank != 0:
        return None
    valid_metrics = compute_metrics(valid_labels, valid_scores)
    test_metrics = compute_metrics(test_labels, test_scores, threshold=valid_metrics["threshold"])
    results = {
        "inference": opt.inference,
        "threshold_source": opt.valid_split,
        opt.valid_split: valid_metrics,
        opt.test_split: test_metrics,
    }
    out_path = os.path.join(run_dir, f"eval_{opt.test_split}_{opt.inference}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{opt.test_split}] AvgRec {test_metrics['avg_rec']:.2f} "
          f"(HumanRec {test_metrics['human_rec']:.2f} / MachineRec {test_metrics['machine_rec']:.2f}), "
          f"AUROC {test_metrics['auroc']:.2f}, FPR95 {test_metrics['fpr_at_tpr_95']:.2f} -> {out_path}")
    return results


def main(opt):
    set_random_seed(opt.random_seed)
    torch.set_float32_matmul_precision("medium")
    strategy = DDPStrategy(find_unused_parameters=True) if opt.device_num > 1 else "auto"
    fabric = Fabric(accelerator=opt.accelerator, precision=opt.precision, devices=opt.device_num, strategy=strategy)
    fabric.launch()

    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    collate = partial(collate_fn, tokenizer=tokenizer, max_length=opt.max_length)
    worker_init_fn = make_worker_init_fn(opt.random_seed)

    split_names = [opt.valid_split, opt.test_split]
    if not opt.eval_only:
        split_names = [opt.train_split] + split_names
    data = load_splits(opt, split_names)

    def eval_loader(split):
        # A loader without its own generator draws its worker seed from the global RNG, which
        # would shift the training RNG state after every evaluation.
        eval_gen = torch.Generator()
        eval_gen.manual_seed(opt.random_seed + 2)
        return DataLoader(TextDataset(data[split]), batch_size=opt.per_gpu_eval_batch_size,
                          num_workers=opt.num_workers, pin_memory=True, shuffle=False, drop_last=False,
                          collate_fn=collate, worker_init_fn=worker_init_fn, generator=eval_gen)

    valid_loader, test_loader = eval_loader(opt.valid_split), eval_loader(opt.test_split)
    if not opt.eval_only:
        train_gen = torch.Generator()
        train_gen.manual_seed(opt.random_seed)
        train_loader = DataLoader(TextDataset(data[opt.train_split]), batch_size=opt.per_gpu_batch_size,
                                  num_workers=opt.num_workers, pin_memory=True, shuffle=True, drop_last=True,
                                  collate_fn=collate, worker_init_fn=worker_init_fn, generator=train_gen)

    model = MDProTector(opt.model_name, num_prototypes=opt.num_prototypes, dim=opt.out_dim, tau=opt.tau,
                        pp_ablation=opt.pp_ablation, pp_space=opt.pp_space, inference=opt.inference)
    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if "emb" in name:
                param.requires_grad = False

    valid_loader, test_loader = fabric.setup_dataloaders(valid_loader, test_loader)

    run_dir = make_run_dir(opt) if fabric.global_rank == 0 else None
    if fabric.global_rank == 0:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)
        print(f"Run directory: {run_dir}")

    if opt.eval_only:
        if not opt.model_path:
            raise ValueError("--eval_only requires --model_path")
        model = fabric.setup_module(model)
        load_checkpoint(model, opt.model_path)
        evaluate(model, valid_loader, test_loader, fabric, opt, run_dir)
        return

    train_loader = fabric.setup_dataloaders(train_loader)
    num_batches_per_epoch = len(train_loader)
    total_steps = opt.total_epoch * num_batches_per_epoch - opt.warmup_steps
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr,
                            betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=opt.lr / 10)
    model = fabric.setup_module(model)
    model.mark_forward_method("initialize_prototypes")
    optimizer = fabric.setup_optimizers(optimizer)

    model.initialize_prototypes(train_loader, fabric, opt.proto_init)

    best_path = os.path.join(run_dir, "model_best.pth") if fabric.global_rank == 0 else None
    best_avg_rec = float("-inf")
    for epoch in range(opt.total_epoch):
        model.train()
        avg_loss = 0
        pbar = enumerate(train_loader)
        if fabric.global_rank == 0:
            pbar = tqdm(pbar, total=num_batches_per_epoch, desc=f"Epoch {epoch + 1}/{opt.total_epoch}")

        for i, (encoded_batch, label) in pbar:
            optimizer.zero_grad()
            # Linear warmup, then cosine decay to lr/10.
            current_step = epoch * num_batches_per_epoch + i
            if current_step < opt.warmup_steps:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = opt.lr * current_step / opt.warmup_steps
            current_lr = optimizer.param_groups[0]["lr"]

            loss, loss_p2c, loss_s2p, loss_pp = model(encoded_batch, label)
            avg_loss = (avg_loss * i + loss.item()) / (i + 1)
            fabric.backward(loss)
            optimizer.step()
            model.renorm_prototypes_()
            if current_step >= opt.warmup_steps:
                schedule.step()

            if fabric.global_rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4g}", avg_loss=f"{avg_loss:.4g}", lr=f"{current_lr:.3g}")
                if current_step % opt.log_every == 0:
                    with open(os.path.join(run_dir, "train_log.jsonl"), "a") as f:
                        f.write(json.dumps({"step": current_step, "epoch": epoch + 1, "lr": current_lr,
                                            "loss": loss.item(), "loss_p2c": loss_p2c.item(),
                                            "loss_s2p": loss_s2p.item(), "loss_pp": loss_pp.item()}) + "\n")

        valid_scores, valid_labels = predict(model, valid_loader, fabric, f"Valid {epoch + 1}/{opt.total_epoch}")
        if fabric.global_rank == 0:
            metrics = compute_metrics(valid_labels, valid_scores)
            print(f"Epoch {epoch + 1} {opt.valid_split}: AvgRec {metrics['avg_rec']:.2f} "
                  f"(HumanRec {metrics['human_rec']:.2f} / MachineRec {metrics['machine_rec']:.2f}), "
                  f"AUROC {metrics['auroc']:.2f}")
            with open(os.path.join(run_dir, "valid_metrics.jsonl"), "a") as f:
                f.write(json.dumps({"epoch": epoch + 1, **metrics}) + "\n")
            if metrics["avg_rec"] > best_avg_rec:
                best_avg_rec = metrics["avg_rec"]
                torch.save(model.state_dict(), best_path)
                print(f"Saved best checkpoint to {best_path}")
        fabric.barrier()

    if opt.test_split:
        best_path = fabric.broadcast(best_path, src=0)
        load_checkpoint(model, best_path)
        evaluate(model, valid_loader, test_loader, fabric, opt, run_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate MD-ProTector")
    parser.add_argument("--config", type=str, default=None, help="YAML file whose keys override the defaults below")

    # Data
    parser.add_argument("--dataset", type=str, default="mage", choices=["mage", "m4", "raid"])
    parser.add_argument("--data_path", type=str, default="data/MAGE/cross_domains_cross_models",
                        help="Dataset folder (MAGE, M4); RAID is downloaded from the Hugging Face Hub")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--valid_split", type=str, default="valid", help="Checkpoint selection and threshold calibration")
    parser.add_argument("--test_split", type=str, default="test", help="Final evaluation; empty string skips it")
    parser.add_argument("--raid_split_manifest", type=str,
                        default=os.path.join(REPO_ROOT, "splits", "raid", "seed0_train90_valid10_indices.json"))
    parser.add_argument("--max_length", type=int, default=512)

    # Model
    parser.add_argument("--model_name", type=str, default="princeton-nlp/unsup-simcse-roberta-base")
    parser.add_argument("--out_dim", type=int, default=768, help="Encoder embedding dimension")
    parser.add_argument("--freeze_embedding_layer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_prototypes", type=int, default=8, help="R, prototypes per class")
    parser.add_argument("--proto_init", type=str, default="kmeans", choices=["kmeans", "random"])
    parser.add_argument("--tau", type=float, default=0.15, help="Temperature shared by all losses and assignments")
    parser.add_argument("--pp_ablation", type=str, default="none", choices=["none", "drop", "repulsion"],
                        help="none: full method; drop: w/o L_PP; repulsion: replace L_PP with prototype repulsion")
    parser.add_argument("--pp_space", type=str, default="residual", choices=["residual", "raw"],
                        help="raw: Prototype Positioning without removing the class hub")
    parser.add_argument("--inference", type=str, default="max", choices=["max", "weighted"],
                        help="max: Eq. 13; weighted: Appendix B.2")

    # Optimization
    parser.add_argument("--total_epoch", type=int, default=30)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--per_gpu_batch_size", type=int, default=256)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=256)
    parser.add_argument("--random_seed", type=int, default=42)

    # Runtime
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50, help="Steps between train_log.jsonl entries")

    # Output
    parser.add_argument("--savedir", type=str, default="runs")
    parser.add_argument("--name", type=str, default="md_protector")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--model_path", type=str, default="", help="Checkpoint for --eval_only")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - known)
        if unknown:
            parser.error(f"Unknown keys in {pre_args.config}: {unknown}")
        parser.set_defaults(**config)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
