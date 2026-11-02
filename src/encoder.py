import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        # BGE uses [CLS] pooling, the other encoders use mean pooling.
        self.pooling = "cls" if "bge" in model_name.lower() else "average"

    def forward(self, encoded_batch):
        hidden = self.model(**encoded_batch).last_hidden_state
        if self.pooling == "cls":
            emb = hidden[:, 0]
        else:
            # NOTE: padding positions are not masked out before the sum. This is how the
            # paper checkpoints were trained, so we keep it.
            emb = hidden.sum(dim=1) / encoded_batch["attention_mask"].sum(dim=1)[..., None]
        return F.normalize(emb, dim=-1)
