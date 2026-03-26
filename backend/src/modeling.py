"""
Few-shot modeling for LexiCache.
Implements prototypical networks with Legal-BERT for clause classification and NER.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from tqdm import tqdm

class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder_name: str = "nlpaueb/legal-bert-base-uncased"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Using device: {self.device}")
        self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, texts: List[str], batch_size: int = 16, enable_grad: bool = False) -> torch.Tensor:
        """Encode texts in batches and return mean-pooled embeddings.

        Args:
            texts: Input strings.
            batch_size: Batch size for tokenization/encoding.
            enable_grad: When True, keeps encoder graph for finetuning.
        """
        if not texts:
            return torch.empty(0, self.hidden_size, device=self.device)

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", leave=False):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.set_grad_enabled(enable_grad):
                outputs = self.encoder(**inputs)

            # Masked mean pooling avoids padding-token contamination in sentence embeddings.
            token_emb = outputs.last_hidden_state
            attn_mask = inputs["attention_mask"].unsqueeze(-1).to(token_emb.dtype)
            masked_sum = (token_emb * attn_mask).sum(dim=1)
            lengths = attn_mask.sum(dim=1).clamp(min=1e-6)
            batch_emb = masked_sum / lengths

            # In inference mode move to CPU to reduce memory; keep on device for training.
            embeddings.append(batch_emb if enable_grad else batch_emb.cpu())
        return torch.cat(embeddings, dim=0)

    def compute_prototypes(self, support_embeds: torch.Tensor, support_labels: torch.Tensor):
        """Compute class prototypes as the mean embedding per class."""
        unique_labels = torch.unique(support_labels)
        prototypes = torch.zeros(
            len(unique_labels),
            self.hidden_size,
            device=support_embeds.device,
            dtype=support_embeds.dtype,
        )
        for i, lbl in enumerate(unique_labels):
            mask = support_labels == lbl
            if mask.sum() == 0:
                continue  # skip empty classes (rare)
            prototypes[i] = support_embeds[mask].mean(dim=0)
        return prototypes, unique_labels

    def classify(self, query_embeds: torch.Tensor, prototypes: torch.Tensor):
        """Classify queries by nearest prototype using Euclidean distance."""
        dists = torch.cdist(query_embeds, prototypes)  # [n_query, n_classes]
        preds = torch.argmin(dists, dim=-1)
        probs = F.softmax(-dists, dim=-1)
        return preds, probs