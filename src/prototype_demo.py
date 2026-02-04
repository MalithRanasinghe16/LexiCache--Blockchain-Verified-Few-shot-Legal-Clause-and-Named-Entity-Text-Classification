# src/prototype_demo.py
"""
LexiCache - Supervisor Prototype Demo
Blockchain Verified Few-shot Legal Clause and Named Entity Text Classification

This demo shows real-time few-shot inference using the meta-trained model
(achieved 89.87% macro F1 on LEDGAR 5-way 5-shot).

Run with:
    python -m src.prototype_demo
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import random
from typing import List, Tuple

print("=" * 70)
print("LexiCache Supervisor Prototype Demo")
print("Blockchain Verified Few-shot Legal Clause Classification")
print("Final Year Project - Informatics Institute of Technology (IIT)")
print("In collaboration with University of Westminster")
print("Student: T.M.M.S. Ranasinghe")
print("Supervisor: Mr. Jihan Jeeth")
print("=" * 70)
print("Model performance: ~89.9% macro F1 (5-way 5-shot on LEDGAR)")
print("Meta-trained projection head loaded from 'projection_head.pth'")
print("=" * 70)
print()

class PrototypicalNetwork(torch.nn.Module):
    def __init__(self, encoder_name: str = "nlpaueb/legal-bert-base-uncased"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Using device: {self.device}")
        self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)


def normalize_text(text: str) -> str:
    """Simple normalization (you can paste your full version here)"""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Add your full date masking and cleaning here if you want
    text = ' '.join(text.split())
    return text


def load_projection(model: PrototypicalNetwork, path: str = "projection_head.pth"):
    projection = torch.nn.Linear(model.hidden_size, model.hidden_size).to(model.device)
    state = torch.load(path, map_location=model.device)
    projection.load_state_dict(state)
    projection.eval()
    print(f"[Demo] Loaded trained projection head from {path}")
    return projection


def few_shot_predict(
    query_text: str,
    model,
    projection,
    dataset,
    n_way: int = 5,
    k_shot: int = 5,
    apply_normalization: bool = True
) -> Tuple[str, float]:
    """Perform few-shot prediction on a new legal clause."""
    train_texts = dataset['train']['text']
    train_labels = np.array(dataset['train']['label'])
    
    query = normalize_text(query_text) if apply_normalization else query_text
    
    # Sample 5 classes
    classes = np.random.choice(np.unique(train_labels), n_way, replace=False)
    
    support_texts = []
    support_labels = []
    
    for cls in classes:
        cls_idx = np.where(train_labels == cls)[0]
        selected = np.random.choice(cls_idx, min(k_shot, len(cls_idx)), replace=False)
        support_texts.extend([train_texts[i] for i in selected])
        support_labels.extend([cls] * len(selected))
    
    support_texts = [normalize_text(t) if apply_normalization else t for t in support_texts]
    
    # Encode
    with torch.no_grad():
        support_emb = model(support_texts, batch_size=8)
        query_emb  = model([query], batch_size=1)
    
    # Project
    support_emb = projection(support_emb.to(model.device))
    query_emb   = projection(query_emb.to(model.device))
    
    # Prototypes
    unique_classes = np.unique(support_labels)
    prototypes = []
    for i, cls in enumerate(unique_classes):
        mask = np.array(support_labels) == cls
        prototypes.append(support_emb[mask].mean(dim=0))
    prototypes = torch.stack(prototypes)
    
    # Distance & prediction
    dists = torch.cdist(query_emb, prototypes)
    pred_idx = torch.argmin(dists, dim=1).item()
    pred_class = int(unique_classes[pred_idx])
    confidence = torch.softmax(-dists, dim=1)[0, pred_idx].item()
    
    return pred_class, confidence


def main():
    print("Loading model and dataset... (first run may take 1–2 minutes)")
    
    model = PrototypicalNetwork()
    projection = load_projection(model)
    dataset = load_dataset("lex_glue", "ledgar")
    
    print("\nPrototype ready! Enter a legal clause/text to classify.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        query = input("Enter legal clause (or 'exit'): ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting demo. Goodbye!")
            break
        if not query:
            print("Please enter some text.")
            continue
        
        print("\nRunning few-shot classification (5-way 5-shot)...")
        pred_class, confidence = few_shot_predict(
            query_text=query,
            model=model,
            projection=projection,
            dataset=dataset,
            apply_normalization=True
        )
        
        print(f"\nPrediction:")
        print(f"  → Predicted provision type ID: {pred_class}")
        print(f"  → Confidence: {confidence:.3%}")
        print("-" * 60)


if __name__ == "__main__":
    main()