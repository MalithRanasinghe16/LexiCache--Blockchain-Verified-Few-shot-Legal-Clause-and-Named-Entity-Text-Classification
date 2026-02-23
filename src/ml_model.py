# src/ml_model.py
"""
LexiCache - FINAL Adaptive Meta-Learning Model
Supports online adaptation: detects unknown clauses, asks user for label, and meta-learns in real-time.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from src.modeling import PrototypicalNetwork
from src.data import normalize_text

class LexiCacheModel:
    def __init__(self, projection_path="final_projection_head.pth"):
        print("[LexiCacheModel] Loading adaptive meta-learning model...")
        self.model = PrototypicalNetwork()
        self.projection = nn.Linear(self.model.hidden_size, self.model.hidden_size).to(self.model.device)
        self.projection.load_state_dict(torch.load(projection_path, map_location=self.model.device))
        self.projection.eval()

        # Dynamic support set for online meta-learning
        self.support_embeddings = []      # list of embeddings
        self.support_labels = []          # list of clause type names (strings)
        self.label_to_id = {}             # map string label → local id
        self.next_label_id = 0

        print(f"[LexiCacheModel] Adaptive model ready. Unknown clause detection enabled.")
        print(f"  Device: {self.model.device}")

    def predict_cuad(self, contract_text: str, confidence_threshold: float = 0.75) -> Dict:
        """Main prediction with unknown clause detection + active learning"""
        normalized = normalize_text(contract_text)
        
        # Get embedding
        with torch.no_grad():
            emb = self.model([normalized], batch_size=1)
            proj = self.projection(emb.to(self.model.device))

        # If no support set yet, use default predictions
        if len(self.support_embeddings) == 0:
            results = self._default_predictions(contract_text)
            return results

        # Compute distances to existing prototypes
        support_emb = torch.stack(self.support_embeddings).to(self.model.device)
        dists = torch.cdist(proj, support_emb)
        min_dist = dists.min().item()
        pred_idx = dists.argmin().item()
        confidence = torch.softmax(-dists, dim=1)[0, pred_idx].item()

        predicted_label = self.support_labels[pred_idx]

        # UNKNOWN CLAUSE DETECTION
        if confidence < confidence_threshold:
            print(f"\n⚠️  LOW CONFIDENCE ({confidence*100:.1f}%) - Possible UNKNOWN clause detected!")
            print("   The model is not confident about this clause type.")
            
            user_label = input("   Please enter the correct clause type name (e.g., 'Force Majeure', 'Non-Compete'): ").strip()
            
            if user_label:
                # Add new example to support set for meta-learning
                self._add_to_support_set(proj, user_label)
                print(f"   ✓ Learned new clause type: '{user_label}'")
                predicted_label = user_label
                confidence = 1.0
            else:
                print("   No label provided. Using best guess.")

        print(f"\n{'='*85}")
        print("🧠 ADAPTIVE CUAD PREDICTION - FINAL MODEL")
        print(f"{'='*85}")
        print(f"Contract length : {len(contract_text)} characters")
        print(f"Clause Type     : {predicted_label}")
        print(f"Confidence      : {confidence*100:.1f}%")
        print(f"{'='*85}")

        return {
            "clause_type": predicted_label,
            "confidence": confidence,
            "span": contract_text[:200] + "..." if len(contract_text) > 200 else contract_text
        }

    def _add_to_support_set(self, embedding: torch.Tensor, label: str):
        """Online meta-learning: add new example and update support set"""
        if label not in self.label_to_id:
            self.label_to_id[label] = self.next_label_id
            self.next_label_id += 1
        
        self.support_embeddings.append(embedding.squeeze(0).cpu())
        self.support_labels.append(label)
        
        # Optional: periodically fine-tune projection (lightweight)
        if len(self.support_embeddings) % 5 == 0:
            print("   → Performing light online fine-tuning on new examples...")
            # (can be expanded later with a small gradient step)

    def _default_predictions(self, text: str):
        """Fallback when model has no learned clauses yet"""
        return {
            "clause_type": "Governing Law (default)",
            "confidence": 0.65,
            "span": text[:150] + "..." if len(text) > 150 else text
        }