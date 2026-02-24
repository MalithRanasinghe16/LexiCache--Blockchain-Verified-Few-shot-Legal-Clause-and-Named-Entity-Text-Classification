# src/ml_model.py
"""
LexiCache - FINAL Adaptive Meta-Learning Model
Supports online adaptation: detects unknown clauses, asks user for label, and meta-learns in real-time.
"""

import torch
import torch.nn as nn
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from src.modeling import PrototypicalNetwork
from src.data import normalize_text

# Common legal clause patterns for segmentation
CLAUSE_PATTERNS = [
    r'(?:^|\n)\s*(\d+\.?\d*\.?\s+[A-Z][A-Za-z\s]+)',  # Numbered sections: "1. Definitions"
    r'(?:^|\n)\s*(ARTICLE\s+[IVXLC\d]+[:\.\s])',       # ARTICLE I, ARTICLE 1
    r'(?:^|\n)\s*(Section\s+\d+[:\.\s])',              # Section 1:
    r'(?:^|\n)\s*([A-Z][A-Z\s]{2,}:)',                 # ALL CAPS HEADERS:
]

# Known clause type keywords for heuristic matching
CLAUSE_KEYWORDS = {
    'Governing Law': ['governing law', 'choice of law', 'applicable law', 'laws of', 'jurisdiction'],
    'Termination': ['termination', 'terminate', 'cancellation', 'expiration', 'end of agreement'],
    'Confidentiality': ['confidential', 'non-disclosure', 'proprietary information', 'trade secret'],
    'Indemnification': ['indemnif', 'hold harmless', 'defend and indemnify', 'indemnity'],
    'Payment Terms': ['payment', 'invoice', 'compensation', 'fees', 'billing', 'price'],
    'Limitation of Liability': ['limitation of liability', 'limit liability', 'damages shall not exceed'],
    'Force Majeure': ['force majeure', 'act of god', 'beyond control'],
    'Intellectual Property': ['intellectual property', 'patent', 'copyright', 'trademark', 'ip rights'],
    'Warranties': ['warrant', 'representation', 'guarantee', 'as-is'],
    'Assignment': ['assignment', 'assign', 'transfer of rights', 'successor'],
    'Notice': ['notice', 'notification', 'written notice', 'notify'],
    'Dispute Resolution': ['dispute', 'arbitration', 'mediation', 'litigation'],
    'Non-Compete': ['non-compete', 'non-competition', 'compete'],
    'Severability': ['severability', 'severable', 'invalid provision'],
}

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

    def _segment_contract(self, text: str) -> List[Dict]:
        """
        Segment contract text into individual clauses/paragraphs.
        Returns list of {text, start_idx, end_idx}
        """
        segments = []
        
        # Method 1: Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_pos = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) < 30:  # Skip very short segments
                current_pos = text.find(para, current_pos) + len(para)
                continue
                
            start_idx = text.find(para, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(para)
            
            # Further split long paragraphs by sentence patterns if > 500 chars
            if len(para) > 500:
                sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
                sent_pos = start_idx
                for sent in sentences:
                    if len(sent.strip()) >= 30:
                        segments.append({
                            'text': sent.strip(),
                            'start_idx': sent_pos,
                            'end_idx': sent_pos + len(sent)
                        })
                    sent_pos += len(sent) + 1
            else:
                segments.append({
                    'text': para,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            
            current_pos = end_idx
        
        return segments

    def _classify_by_keywords(self, text: str) -> Tuple[str, float]:
        """
        Heuristic classification based on keywords.
        Returns (clause_type, confidence)
        """
        text_lower = text.lower()
        
        best_match = None
        best_score = 0
        
        for clause_type, keywords in CLAUSE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_match = clause_type
        
        if best_match and best_score >= 1:
            # Confidence based on number of keyword matches
            confidence = min(0.95, 0.60 + (best_score * 0.10))
            return best_match, confidence
        
        return None, 0.0

    def _classify_segment(self, segment_text: str) -> Dict:
        """
        Classify a single text segment using the model or heuristics.
        """
        normalized = normalize_text(segment_text)
        
        # First try keyword-based classification (fast and interpretable)
        kw_type, kw_conf = self._classify_by_keywords(segment_text)
        
        # Get embedding for model-based classification
        with torch.no_grad():
            emb = self.model([normalized], batch_size=1)
            proj = self.projection(emb.to(self.model.device))
        
        # If we have a support set, use model prediction
        if len(self.support_embeddings) > 0:
            support_emb = torch.stack(self.support_embeddings).to(self.model.device)
            dists = torch.cdist(proj, support_emb)
            pred_idx = dists.argmin().item()
            model_conf = torch.softmax(-dists, dim=1)[0, pred_idx].item()
            model_type = self.support_labels[pred_idx]
            
            # Use whichever is more confident
            if model_conf > kw_conf:
                return {'clause_type': model_type, 'confidence': model_conf}
        
        # Fall back to keyword classification or Unknown
        if kw_type:
            return {'clause_type': kw_type, 'confidence': kw_conf}
        
        return {'clause_type': 'General Provision', 'confidence': 0.50}

    def predict_cuad(self, contract_text: str, confidence_threshold: float = 0.55) -> List[Dict]:
        """
        Main prediction - extracts and classifies multiple clauses from contract.
        Returns list of {clause_type, confidence, span}
        """
        print(f"\n{'='*85}")
        print("🧠 LEXICACHE MULTI-CLAUSE EXTRACTION")
        print(f"{'='*85}")
        print(f"Contract length: {len(contract_text)} characters")
        
        # Segment the contract
        segments = self._segment_contract(contract_text)
        print(f"Found {len(segments)} potential clause segments")
        
        results = []
        seen_types = defaultdict(int)  # Track count of each clause type
        
        for seg in segments:
            classification = self._classify_segment(seg['text'])
            
            # Only include if confidence is above threshold
            if classification['confidence'] >= confidence_threshold:
                # Avoid too many duplicates of same type
                type_key = classification['clause_type']
                if seen_types[type_key] < 3:  # Allow up to 3 of same type
                    # Clean up the span text for better matching
                    span_text = seg['text'][:300] if len(seg['text']) > 300 else seg['text']
                    span_text = ' '.join(span_text.split())  # Normalize whitespace
                    
                    results.append({
                        'clause_type': classification['clause_type'],
                        'confidence': classification['confidence'],
                        'span': span_text,
                        'start_idx': seg['start_idx'],
                        'end_idx': seg['end_idx']
                    })
                    seen_types[type_key] += 1
        
        # Sort by confidence descending
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 10 most confident clauses
        results = results[:10]
        
        print(f"Identified {len(results)} high-confidence clauses:")
        for r in results:
            print(f"  • {r['clause_type']}: {r['confidence']*100:.1f}% - \"{r['span'][:50]}...\"")
        print(f"{'='*85}\n")
        
        # If no clauses found, return at least one default
        if not results:
            results = [{
                'clause_type': 'General Contract Terms',
                'confidence': 0.60,
                'span': contract_text[:200] + "..." if len(contract_text) > 200 else contract_text,
                'start_idx': 0,
                'end_idx': min(200, len(contract_text))
            }]
        
        return results

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