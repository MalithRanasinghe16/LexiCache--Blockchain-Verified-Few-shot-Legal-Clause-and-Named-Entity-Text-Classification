# src/ml_model.py
"""
LexiCache - FINAL Adaptive Meta-Learning Model
Supports online adaptation: detects unknown clauses, asks user for label, and meta-learns in real-time.
"""

import torch
import torch.nn as nn
import numpy as np
import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from datetime import datetime

from src.modeling import PrototypicalNetwork
from src.data import normalize_text

# Common legal clause patterns for segmentation
CLAUSE_PATTERNS = [
    r'(?:^|\n)\s*(\d+\.?\d*\.?\s+[A-Z][A-Za-z\s]+)',  # Numbered sections: "1. Definitions"
    r'(?:^|\n)\s*(ARTICLE\s+[IVXLC\d]+[:\.\s])',       # ARTICLE I, ARTICLE 1
    r'(?:^|\n)\s*(Section\s+\d+[:\.\s])',              # Section 1:
    r'(?:^|\n)\s*([A-Z][A-Z\s]{2,}:)',                 # ALL CAPS HEADERS:
]

# All 41 CUAD clause types with keyword-based heuristics
# From: https://www.atticusprojectai.org/cuad
CLAUSE_KEYWORDS = {
    # Core Agreement Terms
    'Document Name': ['agreement', 'contract', 'master agreement', 'this agreement'],
    'Parties': ['party', 'parties', 'by and between', 'hereinafter'],
    'Agreement Date': ['dated', 'effective date', 'as of', 'entered into'],
    'Effective Date': ['effective', 'commence', 'commencement date', 'start date'],
    'Expiration Date': ['expire', 'expiration', 'term end', 'until'],
    
    # Financial & Payment
    'Payment Terms': ['payment', 'invoice', 'compensation', 'fees', 'billing', 'price', 'pay'],
    'Cap on Liability': ['cap on liability', 'maximum liability', 'aggregate liability', 'liability limited'],
    'Liquidated Damages': ['liquidated damages', 'predetermined damages', 'fixed damages'],
    'Revenue/Profit Sharing': ['revenue sharing', 'profit sharing', 'share of revenue', 'percentage of'],
    'Price Restrictions': ['price', 'pricing', 'minimum price', 'maximum price', 'most favored'],
    'Minimum Commitment': ['minimum purchase', 'minimum commitment', 'minimum quantity', 'minimum volume'],
    'Volume Restriction': ['maximum volume', 'volume restriction', 'quantity restriction'],
    
    # Liability & Risk
    'Limitation of Liability': ['limitation of liability', 'limit liability', 'damages shall not exceed', 'not liable'],
    'Indemnification': ['indemnif', 'hold harmless', 'defend and indemnify', 'indemnity'],
    'Warranty Duration': ['warranty period', 'warranty shall', 'warranted for', 'guarantee period'],
    'Insurance': ['insurance', 'insure', 'coverage', 'insurance policy'],
    
    # Termination & Renewal
    'Termination': ['termination', 'terminate', 'cancellation', 'end of agreement'],
    'Termination for Convenience': ['terminate for convenience', 'without cause', 'at will'],
    'Renewal Term': ['renewal', 'renew', 'automatically renew', 'extend'],
    'Notice Period to Terminate Renewal': ['notice to terminate', 'notice prior to renewal', 'notice of non-renewal'],
    'Post-Termination Services': ['post-termination', 'after termination', 'wind down', 'transition'],
    
    # Restrictions & Competition
    'Non-Compete': ['non-compete', 'non-competition', 'compete', 'competitive'],
    'Exclusivity': ['exclusive', 'exclusivity', 'sole', 'only'],
    'No-Solicit of Customers': ['non-solicitation', 'no-solicit', 'not solicit customers'],
    'No-Solicit of Employees': ['no-solicit employees', 'not hire employees', 'non-solicitation of personnel'],
    'Non-Disparagement': ['non-disparagement', 'not disparage', 'no negative statements'],
    
    # IP & Confidentiality
    'Intellectual Property': ['intellectual property', 'patent', 'copyright', 'trademark', 'ip rights', 'ip'],
    'IP Ownership Assignment': ['assign', 'assignment of ip', 'ownership', 'work for hire'],
    'Joint IP Ownership': ['joint ownership', 'jointly own', 'co-ownership'],
    'License Grant': ['license', 'grant', 'right to use', 'licensed'],
    'Confidentiality': ['confidential', 'non-disclosure', 'proprietary information', 'trade secret'],
    
    # Change & Updates
    'Change of Control': ['change of control', 'change in ownership', 'acquisition', 'merger'],
    'Anti-Assignment': ['not assign', 'no assignment', 'assignment prohibited', 'anti-assignment'],
    'Covenant Not to Sue': ['covenant not to sue', 'agree not to sue', 'waive right to sue'],
    
    # Legal & Governance
    'Governing Law': ['governing law', 'choice of law', 'applicable law', 'laws of', 'governed by'],
    'Dispute Resolution': ['dispute', 'arbitration', 'mediation', 'litigation', 'resolve disputes'],
    'Jurisdiction': ['jurisdiction', 'venue', 'submit to jurisdiction', 'courts of'],
    'Notice': ['notice', 'notification', 'written notice', 'notify', 'give notice'],
    'Force Majeure': ['force majeure', 'act of god', 'beyond control', 'unforeseeable'],
    'Severability': ['severability', 'severable', 'invalid provision', 'unenforceable'],
    'Entire Agreement': ['entire agreement', 'whole agreement', 'complete agreement', 'supersedes'],
    'Amendment': ['amendment', 'modify', 'modification', 'change', 'amend'],
    'Waiver': ['waiver', 'waive', 'failure to enforce'],
    
    # Other
    'Third Party Beneficiary': ['third party beneficiary', 'third-party', 'benefit of'],
    'Audit Rights': ['audit', 'right to audit', 'inspect', 'examination of records'],
    'ROFR/ROFO/ROFN': ['right of first refusal', 'right of first offer', 'rofr', 'rofo', 'first right'],
    'Most Favored Nation': ['most favored nation', 'mfn', 'most favored customer'],
}

class LexiCacheModel:
    def __init__(self, projection_path="final_projection_head.pth", support_set_path="support_set.pkl"):
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
        self.support_set_path = support_set_path
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.75  # Trust model/keywords
        self.medium_confidence_threshold = 0.55  # Show but mark as uncertain
        self.low_confidence_threshold = 0.40  # Flag as unknown, ask user
        
        # Load persistent support set if exists
        self._load_support_set()

        print(f"[LexiCacheModel] Adaptive model ready. Unknown clause detection enabled.")
        print(f"  Device: {self.model.device}")
        print(f"  Support set size: {len(self.support_embeddings)} examples")
        print(f"  Known clause types: {len(CLAUSE_KEYWORDS)} (CUAD standard)")

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
        Classify a single text segment using hybrid approach:
        1. Keyword heuristics (fast, interpretable)
        2. Few-shot model (learned from user feedback)
        3. Flag low-confidence as unknown for user feedback
        """
        normalized = normalize_text(segment_text)
        
        # Get embedding for model-based classification
        with torch.no_grad():
            emb = self.model([normalized], batch_size=1)
            proj = self.projection(emb.to(self.model.device))
        
        # Strategy 1: Try keyword-based classification (fast and interpretable)
        kw_type, kw_conf = self._classify_by_keywords(segment_text)
        
        # Strategy 2: If we have a support set, use meta-learned model
        model_type, model_conf = None, 0.0
        if len(self.support_embeddings) > 0:
            support_emb = torch.stack(self.support_embeddings).to(self.model.device)
            dists = torch.cdist(proj, support_emb)
            pred_idx = dists.argmin().item()
            
            # Convert distance to confidence (closer = higher confidence)
            min_dist = dists[0, pred_idx].item()
            model_conf = max(0.0, 1.0 - (min_dist / 2.0))  # Normalize distance to confidence
            model_type = self.support_labels[pred_idx]
        
        # Decision logic: Choose best prediction
        final_type = None
        final_conf = 0.0
        source = 'unknown'
        
        # Prefer model if it's confident and exists
        if model_type and model_conf >= self.medium_confidence_threshold:
            if model_conf > kw_conf:
                final_type = model_type
                final_conf = model_conf
                source = 'model'
        
        # Fall back to keywords if model is not confident enough
        if not final_type and kw_type and kw_conf >= self.medium_confidence_threshold:
            final_type = kw_type
            final_conf = kw_conf
            source = 'keywords'
        
        # If both are low confidence, pick the better one but flag it
        if not final_type:
            if model_conf >= kw_conf and model_type:
                final_type = model_type
                final_conf = model_conf
                source = 'model_uncertain'
            elif kw_type:
                final_type = kw_type
                final_conf = kw_conf
                source = 'keywords_uncertain'
            else:
                # True unknown - needs user feedback
                final_type = 'Unknown Clause'
                final_conf = 0.40
                source = 'unknown'
        
        return {
            'clause_type': final_type,
            'confidence': final_conf,
            'source': source,
            'embedding': proj  # Store for potential learning
        }

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
                    
                    result_dict = {
                        'clause_type': classification['clause_type'],
                        'confidence': classification['confidence'],
                        'span': span_text,
                        'start_idx': seg['start_idx'],
                        'end_idx': seg['end_idx'],
                        'source': classification.get('source', 'unknown')
                    }
                    
                    # Store embedding for potential learning (convert to list for JSON serialization)
                    if 'embedding' in classification:
                        # Don't include in API response (too large), but could be stored server-side
                        pass
                    
                    results.append(result_dict)
                    seen_types[type_key] += 1
        
        # Sort by confidence descending
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 15 most confident clauses (more comprehensive)
        results = results[:15]
        
        # Mark clauses that need user review
        uncertain_count = 0
        for r in results:
            if r['confidence'] < self.high_confidence_threshold:
                r['needs_review'] = True
                uncertain_count += 1
            else:
                r['needs_review'] = False
        
        print(f"Identified {len(results)} clauses:")
        for r in results:
            confidence_emoji = "🟢" if r['confidence'] >= self.high_confidence_threshold else "🟡" if r['confidence'] >= self.medium_confidence_threshold else "🔴"
            print(f"  {confidence_emoji} {r['clause_type']}: {r['confidence']*100:.1f}% - \"{r['span'][:50]}...\"")
        
        if uncertain_count > 0:
            print(f"\n  ⚠️  {uncertain_count} clause(s) have uncertain predictions - user feedback recommended")
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

    def _load_support_set(self):
        """Load persistent support set from disk"""
        if Path(self.support_set_path).exists():
            try:
                with open(self.support_set_path, 'rb') as f:
                    data = pickle.load(f)
                    self.support_embeddings = data.get('embeddings', [])
                    self.support_labels = data.get('labels', [])
                    self.label_to_id = data.get('label_to_id', {})
                    self.next_label_id = data.get('next_label_id', 0)
                print(f"  ✓ Loaded {len(self.support_embeddings)} examples from persistent storage")
            except Exception as e:
                print(f"  ⚠ Failed to load support set: {e}")
    
    def _save_support_set(self):
        """Save support set to disk for persistence across sessions"""
        try:
            data = {
                'embeddings': self.support_embeddings,
                'labels': self.support_labels,
                'label_to_id': self.label_to_id,
                'next_label_id': self.next_label_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.support_set_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  ✓ Saved {len(self.support_embeddings)} examples to persistent storage")
        except Exception as e:
            print(f"  ⚠ Failed to save support set: {e}")
    
    def learn_from_feedback(self, clause_text: str, correct_label: str) -> bool:
        """
        Online meta-learning: User provides correct label for a clause.
        This immediately updates the support set.
        
        Args:
            clause_text: The text of the clause
            correct_label: The correct clause type (e.g., 'Termination')
        
        Returns:
            True if learning was successful
        """
        try:
            normalized = normalize_text(clause_text)
            
            # Get embedding
            with torch.no_grad():
                emb = self.model([normalized], batch_size=1)
                proj = self.projection(emb.to(self.model.device))
            
            # Add to support set
            if correct_label not in self.label_to_id:
                self.label_to_id[correct_label] = self.next_label_id
                self.next_label_id += 1
            
            self.support_embeddings.append(proj.squeeze(0).cpu())
            self.support_labels.append(correct_label)
            
            print(f"  ✓ Learned: '{correct_label}' (support set now has {len(self.support_embeddings)} examples)")
            
            # Save to disk every 5 new examples
            if len(self.support_embeddings) % 5 == 0:
                self._save_support_set()
            
            return True
        except Exception as e:
            print(f"  ✗ Failed to learn from feedback: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the model's knowledge"""
        label_counts = defaultdict(int)
        for label in self.support_labels:
            label_counts[label] += 1
        
        return {
            'total_examples': len(self.support_embeddings),
            'unique_types': len(self.label_to_id),
            'known_cuad_types': len(CLAUSE_KEYWORDS),
            'label_distribution': dict(label_counts),
            'device': str(self.model.device)
        }