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
    def __init__(self, projection_path="final_projection_head.pth", support_set_path="support_set.pkl", 
                 knowledge_path="clause_knowledge.json"):
        print("[LexiCacheModel] Loading adaptive meta-learning model...")
        self.model = PrototypicalNetwork()
        self.projection = nn.Linear(self.model.hidden_size, self.model.hidden_size).to(self.model.device)
        self.projection.load_state_dict(torch.load(projection_path, map_location=self.model.device))
        self.projection.eval()

        # Dynamic support set for online meta-learning
        self.support_embeddings = []      # list of embeddings
        self.support_labels = []          # list of clause type names (strings)
        self.support_texts = []           # list of original text examples
        self.label_to_id = {}             # map string label → local id
        self.next_label_id = 0
        self.support_set_path = support_set_path
        self.knowledge_path = knowledge_path
        
        # Confidence thresholds (aligned with dual-path requirement)
        self.keyword_high_threshold = 0.75  # Path A: Trust keyword matching
        self.model_high_threshold = 0.70    # Path B: Trust model prediction
        self.unknown_threshold = 0.60       # Below this = Unknown clause
        
        # Persistent knowledge base: learned types, colors, examples
        self.learned_types = {}  # {type_name: {color: str, examples: [str], count: int}}
        self.clause_colors = {}  # {clause_type: color_hex}
        
        # Load persistent data
        self._load_support_set()
        self._load_knowledge_base()

        print(f"[LexiCacheModel] Adaptive model ready. Unknown clause detection enabled.")
        print(f"  Device: {self.model.device}")
        print(f"  Support set size: {len(self.support_embeddings)} examples")
        print(f"  Known CUAD types: {len(CLAUSE_KEYWORDS)}")
        print(f"  Learned custom types: {len(self.learned_types)}")

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
        
        # Require at least 2 keyword matches for confident classification
        # Single keyword match is too weak and could be coincidental
        if best_match and best_score >= 2:
            # Confidence based on number of keyword matches
            confidence = min(0.95, 0.55 + (best_score * 0.10))
            return best_match, confidence
        elif best_match and best_score == 1:
            # Single keyword match - very low confidence
            return best_match, 0.45
        
        return None, 0.0

    def _classify_segment(self, segment_text: str) -> Dict:
        """
        DUAL-PATH CLASSIFICATION:
        Path A: Keyword/heuristic matching (fast, rule-based)
        Path B: Model-based few-shot (learned from feedback)
        
        Decision rule:
        - If Path A confidence ≥ 0.75 → use it
        - Else if Path B confidence ≥ 0.70 → use it  
        - Else → mark as "Unknown clause"
        """
        normalized = normalize_text(segment_text)
        
        # Get embedding for model-based classification
        with torch.no_grad():
            emb = self.model([normalized], batch_size=1)
            proj = self.projection(emb.to(self.model.device))
        
        # PATH A: Keyword-based classification (fast, interpretable)
        kw_type, kw_conf = self._classify_by_keywords(segment_text)
        
        # PATH B: Model-based few-shot classification
        model_type, model_conf = None, 0.0
        if len(self.support_embeddings) > 0:
            support_emb = torch.stack(self.support_embeddings).to(self.model.device)
            dists = torch.cdist(proj, support_emb)
            pred_idx = dists.argmin().item()
            
            # Convert distance to confidence (closer = higher confidence)
            min_dist = dists[0, pred_idx].item()
            model_conf = max(0.0, 1.0 - (min_dist / 2.0))  # Normalize distance to confidence
            model_type = self.support_labels[pred_idx]
        
        # DECISION RULE: Apply thresholds per requirements
        final_type = None
        final_conf = 0.0
        source = 'unknown'
        
        # Priority 1: Path A (keywords) with high confidence
        if kw_type and kw_conf >= self.keyword_high_threshold:
            final_type = kw_type
            final_conf = kw_conf
            source = 'keywords'
        # Priority 2: Path B (model) with high confidence
        elif model_type and model_conf >= self.model_high_threshold:
            final_type = model_type
            final_conf = model_conf
            source = 'model'
        # Priority 3: Medium confidence predictions (use but mark uncertain)
        elif kw_type and kw_conf >= 0.50:
            final_type = kw_type
            final_conf = kw_conf
            source = 'keywords_uncertain'
        elif model_type and model_conf >= 0.50:
            final_type = model_type
            final_conf = model_conf
            source = 'model_uncertain'
        
        # If no confident prediction → Unknown clause (special category)
        if not final_type:
            final_type = 'Unknown clause'
            final_conf = 0.45  # Fixed low confidence for unknowns
            source = 'unknown'
        
        return {
            'clause_type': final_type,
            'confidence': final_conf,
            'source': source,
            'embedding': proj  # Store for potential learning
        }

    def predict_cuad(self, contract_text: str, confidence_threshold: float = 0.40) -> List[Dict]:
        """
        Main prediction - extracts and classifies ALL clauses from contract.
        NOW INCLUDES "Unknown clause" category for low-confidence segments.
        Returns ALL detected segments (not just top 15).
        """
        print(f"\n{'='*85}")
        print("🧠 LEXICACHE MULTI-CLAUSE EXTRACTION (with Unknown Detection)")
        print(f"{'='*85}")
        print(f"Contract length: {len(contract_text)} characters")
        
        # Segment the contract
        segments = self._segment_contract(contract_text)
        print(f"Found {len(segments)} potential clause segments")
        
        results = []
        seen_types = defaultdict(int)  # Track count of each clause type
        unknown_count = 0
        
        for seg in segments:
            classification = self._classify_segment(seg['text'])
            
            # ALWAYS include segments (even unknowns) - never discard
            # Only exclude very low confidence noise
            if classification['confidence'] >= confidence_threshold:
                type_key = classification['clause_type']
                
                # Allow more unknowns (they need user feedback)
                max_per_type = 10 if type_key == 'Unknown clause' else 3
                
                if seen_types[type_key] < max_per_type:
                    # Clean up the span text for better matching
                    span_text = seg['text'][:300] if len(seg['text']) > 300 else seg['text']
                    span_text = ' '.join(span_text.split())  # Normalize whitespace
                    
                    result_dict = {
                        'clause_type': classification['clause_type'],
                        'confidence': classification['confidence'],
                        'span': span_text,
                        'start_idx': seg['start_idx'],
                        'end_idx': seg['end_idx'],
                        'source': classification.get('source', 'unknown'),
                        'is_unknown': type_key == 'Unknown clause'
                    }
                    
                    if type_key == 'Unknown clause':
                        unknown_count += 1
                    
                    # Store embedding for potential learning (convert to list for JSON serialization)
                    if 'embedding' in classification:
                        # Don't include in API response (too large), but could be stored server-side
                        pass
                    
                    results.append(result_dict)
                    seen_types[type_key] += 1
        
        # Sort: Unknown clauses last, then by confidence descending
        results.sort(key=lambda x: (x['is_unknown'], -x['confidence']))
        
        # NO LIMIT - return ALL detected clauses (including unknowns)
        # Frontend will handle display/filtering
        
        # Mark clauses that need user review
        needs_review_count = 0
        for r in results:
            if r['is_unknown'] or r['confidence'] < self.model_high_threshold:
                r['needs_review'] = True
                needs_review_count += 1
            else:
                r['needs_review'] = False
        
        print(f"Identified {len(results)} total clauses:")
        known_count = len(results) - unknown_count
        print(f"  ✓ {known_count} known clause types")
        print(f"  ❓ {unknown_count} unknown clauses (need user teaching)")
        
        if unknown_count > 0:
            print(f"\n  💡 Tip: Teach the system by renaming 'Unknown clause' items")
        print(f"{'='*85}\n")
        
        # Always return results (even if empty - frontend handles it)
        return results

    def _load_support_set(self):
        """Load persistent support set from disk"""
        if Path(self.support_set_path).exists():
            try:
                with open(self.support_set_path, 'rb') as f:
                    data = pickle.load(f)
                    self.support_embeddings = data.get('embeddings', [])
                    self.support_labels = data.get('labels', [])
                    self.support_texts = data.get('texts', [])
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
                'texts': self.support_texts,
                'label_to_id': self.label_to_id,
                'next_label_id': self.next_label_id,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.support_set_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  ✓ Saved {len(self.support_embeddings)} examples to persistent storage")
        except Exception as e:
            print(f"  ⚠ Failed to save support set: {e}")
    
    def _load_knowledge_base(self):
        """Load learned clause types, colors, and examples from JSON"""
        if Path(self.knowledge_path).exists():
            try:
                with open(self.knowledge_path, 'r') as f:
                    data = json.load(f)
                    self.learned_types = data.get('learned_types', {})
                    self.clause_colors = data.get('clause_colors', {})
                    
                    # Rebuild support set from saved examples for multi-user persistence
                    learned_examples = data.get('learned_examples', [])
                    if learned_examples:
                        print(f"  🔄 Rebuilding support set from {len(learned_examples)} saved examples...")
                        for example in learned_examples:
                            clause_type = example['clause_type']
                            text = example['text']
                            
                            # Generate embedding for this example
                            try:
                                normalized = normalize_text(text)
                                with torch.no_grad():
                                    emb = self.model([normalized], batch_size=1)
                                    proj = self.projection(emb.to(self.model.device))
                                
                                # Add to support set
                                if clause_type not in self.label_to_id:
                                    self.label_to_id[clause_type] = self.next_label_id
                                    self.next_label_id += 1
                                
                                self.support_embeddings.append(proj.squeeze(0).cpu())
                                self.support_labels.append(clause_type)
                                self.support_texts.append(text)
                            except Exception as e:
                                print(f"    ⚠ Failed to rebuild embedding for {clause_type}: {e}")
                        
                        print(f"  ✓ Rebuilt support set with {len(self.support_embeddings)} examples")
                
                print(f"  ✓ Loaded {len(self.learned_types)} learned clause types")
            except Exception as e:
                print(f"  ⚠ Failed to load knowledge base: {e}")
    
    def _save_knowledge_base(self):
        """Save learned clause types and colors to JSON for multi-user persistence"""
        try:
            # Build list of learned examples (text + type) for rebuilding support set
            learned_examples = []
            for i, label in enumerate(self.support_labels):
                if i < len(self.support_texts):
                    learned_examples.append({
                        'clause_type': label,
                        'text': self.support_texts[i]
                    })
            
            data = {
                'learned_types': self.learned_types,
                'clause_colors': self.clause_colors,
                'learned_examples': learned_examples,  # KEY: Save actual examples for rebuilding
                'timestamp': datetime.now().isoformat(),
                'total_learned': len(self.learned_types),
                'total_examples': len(learned_examples)
            }
            with open(self.knowledge_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ Saved knowledge base ({len(self.learned_types)} types, {len(learned_examples)} examples)")
        except Exception as e:
            print(f"  ⚠ Failed to save knowledge base: {e}")
    
    def learn_from_feedback(self, clause_text: str, correct_label: str, color: str = None) -> bool:
        """
        Online meta-learning: User teaches the system a new clause type.
        Used when renaming "Unknown clause" → custom type.
        
        Args:
            clause_text: The text of the clause
            correct_label: The correct clause type (e.g., 'Escrow Provision')
            color: Optional hex color for this type (e.g., '#FF5733')
        
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
                
                # Track as new learned type
                if correct_label not in self.learned_types:
                    self.learned_types[correct_label] = {
                        'examples': [],
                        'count': 0,
                        'first_learned': datetime.now().isoformat()
                    }
            
            self.support_embeddings.append(proj.squeeze(0).cpu())
            self.support_labels.append(correct_label)
            self.support_texts.append(clause_text)  # Store full text for persistence
            
            # Update learned types tracking
            self.learned_types[correct_label]['count'] += 1
            self.learned_types[correct_label]['examples'].append(clause_text[:200])  # Store snippet
            
            # Store color if provided
            if color:
                self.clause_colors[correct_label] = color
            
            print(f"  ✓ Learned: '{correct_label}' (support set now has {len(self.support_embeddings)} examples)")
            
            # Save both support set and knowledge base
            self._save_support_set()
            self._save_knowledge_base()
            
            return True
        except Exception as e:
            print(f"  ✗ Failed to learn from feedback: {e}")
            return False
    
    def rename_unknown_clause(self, clause_text: str, old_span: str, new_type_name: str, color: str = None) -> Dict:
        """
        Rename an "Unknown clause" to a user-defined type.
        This teaches the model the new clause type.
        
        Args:
            clause_text: Full contract text
            old_span: The original span text that was marked as Unknown
            new_type_name: User's name for this clause type
            color: User-chosen color for this type
        
        Returns:
            Updated classification results
        """
        # Teach the model
        success = self.learn_from_feedback(old_span, new_type_name, color)
        
        if success:
            # Re-classify the entire contract with new knowledge
            return self.predict_cuad(clause_text)
        
        return None
    
    def update_clause_color(self, clause_type: str, color: str) -> bool:
        """
        Update the color for a specific clause type.
        
        Args:
            clause_type: Name of the clause type
            color: Hex color code (e.g., '#FF5733')
        
        Returns:
            True if successful
        """
        try:
            self.clause_colors[clause_type] = color
            self._save_knowledge_base()
            return True
        except Exception as e:
            print(f"  ✗ Failed to update color: {e}")
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
            'learned_custom_types': len(self.learned_types),
            'label_distribution': dict(label_counts),
            'learned_types_detail': self.learned_types,
            'clause_colors': self.clause_colors,
            'device': str(self.model.device)
        }
    
    def get_all_clause_types_with_colors(self) -> Dict:
        """
        Get all known clause types with their assigned colors.
        Used by frontend to build legend.
        
        Returns:
            {clause_type: color_hex}
        """
        all_types = {}
        
        # Add CUAD standard types (will get auto-generated colors in frontend if not set)
        for clause_type in CLAUSE_KEYWORDS.keys():
            all_types[clause_type] = self.clause_colors.get(clause_type, None)
        
        # Add learned custom types
        for clause_type in self.learned_types.keys():
            all_types[clause_type] = self.clause_colors.get(clause_type, None)
        
        # Special: Unknown clause gets fixed color
        all_types['Unknown clause'] = self.clause_colors.get('Unknown clause', '#9CA3AF')  # Gray
        
        return all_types