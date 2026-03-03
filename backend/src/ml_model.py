"""
Adaptive meta-learning model for LexiCache.
Supports online adaptation: detects unknown clauses, accepts user labels, and meta-learns in real-time.
"""

import torch
import torch.nn as nn
import numpy as np
import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, cast
from collections import defaultdict
from datetime import datetime

from src.modeling import PrototypicalNetwork
from src.data import normalize_text

# Heading patterns used in the contract segmenter
_HEADING_LINE_PATTERNS = [
    re.compile(r'^\s*(ARTICLE|SECTION|CHAPTER|PART|SCHEDULE|EXHIBIT|ANNEX)\s+[IVXLCDM\d]+', re.IGNORECASE),
    re.compile(r'^\s*\d+(\.\d+)*\.?\s+[A-Z]'),                          # 1. Title, 1.2 Title
    re.compile(r'^\s*[IVXLCDM]+\.\s+[A-Z]'),                            # IV. Title
    re.compile(r'^\s*\([a-z0-9ivx]+\)\s+[A-Z]'),                        # (a) / (iv) Title
    re.compile(r'^\s*[A-Z][A-Z\s\-]{3,}$'),                             # ALL CAPS HEADING
    re.compile(r'^\s*[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\s*:\s*$'),        # Title Case Heading:
    re.compile(r'^\s*[A-Z][A-Za-z\s\-&,]{2,60}\s*:\s*$'),              # Mixed case heading ending in colon
]

# Weighted clause keywords: { clause_type: [(keyword, weight), ...] }
# weight=2 for specific multi-word phrases, weight=1 for generic terms
CLAUSE_KEYWORDS_WEIGHTED = {
    # Core Agreement Terms
    'Document Name': [
        ('this agreement', 2), ('master agreement', 2), ('master services agreement', 2),
        ('service agreement', 2), ('purchase agreement', 2), ('license agreement', 2),
        ('software agreement', 2), ('agreement', 1), ('contract', 1),
    ],
    'Parties': [
        ('by and between', 2), ('hereinafter referred to as', 2), ('the parties', 2),
        ('collectively referred to', 2), ('hereinafter called', 2),
        ('party', 1), ('parties', 1), ('hereinafter', 1),
    ],
    'Agreement Date': [
        ('effective as of', 2), ('dated as of', 2), ('entered into as of', 2),
        ('this agreement is dated', 2), ('agreement date', 2),
        ('dated', 1), ('as of', 1), ('effective date', 1),
    ],
    'Effective Date': [
        ('effective date shall be', 2), ('commence on', 2), ('commencement date', 2),
        ('shall become effective', 2), ('takes effect', 2),
        ('effective', 1), ('start date', 1), ('begin', 1),
    ],
    'Expiration Date': [
        ('term shall expire', 2), ('expiration date', 2), ('term end date', 2),
        ('agreement shall terminate on', 2), ('valid until', 2),
        ('expire', 1), ('expiration', 1), ('until', 1),
    ],

    # Financial and Payment
    'Payment Terms': [
        ('payment shall be due', 2), ('invoice date', 2), ('net 30', 2), ('net 60', 2),
        ('due date', 2), ('payment schedule', 2), ('milestone payment', 2),
        ('retainer fee', 2), ('billing cycle', 2), ('late payment', 2),
        ('payment', 1), ('invoice', 1), ('compensation', 1), ('fees', 1), ('pay', 1),
    ],
    'Cap on Liability': [
        ('cap on liability', 2), ('maximum aggregate liability', 2),
        ('liability shall not exceed', 2), ('aggregate damages', 2),
        ('liability cap', 2), ('maximum liability', 2),
        ('aggregate liability', 1), ('liability limited', 1),
    ],
    'Liquidated Damages': [
        ('liquidated damages', 2), ('predetermined damages', 2),
        ('fixed damages', 2), ('damages clause', 2), ('agreed damages', 2),
        ('liquidated', 1), ('penalty clause', 1),
    ],
    'Revenue/Profit Sharing': [
        ('revenue sharing', 2), ('profit sharing', 2), ('revenue split', 2),
        ('share of revenues', 2), ('percentage of net revenue', 2),
        ('revenue share', 2), ('profit split', 2),
        ('percentage of', 1), ('split', 1),
    ],
    'Price Restrictions': [
        ('most favored customer', 2), ('most favored nation pricing', 2),
        ('minimum resale price', 2), ('price floor', 2), ('price ceiling', 2),
        ('list price', 2), ('price restriction', 2),
        ('price', 1), ('pricing', 1), ('minimum price', 1), ('maximum price', 1),
    ],
    'Minimum Commitment': [
        ('minimum purchase commitment', 2), ('minimum order quantity', 2),
        ('minimum volume commitment', 2), ('minimum annual', 2),
        ('minimum purchase', 1), ('minimum commitment', 1), ('minimum quantity', 1),
    ],
    'Volume Restriction': [
        ('maximum volume', 2), ('volume restriction', 2), ('quantity restriction', 2),
        ('purchase limit', 2), ('order cap', 2),
        ('volume cap', 1), ('maximum quantity', 1),
    ],

    # Liability and Risk
    'Limitation of Liability': [
        ('in no event shall', 2), ('shall not be liable for', 2),
        ('limitation of liability', 2), ('limit on liability', 2),
        ('damages shall not exceed', 2), ('exclude all liability', 2),
        ('consequential damages', 2), ('indirect damages', 2),
        ('limitation of liability', 1), ('not liable', 1),
    ],
    'Indemnification': [
        ('shall indemnify', 2), ('indemnify and hold harmless', 2),
        ('defend and indemnify', 2), ('indemnification obligation', 2),
        ('losses and damages', 2), ('indemnification clause', 2),
        ('indemnif', 1), ('hold harmless', 1), ('indemnity', 1),
    ],
    'Warranty Duration': [
        ('warranty period', 2), ('warranty shall remain', 2),
        ('warranted for a period', 2), ('guarantee period', 2),
        ('warranty term', 2), ('defect warranty', 2),
        ('warranty shall', 1), ('warranted for', 1), ('warranty', 1),
    ],
    'Insurance': [
        ('maintain insurance', 2), ('general liability insurance', 2),
        ('professional liability insurance', 2), ('insurance coverage', 2),
        ('insurance policy', 2), ('certificate of insurance', 2),
        ('insurance', 1), ('insure', 1), ('coverage', 1),
    ],

    # Termination and Renewal
    'Termination': [
        ('may terminate', 2), ('notice of termination', 2), ('material breach', 2),
        ('cure period', 2), ('right to terminate', 2), ('immediately upon termination', 2),
        ('termination rights', 2), ('agreement shall terminate', 2),
        ('termination', 1), ('terminate', 1), ('cancellation', 1),
    ],
    'Termination for Convenience': [
        ('terminate for convenience', 2), ('without cause termination', 2),
        ('at will termination', 2), ('terminate without reason', 2),
        ('terminate for any reason', 2), ('convenience termination', 2),
        ('without cause', 1), ('at will', 1),
    ],
    'Renewal Term': [
        ('automatically renew', 2), ('automatic renewal', 2),
        ('renewal term', 2), ('successive term', 2), ('auto-renewal', 2),
        ('evergreen clause', 2), ('rollover', 2),
        ('renewal', 1), ('renew', 1), ('extend', 1),
    ],
    'Notice Period to Terminate Renewal': [
        ('prior written notice to terminate', 2), ('notice of non-renewal', 2),
        ('days notice prior to renewal', 2), ('terminate renewal', 2),
        ('renewal notice period', 2),
        ('notice to terminate', 1), ('notice prior to renewal', 1),
    ],
    'Post-Termination Services': [
        ('post-termination obligations', 2), ('wind-down services', 2),
        ('transition assistance', 2), ('after termination', 2),
        ('following termination', 2), ('survival of obligations', 2),
        ('post-termination', 1), ('wind down', 1), ('transition', 1),
    ],

    # Restrictions and Competition
    'Non-Compete': [
        ('non-compete agreement', 2), ('covenant not to compete', 2),
        ('shall not compete', 2), ('competing business', 2),
        ('competitive activity', 2), ('competitive services', 2),
        ('non-compete', 1), ('non-competition', 1), ('compete', 1),
    ],
    'Exclusivity': [
        ('exclusive right', 2), ('exclusive license', 2), ('exclusive provider', 2),
        ('sole and exclusive', 2), ('exclusivity period', 2),
        ('exclusive basis', 2), ('exclusive territory', 2),
        ('exclusive', 1), ('exclusivity', 1), ('sole', 1),
    ],
    'No-Solicit of Customers': [
        ('not solicit customers', 2), ('customer non-solicitation', 2),
        ('shall not solicit any client', 2), ('solicitation of customers', 2),
        ('non-solicitation of customers', 2),
        ('non-solicitation', 1), ('no-solicit', 1),
    ],
    'No-Solicit of Employees': [
        ('not solicit employees', 2), ('employee non-solicitation', 2),
        ('shall not hire', 2), ('non-solicitation of personnel', 2),
        ('poaching restriction', 2), ('non-solicitation of employees', 2),
        ('non-solicitation of personnel', 1), ('not hire employees', 1),
    ],
    'Non-Disparagement': [
        ('non-disparagement', 2), ('shall not disparage', 2),
        ('no negative statements', 2), ('refrain from making negative', 2),
        ('disparage', 1), ('defamatory', 1),
    ],

    # IP and Confidentiality
    'Intellectual Property': [
        ('intellectual property rights', 2), ('all intellectual property', 2),
        ('patent rights', 2), ('copyright ownership', 2), ('trademark rights', 2),
        ('trade secret', 2), ('ip rights', 2), ('proprietary rights', 2),
        ('intellectual property', 1), ('patent', 1), ('copyright', 1), ('ip', 1),
    ],
    'IP Ownership Assignment': [
        ('assigns all rights', 2), ('assignment of intellectual property', 2),
        ('work for hire', 2), ('work made for hire', 2), ('ip assignment', 2),
        ('transfer of ownership', 2), ('all right title and interest', 2),
        ('work product', 2), ('moral rights', 2), ('derivative works', 2),
        ('assign', 1), ('ownership', 1),
    ],
    'Joint IP Ownership': [
        ('joint ownership', 2), ('jointly own', 2), ('co-ownership of ip', 2),
        ('joint inventors', 2), ('shared ownership', 2),
        ('joint ownership', 1), ('co-ownership', 1),
    ],
    'License Grant': [
        ('hereby grants', 2), ('grants a license', 2), ('non-exclusive license', 2),
        ('exclusive license to', 2), ('royalty-free license', 2),
        ('sublicensable license', 2), ('right to use the software', 2),
        ('source code license', 2),
        ('license', 1), ('grant', 1), ('right to use', 1), ('licensed', 1),
    ],
    'Confidentiality': [
        ('confidential information', 2), ('non-disclosure agreement', 2),
        ('shall keep confidential', 2), ('disclose to third parties', 2),
        ('return of confidential materials', 2), ('disclosure restrictions', 2),
        ('nda', 2), ('proprietary information', 2), ('trade secret', 2),
        ('confidential', 1), ('non-disclosure', 1),
    ],

    # Change and Assignments
    'Change of Control': [
        ('change of control', 2), ('change in ownership', 2),
        ('merger or acquisition', 2), ('acquisition of control', 2),
        ('control event', 2), ('change-of-control provision', 2),
        ('change of control', 1), ('acquisition', 1), ('merger', 1),
    ],
    'Anti-Assignment': [
        ('shall not assign', 2), ('assignment without consent', 2),
        ('may not assign', 2), ('assignment prohibited', 2),
        ('non-assignable', 2), ('anti-assignment clause', 2),
        ('not assign', 1), ('no assignment', 1),
    ],
    'Covenant Not to Sue': [
        ('covenant not to sue', 2), ('agrees not to sue', 2),
        ('waives right to sue', 2), ('releases claims', 2),
        ('no right to sue', 2),
        ('covenant not to sue', 1), ('agree not to sue', 1),
    ],

    # Legal and Governance
    'Governing Law': [
        ('shall be governed by', 2), ('this agreement is governed by', 2),
        ('laws of the state of', 2), ('choice of law', 2),
        ('applicable law', 2), ('state law', 2), ('federal law', 2),
        ('governing law', 1), ('laws of', 1), ('governed by', 1),
    ],
    'Dispute Resolution': [
        ('binding arbitration', 2), ('submit to arbitration', 2),
        ('dispute resolution process', 2), ('resolve disputes', 2),
        ('mandatory mediation', 2), ('alternative dispute resolution', 2),
        ('adr', 2), ('arbitration clause', 2),
        ('dispute', 1), ('arbitration', 1), ('mediation', 1),
    ],
    'Jurisdiction': [
        ('consent to jurisdiction', 2), ('exclusive jurisdiction', 2),
        ('courts of competent jurisdiction', 2), ('submit to jurisdiction', 2),
        ('venue shall be', 2), ('choice of forum', 2),
        ('jurisdiction', 1), ('venue', 1),
    ],
    'Notice': [
        ('written notice shall be', 2), ('notice shall be given', 2),
        ('notice shall be sent to', 2), ('receipt of notice', 2),
        ('notice of default', 2), ('notice period', 2),
        ('notice', 1), ('notification', 1), ('written notice', 1), ('notify', 1),
    ],
    'Force Majeure': [
        ('force majeure', 2), ('act of god', 2), ('beyond reasonable control', 2),
        ('unforeseeable circumstances', 2), ('natural disaster', 2),
        ('government action', 2), ('pandemic', 2), ('epidemic', 2),
        ('force majeure', 1), ('beyond control', 1),
    ],
    'Severability': [
        ('if any provision', 2), ('severability clause', 2),
        ('should any provision be found', 2), ('unenforceable provision', 2),
        ('shall be deemed severable', 2), ('remaining provisions', 2),
        ('severability', 1), ('severable', 1), ('invalid provision', 1),
    ],
    'Entire Agreement': [
        ('entire agreement between', 2), ('supersedes all prior agreements', 2),
        ('complete agreement', 2), ('entire understanding', 2),
        ('all prior representations', 2), ('integration clause', 2),
        ('entire agreement', 1), ('whole agreement', 1), ('supersedes', 1),
    ],
    'Amendment': [
        ('may be amended', 2), ('modifications must be in writing', 2),
        ('written agreement to modify', 2), ('no amendment unless', 2),
        ('amendment shall be valid', 2),
        ('amendment', 1), ('modify', 1), ('modification', 1), ('amend', 1),
    ],
    'Waiver': [
        ('failure to exercise', 2), ('waiver of any right', 2),
        ('shall not constitute a waiver', 2), ('no waiver by', 2),
        ('waiver is not a continuing waiver', 2),
        ('waiver', 1), ('waive', 1), ('failure to enforce', 1),
    ],

    # Other and Miscellaneous
    'Third Party Beneficiary': [
        ('no third party beneficiaries', 2), ('intended beneficiary', 2),
        ('third-party rights', 2), ('benefit of third parties', 2),
        ('third party beneficiary', 1),
    ],
    'Audit Rights': [
        ('right to audit', 2), ('audit rights', 2), ('books and records', 2),
        ('may inspect', 2), ('examination of records', 2),
        ('financial audit', 2), ('audit', 1), ('inspect', 1),
    ],
    'ROFR/ROFO/ROFN': [
        ('right of first refusal', 2), ('right of first offer', 2),
        ('right of first negotiation', 2), ('rofr', 2), ('rofo', 2), ('rofn', 2),
        ('first right', 1), ('preemptive right', 1),
    ],
    'Most Favored Nation': [
        ('most favored nation', 2), ('mfn clause', 2), ('most favored customer', 2),
        ('most-favored treatment', 2), ('mfn', 1),
    ],
    # LEDGAR common provisions (not in CUAD 41)
    'Representations and Warranties': [
        ('represents and warrants', 2), ('representations and warranties', 2),
        ('as-is disclaimer', 2), ('no representation', 2),
        ('represents', 1), ('warrants', 1), ('warranty', 1),
    ],
    'Definitions': [
        ('as used in this agreement', 2), ('for purposes of this agreement', 2),
        ('the following terms shall have', 2), ('defined terms', 2),
        ('definitions section', 2), ('"defined term"', 2),
        ('means', 1), ('defined as', 1), ('definition', 1),
    ],
    'Assignment': [
        ('may assign this agreement', 2), ('assigning party', 2),
        ('assigns its rights', 2), ('delegation of duties', 2),
        ('assignment of agreement', 1), ('assign', 1),
    ],
    'Counterparts': [
        ('may be executed in counterparts', 2), ('electronic signature', 2),
        ('facsimile signature', 2), ('deemed original', 2),
        ('counterpart', 1), ('same instrument', 1),
    ],
    'Relationship of Parties': [
        ('independent contractor', 2), ('no partnership', 2),
        ('no joint venture', 2), ('no agency relationship', 2),
        ('not an employee', 2), ('relationship of the parties', 2),
        ('independent contractor', 1), ('employer-employee', 1),
    ],
}

# Flat CLAUSE_KEYWORDS dict (for backward compatibility with existing API endpoint)
CLAUSE_KEYWORDS = {
    k: [kw for kw, _ in pairs]
    for k, pairs in CLAUSE_KEYWORDS_WEIGHTED.items()
}


# LexiCacheModel

class LexiCacheModel:
    def __init__(self, projection_path="final_projection_head.pth", support_set_path="support_set.pkl",
                 knowledge_path="clause_knowledge.json"):
        print("[LexiCacheModel] Loading adaptive meta-learning model...")
        self.model = PrototypicalNetwork()
        self.projection = nn.Linear(self.model.hidden_size, self.model.hidden_size).to(self.model.device)
        self.projection.load_state_dict(torch.load(projection_path, map_location=self.model.device))
        self.projection.eval()

        # Dynamic support set for online meta-learning
        self.support_embeddings: List[Any] = []
        self.support_labels: List[str] = []
        self.support_texts: List[str] = []
        self.label_to_id: Dict[str, int] = {}
        self.next_label_id: int = 0
        self.support_set_path: str = support_set_path
        self.knowledge_path: str = knowledge_path

        # Classification thresholds
        self.keyword_high_threshold: float = 0.85
        self.model_high_threshold: float = 0.75
        self.unknown_threshold: float = 0.35

        # Hybrid weighting: keywords are more reliable for CUAD
        self.kw_weight: float = 0.70
        self.model_weight: float = 0.30

        # Persistent knowledge base
        self.learned_types: Dict[str, Dict[str, Any]] = {}
        self.clause_colors: Dict[str, str] = {}

        self._load_support_set()
        self._load_knowledge_base()

        print(f"[LexiCacheModel] Adaptive model ready. Unknown clause detection enabled.")
        print(f"  Device: {self.model.device}")
        print(f"  Support set size: {len(self.support_embeddings)} examples")
        print(f"  Known CUAD types: {len(CLAUSE_KEYWORDS_WEIGHTED)}")
        print(f"  Learned custom types: {len(self.learned_types)}")

    # Segmentation

    @staticmethod
    def _classify_line(line: str) -> str:
        """Classify a single line as 'HEADING', 'BLANK', or 'BODY'."""
        stripped = line.strip()
        if not stripped:
            return 'BLANK'
        for pattern in _HEADING_LINE_PATTERNS:
            if pattern.match(stripped):
                return 'HEADING'
        # Short with no sentence-ending punctuation → likely heading
        if len(stripped) <= 80 and not any(stripped.endswith(p) for p in ['.', '!', '?', ';', ',']):
            alpha = [c for c in stripped if c.isalpha()]
            if alpha:
                caps_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
                if caps_ratio >= 0.65:
                    return 'HEADING'
            if len(stripped.split()) <= 6:
                return 'HEADING'
        return 'BODY'

    def _segment_contract(self, text: str) -> List[Dict[str, Any]]:
        """
        Legal-specific segmenter:
        1. Classify every line as HEADING / BLANK / BODY.
        2. Group consecutive BODY lines into paragraph blocks.
        3. Merge small adjacent body blocks (< 80 chars) into the next block.
        4. Attach preceding heading text as `context_heading` to each segment.
        5. Further split very long paragraphs on sentence boundaries.
        """
        lines = text.splitlines()
        tagged = [(line, self._classify_line(line)) for line in lines]

        # --- Group into raw blocks -------------------------------------------
        blocks: List[Dict[str, Any]] = []   # list of {'type': str, 'lines': [str], 'start_idx': int}
        current_type = None
        current_lines = []
        char_pos = 0
        start_idx = 0

        for raw_line, tag in tagged:
            if tag != current_type:
                if current_lines:
                    blocks.append({
                        'type': current_type,
                        'lines': list(current_lines),
                        'start_idx': start_idx,
                    })
                current_type = tag
                current_lines = [raw_line]
                start_idx = char_pos
            else:
                current_lines.append(raw_line)
            char_pos += len(raw_line) + 1  # +1 for newline

        if current_lines:
            blocks.append({'type': current_type, 'lines': current_lines, 'start_idx': start_idx})

        # --- Merge tiny BODY blocks upstream ----------------------------------
        merged_blocks = []
        i = 0
        while i < len(blocks):
            blk: Dict[str, Any] = blocks[i]
            blk_type: str = cast(str, blk['type'])
            blk_lines: List[str] = cast(List[str], blk['lines'])
            if blk_type == 'BODY':
                body_text = '\n'.join(blk_lines).strip()
                # If this block is very short, peek ahead and merge with next BODY block
                if len(body_text) < 80 and i + 1 < len(blocks):
                    j = i + 1
                    # Skip over BLANK blocks
                    while j < len(blocks) and cast(str, blocks[j]['type']) == 'BLANK':
                        j += 1
                    if j < len(blocks) and cast(str, blocks[j]['type']) == 'BODY':
                        next_lines: List[str] = cast(List[str], blocks[j]['lines'])
                        merged_lines: List[str] = blk_lines + [''] + next_lines
                        blocks[j] = {
                            'type': 'BODY',
                            'lines': merged_lines,
                            'start_idx': blk['start_idx'],
                        }
                        i += 1
                        continue
            merged_blocks.append(blk)
            i += 1

        # --- Build segments with heading context ------------------------------
        segments = []
        last_heading_text = ''
        char_pos = 0

        for blk in merged_blocks:
            raw_text = '\n'.join(blk['lines'])
            stripped = raw_text.strip()

            if blk['type'] == 'HEADING':
                # Strip out empty lines within the heading block
                heading_lines = [l.strip() for l in blk['lines'] if l.strip()]
                last_heading_text = ' '.join(heading_lines)
                char_pos += len(raw_text) + 1
                continue  # headings are NOT added as classifiable segments

            if blk['type'] == 'BLANK':
                char_pos += len(raw_text) + 1
                continue

            # BODY block
            if len(stripped) < 30:
                # Include context heading in classification if body is short but heading is meaningful
                combined_text = f"{last_heading_text} {stripped}".strip() if last_heading_text else stripped
                if len(combined_text) >= 30:
                    # Use combined heading + body for better context
                    start_idx = blk['start_idx']
                    end_idx = start_idx + len(stripped)
                    
                    segments.append({
                        'text': combined_text,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'context_heading': last_heading_text,
                    })
                char_pos += len(raw_text) + 1
                continue

            start_idx = blk['start_idx']
            end_idx = start_idx + len(stripped)

            # Further split very long paragraphs by sentence boundaries
            if len(stripped) > 600:
                sentences = re.split(r'(?<=[.!?;])\s+(?=[A-Z\(\"])', stripped)
                sent_pos = start_idx
                for sent in sentences:
                    sent_stripped = sent.strip()
                    if len(sent_stripped) >= 30 and not self._is_heading(sent_stripped):
                        segments.append({
                            'text': sent_stripped,
                            'start_idx': sent_pos,
                            'end_idx': sent_pos + len(sent_stripped),
                            'context_heading': last_heading_text,
                        })
                    sent_pos += len(sent) + 1
            else:
                segments.append({
                    'text': stripped,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'context_heading': last_heading_text,
                })

            char_pos += len(raw_text) + 1

        return segments

    def _is_heading(self, text: str) -> bool:
        """
        Detect if a segment is likely a heading/subheading/title.
        Returns True if it's a heading (should NOT be classified as a clause).
        """
        text_stripped = text.strip()

        if len(text_stripped) < 15:
            return True
        if len(text_stripped) > 200:
            return False

        for pattern in _HEADING_LINE_PATTERNS:
            if pattern.match(text_stripped):
                return True

        if text_stripped.endswith(':'):
            return True

        alpha_chars = [c for c in text_stripped if c.isalpha()]
        if alpha_chars:
            caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if caps_ratio > 0.7:
                return True

        has_sentence_ending = any(text_stripped.endswith(p) for p in ['.', '!', '?', ';'])
        if len(text_stripped) < 60 and not has_sentence_ending:
            if text_stripped[0].isupper() and text_stripped.count(' ') < 8:
                return True

        return False

    # Keyword classification (with heading boost)

    def _classify_by_keywords(self, text: str, context_heading: str = '') -> Tuple[Optional[str], float]:
        """
        Weighted heuristic classification based on keywords.
        Keywords found in context_heading get a 2x contribution boost.
        Returns (clause_type, confidence).
        """
        text_lower: str = text.lower()
        heading_lower: str = context_heading.lower() if context_heading else ''

        best_match = None
        best_score = 0.0

        for clause_type, kw_pairs in CLAUSE_KEYWORDS_WEIGHTED.items():
            score = 0.0
            for kw, weight in kw_pairs:
                in_body = kw in text_lower
                in_heading = bool(heading_lower) and kw in heading_lower

                if in_heading and in_body:
                    score += weight * 4.0   
                elif in_heading:
                    score += weight * 2.5   
                elif in_body:
                    score += weight * 1.0   

            if score > best_score:
                best_score = score
                best_match = clause_type

        #CONFIDENCE SCALING - Higher base confidence
        if best_match and best_score >= 6.0:
            # Very strong match (multiple weighted keywords)
            confidence = min(0.98, 0.75 + (best_score * 0.03))
            return best_match, confidence
        elif best_match and best_score >= 3.0:
            # Strong match
            confidence = min(0.88, 0.65 + (best_score * 0.05))
            return best_match, confidence
        elif best_match and best_score >= 1.5:
            # Moderate match
            confidence = min(0.75, 0.52 + (best_score * 0.08))
            return best_match, confidence
        elif best_match and best_score >= 1.0:
            # Weak but valid match
            return best_match, 0.48
        
        return None, 0.0

    # Hybrid ensemble classification

    def _classify_segment(self, segment_text: str, context_heading: str = '') -> Dict:
        """
        Hybrid ensemble classification combining keyword and model-based approaches.
        """
        normalized = normalize_text(segment_text)

        # Get embedding for model-based classification
        with torch.no_grad():
            emb = self.model([normalized], batch_size=1)
            proj = self.projection(emb.to(self.model.device))

        # PATH A: Keyword-based (heading-boosted)
        kw_type, kw_conf = self._classify_by_keywords(segment_text, context_heading)

        # PATH B: Model-based few-shot
        model_type, model_conf = None, 0.0
        if len(self.support_embeddings) > 0:
            support_emb = torch.stack(self.support_embeddings).to(self.model.device)
            dists = torch.cdist(proj, support_emb)
            pred_idx = dists.argmin().item()
            min_dist = dists[0, pred_idx].item()
            # Improved model confidence calculation
            model_conf = max(0.0, min(0.95, 1.0 - (min_dist / 1.8)))
            model_type = self.support_labels[pred_idx]

        # ── Hybrid scoring ──────────────────────────────────────
        candidates = {}

        if kw_type:
            kw_score = kw_conf * self.kw_weight
            candidates[kw_type] = candidates.get(kw_type, 0.0) + kw_score
            
        if model_type:
            model_score = model_conf * self.model_weight
            candidates[model_type] = candidates.get(model_type, 0.0) + model_score

        # Boost confidence when both systems agree
        if kw_type and model_type and kw_type == model_type:
            # Both systems agree - very high confidence
            agreement_bonus = 0.15
            candidates[kw_type] = min(0.98, candidates[kw_type] + agreement_bonus)

        # Determine best candidate
        final_type = None
        final_conf = 0.0
        source = 'unknown'
        needs_review = False

        if candidates:
            final_type = max(candidates, key=lambda k: candidates.get(k, 0.0))
            final_conf = candidates[final_type]

            # Determine source
            if kw_type == model_type:
                source = 'hybrid_agree'
            elif kw_type == final_type:
                source = 'keywords'
            else:
                source = 'model'

            # Flag disagreement between keyword and model
            if kw_type and model_type and kw_type != model_type:
                conf_gap = abs(kw_conf - model_conf)
                if conf_gap < 0.25:
                    needs_review = True

            # Tiered thresholds based on agreement
            if kw_type and model_type and kw_type == model_type:
                # Both agree - very lenient threshold
                min_threshold = 0.35
            elif kw_type or model_type:
                # One source found it - moderate threshold
                min_threshold = 0.40
            else:
                # Neither found it - strict threshold
                min_threshold = 0.50

            if final_conf < min_threshold:
                final_type = None

        # Unknown clause handling
        if not final_type:
            final_type = 'Unknown clause'
            final_conf = 0.32  # Low but not too low
            source = 'unknown'
            needs_review = True

        return {
            'clause_type': final_type,
            'confidence': round(final_conf, 4),
            'source': source,
            'needs_review': needs_review,
            'embedding': proj,
            # Debug info
            'kw_type': kw_type,
            'kw_conf': round(kw_conf, 4) if kw_conf else 0.0,
            'model_type': model_type,
            'model_conf': round(model_conf, 4) if model_conf else 0.0,
        }

    # Post-processing: merge, demote, context-promote

    def _merge_adjacent_clauses(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-processing rules: merge adjacent same-type clauses, demote short spans, context-promote."""
        if not results:
            return results

        # --- Rule 1: Merge adjacent same-type --------------------------------
        merged: List[Dict[str, Any]] = [results[0].copy()]
        for curr in list(results[1:]):
            prev = merged[-1]
            same_type = prev['clause_type'] == curr['clause_type']
            not_unknown = curr['clause_type'] != 'Unknown clause'
            combined_len = len(prev['span']) + len(curr['span'])

            if same_type and not_unknown and combined_len <= 800:
                # Use average confidence for merged segments
                avg_conf = (prev['confidence'] + curr['confidence']) / 2
                
                merged[-1] = {
                    **prev,
                    'span': prev['span'] + ' … ' + curr['span'],
                    'end_idx': curr['end_idx'],
                    'confidence': round(min(0.95, avg_conf + 0.05), 4),
                    'source': prev['source'],
                }
            else:
                merged.append(curr.copy())

        # Rule 2: Demote very short segments
        for seg in merged:
            if len(seg['span'].strip()) < 25:
                seg['confidence'] = min(seg['confidence'], 0.50)
                seg['needs_review'] = True

        # Rule 3: Context-promote sandwiched unknowns
        for i in range(1, len(merged) - 1):
            if merged[i]['clause_type'] == 'Unknown clause':
                prev_type = merged[i - 1]['clause_type']
                next_type = merged[i + 1]['clause_type']
                prev_conf = merged[i - 1]['confidence']
                next_conf = merged[i + 1]['confidence']

                if (prev_type == next_type
                        and prev_type != 'Unknown clause'
                        and prev_conf >= 0.70  # Reduced from 0.75
                        and next_conf >= 0.70):
                    merged[i]['clause_type'] = prev_type
                    merged[i]['confidence'] = 0.65
                    merged[i]['source'] = 'context_promoted'
                    merged[i]['needs_review'] = False

        return merged

    # Main prediction pipeline

    def predict_cuad(self, contract_text: str, confidence_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Main prediction - extracts and classifies all clauses from a contract."""
        print(f"\n{'='*85}")
        print("LEXICACHE MULTI-CLAUSE EXTRACTION")
        print(f"{'='*85}")
        print(f"Contract length: {len(contract_text)} characters")

        segments = self._segment_contract(contract_text)
        print(f"Found {len(segments)} potential clause segments after legal-specific segmentation")

        results: List[Dict[str, Any]] = []
        seen_types: defaultdict[str, int] = defaultdict(int)
        unknown_count: int = 0

        for seg in segments:
            classification = self._classify_segment(
                seg['text'],
                context_heading=seg.get('context_heading', '')
            )

            # Show all detected clauses above threshold
            if classification['confidence'] >= confidence_threshold:
                type_key = classification['clause_type']
                
                span_text = seg['text'][:400] if len(seg['text']) > 400 else seg['text']
                span_text = ' '.join(span_text.split())

                result_dict = {
                    'clause_type': classification['clause_type'],
                    'confidence': classification['confidence'],
                    'span': span_text,
                    'start_idx': seg['start_idx'],
                    'end_idx': seg['end_idx'],
                    'source': classification.get('source', 'unknown'),
                    'is_unknown': type_key == 'Unknown clause',
                    'needs_review': classification.get('needs_review', False),
                    'context_heading': seg.get('context_heading', ''),
                }

                if type_key == 'Unknown clause':
                    unknown_count += 1

                results.append(result_dict)
                seen_types[type_key] += 1

        # Sort: unknown last, then by confidence descending
        results.sort(key=lambda x: (x['is_unknown'], -x['confidence']))

        # Post-processing: merge, demote, context-promote
        results = self._merge_adjacent_clauses(results)

        # Final needs_review pass
        for r in results:
            if r['is_unknown']:
                r['needs_review'] = True
            elif r['confidence'] < 0.60:
                r['needs_review'] = True

        known_count = len([r for r in results if not r['is_unknown']])
        unknown_count_final = len([r for r in results if r['is_unknown']])
        print(f"Identified {len(results)} total clauses after merging:")
        print(f"  {known_count} known clause types")
        print(f"  {unknown_count_final} unknown clauses (need user teaching)")
        if unknown_count_final > 0:
            print(f"  Tip: Teach the system by renaming 'Unknown clause' items")
        print(f"{'='*85}\n")

        return results

    # Persistence

    def _load_support_set(self):
        if Path(self.support_set_path).exists():
            try:
                with open(self.support_set_path, 'rb') as f:
                    data = pickle.load(f)
                    self.support_embeddings = data.get('embeddings', [])
                    self.support_labels = data.get('labels', [])
                    self.support_texts = data.get('texts', [])
                    self.label_to_id = data.get('label_to_id', {})
                    self.next_label_id = data.get('next_label_id', 0)
                print(f"  Loaded {len(self.support_embeddings)} examples from persistent storage")
            except Exception as e:
                print(f"  Failed to load support set: {e}")

    def _save_support_set(self):
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
            print(f"  Saved {len(self.support_embeddings)} examples to persistent storage")
        except Exception as e:
            print(f"  Failed to save support set: {e}")

    def _load_knowledge_base(self):
        if Path(self.knowledge_path).exists():
            try:
                with open(self.knowledge_path, 'r') as f:
                    data = json.load(f)
                    self.learned_types = data.get('learned_types', {})
                    self.clause_colors = data.get('clause_colors', {})
                    learned_examples = data.get('learned_examples', [])
                    if learned_examples:
                        print(f"  Rebuilding support set from {len(learned_examples)} saved examples...")
                        for example in learned_examples:
                            clause_type = example['clause_type']
                            text = example['text']
                            try:
                                normalized = normalize_text(text)
                                with torch.no_grad():
                                    emb = self.model([normalized], batch_size=1)
                                    proj = self.projection(emb.to(self.model.device))
                                if clause_type not in self.label_to_id:
                                    self.label_to_id[clause_type] = self.next_label_id
                                    self.next_label_id += 1
                                self.support_embeddings.append(proj.squeeze(0).cpu())
                                self.support_labels.append(clause_type)
                                self.support_texts.append(text)
                            except Exception as e:
                                print(f"    Failed to rebuild embedding for {clause_type}: {e}")
                        print(f"  Rebuilt support set with {len(self.support_embeddings)} examples")
                print(f"  Loaded {len(self.learned_types)} learned clause types")
            except Exception as e:
                print(f"  Failed to load knowledge base: {e}")

    def _save_knowledge_base(self):
        try:
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
                'learned_examples': learned_examples,
                'timestamp': datetime.now().isoformat(),
                'total_learned': len(self.learned_types),
                'total_examples': len(learned_examples)
            }
            with open(self.knowledge_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved knowledge base ({len(self.learned_types)} types, {len(learned_examples)} examples)")
        except Exception as e:
            print(f"  Failed to save knowledge base: {e}")

    # Online learning

    def learn_from_feedback(self, clause_text: str, correct_label: str, color: Optional[str] = None) -> bool:
        try:
            normalized = normalize_text(clause_text)
            with torch.no_grad():
                emb = self.model([normalized], batch_size=1)
                proj = self.projection(emb.to(self.model.device))

            if correct_label not in self.label_to_id:
                self.label_to_id[correct_label] = self.next_label_id
                self.next_label_id += 1
                if correct_label not in self.learned_types:
                    self.learned_types[correct_label] = {
                        'examples': [],
                        'count': 0,
                        'first_learned': datetime.now().isoformat()
                    }

            self.support_embeddings.append(proj.squeeze(0).cpu())
            self.support_labels.append(correct_label)
            self.support_texts.append(clause_text)

            entry: Dict[str, Any] = self.learned_types[correct_label]
            entry['count'] = int(entry.get('count', 0)) + 1
            existing = entry.get('examples')
            examples: List[str] = cast(List[str], existing) if isinstance(existing, list) else []
            examples.append(clause_text[:200])  # type: ignore[index]
            entry['examples'] = examples

            if color:
                self.clause_colors[correct_label] = color

            print(f"  Learned: '{correct_label}' (support set now has {len(self.support_embeddings)} examples)")
            self._save_support_set()
            self._save_knowledge_base()
            return True
        except Exception as e:
            print(f"  Failed to learn from feedback: {e}")
            return False

    def rename_unknown_clause(self, clause_text: str, old_span: str, new_type_name: str, color: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        success = self.learn_from_feedback(old_span, new_type_name, color)
        if success:
            return self.predict_cuad(clause_text)
        return None

    def update_clause_color(self, clause_type: str, color: str) -> bool:
        try:
            self.clause_colors[clause_type] = color
            self._save_knowledge_base()
            return True
        except Exception as e:
            print(f"  Failed to update color: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        label_counts: defaultdict[str, int] = defaultdict(int)
        for label in self.support_labels:
            label_counts[label] += 1
        return {
            'total_examples': len(self.support_embeddings),
            'unique_types': len(self.label_to_id),
            'known_cuad_types': len(CLAUSE_KEYWORDS_WEIGHTED),
            'learned_custom_types': len(self.learned_types),
            'label_distribution': dict(label_counts),
            'learned_types_detail': self.learned_types,
            'clause_colors': self.clause_colors,
            'device': str(self.model.device)
        }

    def get_all_clause_types_with_colors(self) -> Dict[str, Optional[str]]:
        all_types: Dict[str, Optional[str]] = {}
        for clause_type in CLAUSE_KEYWORDS_WEIGHTED.keys():
            all_types[clause_type] = self.clause_colors.get(clause_type)
        for clause_type in list(self.learned_types.keys()):
            all_types[str(clause_type)] = self.clause_colors.get(str(clause_type))
        all_types['Unknown clause'] = self.clause_colors.get('Unknown clause', '#9CA3AF')
        return all_types