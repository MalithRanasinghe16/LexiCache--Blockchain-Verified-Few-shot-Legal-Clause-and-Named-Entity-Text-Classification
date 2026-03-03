"""Unit tests for the LexiCache ML pipeline.
Tests segmentation, heading detection, keyword classification, and merge logic
without loading the heavy neural model (uses monkey-patching).
"""

import sys
import os
import types
import pytest  # type: ignore[import]

# ---------------------------------------------------------------------------
# Minimal stubs so we can import ml_model without torch / sentence-transformers
# ---------------------------------------------------------------------------
# Stub `torch`
torch_stub = types.ModuleType("torch")
torch_stub.no_grad = lambda: __import__('contextlib').nullcontext()  # type: ignore[attr-defined]
torch_stub.load = lambda *a, **kw: {}  # type: ignore[attr-defined]
torch_stub.cdist = None  # type: ignore[attr-defined]
torch_stub.stack = None  # type: ignore[attr-defined]
torch_nn = types.ModuleType("torch.nn")
class _FakeLinear:
    def __init__(self, *a, **kw): pass
    def load_state_dict(self, *a, **kw): pass
    def eval(self): pass
torch_nn.Linear = _FakeLinear  # type: ignore[attr-defined]
torch_stub.nn = torch_nn  # type: ignore[attr-defined]
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_nn)

# Stub `src.modeling`
modeling_stub = types.ModuleType("src.modeling")
class _FakeProto:
    hidden_size = 768
    device = "cpu"
    def __call__(self, *a, **kw): return None
modeling_stub.PrototypicalNetwork = _FakeProto  # type: ignore[attr-defined]
sys.modules.setdefault("src.modeling", modeling_stub)

# Stub `src.data`
data_stub = types.ModuleType("src.data")
data_stub.normalize_text = lambda t: t.lower().strip()  # type: ignore[attr-defined]
sys.modules.setdefault("src.data", data_stub)

# Now import the real module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ml_model import LexiCacheModel, CLAUSE_KEYWORDS_WEIGHTED  # type: ignore[import]


# ---------------------------------------------------------------------------
# Fixture: create a model instance without loading any weights
# ---------------------------------------------------------------------------
@pytest.fixture
def model(monkeypatch):  # type: ignore[misc]
    """Return LexiCacheModel with heavy I/O monkeypatched out."""
    monkeypatch.setattr(LexiCacheModel, "_load_support_set", lambda self: None)
    monkeypatch.setattr(LexiCacheModel, "_load_knowledge_base", lambda self: None)

    # Patch __init__ minimally
    import torch.nn as nn  # type: ignore[import]
    class FakeProj:
        def load_state_dict(self, *a, **kw): pass
        def eval(self): pass
        def __call__(self, x): return x

    class FakeModel:
        hidden_size = 768
        device = "cpu"
        def __call__(self, texts, batch_size=1): return None

    m = LexiCacheModel.__new__(LexiCacheModel)
    m.model = FakeModel()
    m.projection = FakeProj()
    m.support_embeddings = []
    m.support_labels = []
    m.support_texts = []
    m.label_to_id = {}
    m.next_label_id = 0
    m.support_set_path = "support_set.pkl"
    m.knowledge_path = "clause_knowledge.json"
    m.keyword_high_threshold = 0.75
    m.model_high_threshold = 0.70
    m.unknown_threshold = 0.60
    m.kw_weight = 0.60
    m.model_weight = 0.40
    m.learned_types = {}
    m.clause_colors = {}
    return m


# ===========================================================================
# 1. _is_heading() tests
# ===========================================================================
class TestIsHeading:
    HEADINGS = [
        "ARTICLE I",
        "ARTICLE IV GENERAL PROVISIONS",
        "Section 2.1",
        "1. Definitions",
        "1.2 Payment Terms",
        "TERMINATION:",
        "IV. Governing Law",
        "CONFIDENTIALITY OBLIGATIONS",
        "Entire Agreement:",
        "Hi",                   # too short
    ]
    BODIES = [
        "This Agreement shall be governed by the laws of the State of California.",
        "Each party shall indemnify and hold harmless the other party from any claims.",
        "The term of this Agreement commences on the Effective Date and continues for one (1) year.",
        "Any notice required or permitted under this Agreement shall be in writing.",
        "In no event shall either party be liable for consequential or incidental damages.",
        "The parties agree that this Agreement constitutes the entire agreement between them.",
    ]

    def test_headings_detected(self, model):
        for h in self.HEADINGS:
            assert model._is_heading(h), f"Expected heading: {h!r}"

    def test_bodies_not_headings(self, model):
        for b in self.BODIES:
            assert not model._is_heading(b), f"Expected body (not heading): {b!r}"


# ===========================================================================
# 2. _segment_contract() tests
# ===========================================================================
SAMPLE_CONTRACT = """\
MASTER SERVICES AGREEMENT

ARTICLE I — DEFINITIONS

As used in this Agreement, the following terms shall have the meanings set forth below.
"Confidential Information" means any information disclosed by one party to the other.

ARTICLE II — PAYMENT TERMS

Payment shall be due within thirty (30) days of invoice date. Late payments shall accrue
interest at the rate of 1.5% per month.

ARTICLE III — TERMINATION

Either party may terminate this Agreement upon thirty (30) days prior written notice
to the other party. Termination for material breach shall be effective immediately upon notice.

ARTICLE IV — GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws of the
State of California, without regard to its conflict of law provisions.
"""


class TestSegmentContract:
    def test_returns_list(self, model):
        segs = model._segment_contract(SAMPLE_CONTRACT)
        assert isinstance(segs, list)

    def test_no_standalone_headings(self, model):
        segs = model._segment_contract(SAMPLE_CONTRACT)
        for seg in segs:
            assert not model._is_heading(seg['text']), (
                f"A heading leaked into segments: {seg['text']!r}"
            )

    def test_segments_have_required_keys(self, model):
        segs = model._segment_contract(SAMPLE_CONTRACT)
        for seg in segs:
            assert 'text' in seg
            assert 'start_idx' in seg
            assert 'end_idx' in seg
            assert 'context_heading' in seg

    def test_context_heading_propagated(self, model):
        segs = model._segment_contract(SAMPLE_CONTRACT)
        # At least one segment should have a heading context
        headings_seen = [s['context_heading'] for s in segs if s['context_heading']]
        assert headings_seen, "Expected heading context to be propagated to body segments"

    def test_reasonable_segment_count(self, model):
        segs = model._segment_contract(SAMPLE_CONTRACT)
        # Should have at least 3 content segments (one per article body)
        assert len(segs) >= 3, f"Too few segments: {len(segs)}"
        # Should not explode (e.g. split every sentence on a short contract)
        assert len(segs) <= 20, f"Too many segments: {len(segs)}"


# ===========================================================================
# 3. _classify_by_keywords() tests
# ===========================================================================
class TestClassifyByKeywords:
    def test_payment_terms_detected(self, model):
        text = "Payment shall be due within thirty days of invoice date. Late payments accrue interest."
        clause_type, conf = model._classify_by_keywords(text)
        assert clause_type == 'Payment Terms', f"Got: {clause_type}"
        assert conf >= 0.40

    def test_governing_law_detected(self, model):
        text = "This Agreement shall be governed by the laws of the State of New York."
        clause_type, conf = model._classify_by_keywords(text)
        assert clause_type == 'Governing Law', f"Got: {clause_type}"

    def test_termination_detected(self, model):
        text = "Either party may terminate this agreement upon written notice of material breach."
        clause_type, conf = model._classify_by_keywords(text)
        assert clause_type == 'Termination', f"Got: {clause_type}"

    def test_confidentiality_detected(self, model):
        text = "Each party shall keep all confidential information strictly confidential and shall not disclose to third parties."
        clause_type, conf = model._classify_by_keywords(text)
        assert clause_type == 'Confidentiality', f"Got: {clause_type}"

    def test_heading_boost_increases_confidence(self, model):
        text = "Either party may terminate upon 30 days notice."
        _, conf_no_heading = model._classify_by_keywords(text, context_heading='')
        _, conf_with_heading = model._classify_by_keywords(text, context_heading='TERMINATION')
        assert conf_with_heading >= conf_no_heading, (
            "Heading boost should not decrease confidence"
        )

    def test_unknown_for_gibberish(self, model):
        text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor."
        clause_type, conf = model._classify_by_keywords(text)
        # Either None or very low confidence
        assert clause_type is None or conf < 0.55


# ===========================================================================
# 4. _merge_adjacent_clauses() tests
# ===========================================================================
class TestMergeAdjacentClauses:
    def _make_seg(self, clause_type, span, confidence=0.80, start=0, end=100):
        return {
            'clause_type': clause_type,
            'confidence': confidence,
            'span': span,
            'start_idx': start,
            'end_idx': end,
            'source': 'keywords',
            'is_unknown': clause_type == 'Unknown clause',
            'needs_review': False,
            'context_heading': '',
        }

    def test_merges_same_type(self, model):
        segs = [
            self._make_seg('Termination', 'Either party may terminate.', end=50),
            self._make_seg('Termination', 'Termination for breach is immediate.', start=51, end=100),
        ]
        merged = model._merge_adjacent_clauses(segs)
        assert len(merged) == 1
        assert 'Either party may terminate.' in merged[0]['span']
        assert 'Termination for breach is immediate.' in merged[0]['span']

    def test_does_not_merge_different_types(self, model):
        segs = [
            self._make_seg('Termination', 'Either party may terminate.', end=50),
            self._make_seg('Governing Law', 'Governed by California law.', start=51, end=100),
        ]
        merged = model._merge_adjacent_clauses(segs)
        assert len(merged) == 2

    def test_does_not_merge_unknowns(self, model):
        segs = [
            self._make_seg('Unknown clause', 'Some unknown text here.', end=50),
            self._make_seg('Unknown clause', 'More unknown text.', start=51, end=100),
        ]
        merged = model._merge_adjacent_clauses(segs)
        # Unknowns should NOT be merged (they need individual user review)
        assert len(merged) == 2

    def test_demotes_short_span(self, model):
        segs = [self._make_seg('Termination', 'Short.', confidence=0.85)]
        merged = model._merge_adjacent_clauses(segs)
        assert merged[0]['confidence'] <= 0.45
        assert merged[0]['needs_review'] is True

    def test_context_promotes_sandwiched_unknown(self, model):
        segs = [
            self._make_seg('Termination', 'Either party may terminate upon notice.', confidence=0.85, end=80),
            self._make_seg('Unknown clause', 'This paragraph relates to the same topic.', start=81, end=160),
            self._make_seg('Termination', 'Termination for breach shall be immediate.', confidence=0.82, start=161, end=240),
        ]
        merged = model._merge_adjacent_clauses(segs)
        # The unknown in the middle should be promoted to Termination
        middle = merged[1]
        assert middle['clause_type'] == 'Termination', f"Expected promoted, got: {middle['clause_type']}"
        assert middle['source'] == 'context_promoted'

    def test_does_not_merge_if_combined_too_long(self, model):
        long_span = 'x ' * 400   # 800 chars
        segs = [
            self._make_seg('Termination', long_span.strip(), end=800),
            self._make_seg('Termination', 'Another termination clause.', start=801, end=830),
        ]
        merged = model._merge_adjacent_clauses(segs)
        # Combined would exceed 800 — should NOT merge
        assert len(merged) == 2


# ===========================================================================
# 5. CLAUSE_KEYWORDS_WEIGHTED completeness test
# ===========================================================================
class TestKeywordDictionary:
    REQUIRED_CUAD_TYPES = [
        'Document Name', 'Parties', 'Agreement Date', 'Effective Date', 'Expiration Date',
        'Payment Terms', 'Cap on Liability', 'Liquidated Damages', 'Revenue/Profit Sharing',
        'Price Restrictions', 'Minimum Commitment', 'Volume Restriction',
        'Limitation of Liability', 'Indemnification', 'Warranty Duration', 'Insurance',
        'Termination', 'Termination for Convenience', 'Renewal Term',
        'Notice Period to Terminate Renewal', 'Post-Termination Services',
        'Non-Compete', 'Exclusivity', 'No-Solicit of Customers', 'No-Solicit of Employees',
        'Non-Disparagement', 'Intellectual Property', 'IP Ownership Assignment',
        'Joint IP Ownership', 'License Grant', 'Confidentiality',
        'Change of Control', 'Anti-Assignment', 'Covenant Not to Sue',
        'Governing Law', 'Dispute Resolution', 'Jurisdiction', 'Notice', 'Force Majeure',
        'Severability', 'Entire Agreement', 'Amendment', 'Waiver',
        'Third Party Beneficiary', 'Audit Rights', 'ROFR/ROFO/ROFN', 'Most Favored Nation',
    ]

    def test_all_cuad_types_present(self):
        for ct in self.REQUIRED_CUAD_TYPES:
            assert ct in CLAUSE_KEYWORDS_WEIGHTED, f"Missing CUAD type: {ct!r}"

    def test_each_type_has_keywords(self):
        for ct, kw_pairs in CLAUSE_KEYWORDS_WEIGHTED.items():
            assert len(kw_pairs) >= 3, f"Too few keywords for {ct!r}: {len(kw_pairs)}"

    def test_weights_are_valid(self):
        for ct, kw_pairs in CLAUSE_KEYWORDS_WEIGHTED.items():
            for kw, weight in kw_pairs:
                assert isinstance(weight, (int, float)) and weight > 0, (
                    f"Bad weight for {ct!r} → {kw!r}: {weight}"
                )
