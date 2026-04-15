"""Hybrid evaluation: Fine-tuned Legal-BERT + keyword scoring on CUAD test split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
# CUAD 41 canonical categories
CUAD_41_CATEGORIES: List[str] = [
    "Document Name", "Parties", "Agreement Date", "Effective Date",
    "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
    "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit Of Customers", "No-Solicit Of Employees", "Non-Disparagement",
    "Termination For Convenience", "ROFR/ROFO/ROFN", "Change Of Control",
    "Anti-Assignment", "Revenue/Profit Sharing", "Price Restrictions",
    "Minimum Commitment", "Volume Restriction", "Ip Ownership Assignment",
    "Joint Ip Ownership", "License Grant", "Non-Transferable License",
    "Affiliate License-Licensor", "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
    "Source Code Escrow", "Post-Termination Services", "Audit Rights",
    "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
    "Warranty Duration", "Insurance", "Covenant Not To Sue",
    "Third Party Beneficiary", "Indemnification",
]
NUM_LABELS = len(CUAD_41_CATEGORIES)
CAT_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CUAD_41_CATEGORIES)}
# Label aliases (same as training script)
LABEL_ALIASES: Dict[str, str] = {
    "termination": "Termination For Convenience",
    "termination for convenience": "Termination For Convenience",
    "termination for cause": "Termination For Convenience",
    "limitation of liability": "Cap On Liability",
    "cap on liability": "Cap On Liability",
    "liability cap": "Cap On Liability",
    "uncapped liability": "Uncapped Liability",
    "unlimited liability": "Uncapped Liability",
    "ip ownership assignment": "Ip Ownership Assignment",
    "intellectual property assignment": "Ip Ownership Assignment",
    "joint ip ownership": "Joint Ip Ownership",
    "joint intellectual property": "Joint Ip Ownership",
    "license grant": "License Grant",
    "non-transferable license": "Non-Transferable License",
    "irrevocable or perpetual license": "Irrevocable Or Perpetual License",
    "perpetual license": "Irrevocable Or Perpetual License",
    "irrevocable license": "Irrevocable Or Perpetual License",
    "unlimited license": "Unlimited/All-You-Can-Eat-License",
    "affiliate license-licensor": "Affiliate License-Licensor",
    "affiliate license-licensee": "Affiliate License-Licensee",
    "no-solicit of customers": "No-Solicit Of Customers",
    "no-solicit of employees": "No-Solicit Of Employees",
    "non-solicitation": "No-Solicit Of Employees",
    "change of control": "Change Of Control",
    "rofr/rofo/rofn": "ROFR/ROFO/ROFN",
    "right of first refusal": "ROFR/ROFO/ROFN",
    "covenant not to sue": "Covenant Not To Sue",
    "third party beneficiary": "Third Party Beneficiary",
    "source code escrow": "Source Code Escrow",
    "most favored nation": "Most Favored Nation",
    "revenue/profit sharing": "Revenue/Profit Sharing",
    "revenue sharing": "Revenue/Profit Sharing",
    "profit sharing": "Revenue/Profit Sharing",
    "notice period to terminate renewal": "Notice Period To Terminate Renewal",
    "post-termination services": "Post-Termination Services",
    "audit rights": "Audit Rights",
    "indemnification": "Indemnification",
    "insurance": "Insurance",
    "warranty duration": "Warranty Duration",
    "liquidated damages": "Liquidated Damages",
    "minimum commitment": "Minimum Commitment",
    "volume restriction": "Volume Restriction",
    "price restrictions": "Price Restrictions",
    "exclusivity": "Exclusivity",
    "non-compete": "Non-Compete",
    "anti-assignment": "Anti-Assignment",
    "governing law": "Governing Law",
    "renewal term": "Renewal Term",
    "expiration date": "Expiration Date",
    "effective date": "Effective Date",
    "agreement date": "Agreement Date",
    "parties": "Parties",
    "document name": "Document Name",
    "non-disparagement": "Non-Disparagement",
}


def map_clause_type(raw: str) -> Optional[str]:
    if not raw:
        return None
    if raw in CAT_TO_IDX:
        return raw
    lower = raw.strip().lower()
    if lower in LABEL_ALIASES:
        return LABEL_ALIASES[lower]
    for cat in CUAD_41_CATEGORIES:
        if cat.lower() == lower:
            return cat
    return None
# Keyword scoring (symbolic component of the hybrid)
KEYWORD_MAP: Dict[str, List[Tuple[str, float]]] = {
    "Document Name": [
        ("this agreement", 2), ("master agreement", 2), ("master services agreement", 3),
        ("service agreement", 2), ("license agreement", 2), ("agreement", 1),
    ],
    "Parties": [
        ("by and between", 3), ("hereinafter referred to as", 3),
        ("collectively referred to as", 3), ("hereinafter called", 2),
    ],
    "Agreement Date": [
        ("effective as of", 2), ("dated as of", 2), ("entered into as of", 2),
        ("this agreement is dated", 2), ("dated", 1),
    ],
    "Effective Date": [
        ("effective date shall be", 2), ("commence on", 2), ("commencement date", 2),
        ("shall become effective", 2), ("effective date", 1),
    ],
    "Expiration Date": [
        ("shall expire", 2), ("expiration date", 2), ("term shall end", 2),
        ("agreement expires", 2), ("expire", 1), ("expiration", 1),
    ],
    "Renewal Term": [
        ("automatically renew", 2), ("renewal term", 2), ("successive terms", 2),
        ("renew for", 2), ("renewal", 1),
    ],
    "Notice Period To Terminate Renewal": [
        ("notice of non-renewal", 3), ("written notice of termination", 2),
        ("days prior written notice", 2), ("notice period", 1),
    ],
    "Governing Law": [
        ("governed by the laws", 3), ("governing law", 2), ("laws of the state", 2),
        ("jurisdiction of", 2), ("applicable law", 1),
    ],
    "Most Favored Nation": [
        ("most favored nation", 3), ("most-favored-nation", 3), ("mfn", 2),
        ("no less favorable", 2), ("best pricing", 1),
    ],
    "Non-Compete": [
        ("shall not compete", 3), ("non-compete", 3), ("non-competition", 3),
        ("covenant not to compete", 3), ("competitive business", 2),
    ],
    "Exclusivity": [
        ("exclusive right", 2), ("exclusive license", 2), ("exclusively", 2),
        ("sole and exclusive", 3), ("exclusivity", 2), ("exclusive", 1),
    ],
    "No-Solicit Of Customers": [
        ("solicit customers", 3), ("no-solicit of customers", 3),
        ("solicit any customer", 3), ("poach customers", 2),
    ],
    "No-Solicit Of Employees": [
        ("solicit employees", 3), ("no-solicit of employees", 3),
        ("hire employees", 2), ("recruit employees", 2), ("non-solicitation", 2),
    ],
    "Non-Disparagement": [
        ("non-disparagement", 3), ("shall not disparage", 3),
        ("disparaging statements", 2), ("negative statements", 2),
    ],
    "Termination For Convenience": [
        ("terminate for convenience", 3), ("without cause termination", 3),
        ("terminate for any reason", 3), ("terminate without cause", 3),
        ("may terminate this agreement upon", 2), ("right to terminate", 2),
        ("termination", 1), ("terminate", 1),
    ],
    "ROFR/ROFO/ROFN": [
        ("right of first refusal", 3), ("right of first offer", 3),
        ("right of first negotiation", 3), ("rofr", 2), ("rofo", 2),
    ],
    "Change Of Control": [
        ("change of control", 3), ("change in control", 3),
        ("acquisition of", 2), ("merger or acquisition", 2), ("takeover", 2),
    ],
    "Anti-Assignment": [
        ("may not assign", 3), ("shall not assign", 3), ("without prior written consent", 2),
        ("assignment is prohibited", 3), ("non-assignable", 2), ("anti-assignment", 2),
    ],
    "Revenue/Profit Sharing": [
        ("revenue sharing", 3), ("profit sharing", 3), ("share of revenue", 3),
        ("percentage of revenue", 2), ("revenue split", 2),
    ],
    "Price Restrictions": [
        ("price restriction", 3), ("price ceiling", 2), ("maximum price", 2),
        ("price control", 2), ("pricing restriction", 2),
    ],
    "Minimum Commitment": [
        ("minimum purchase", 3), ("minimum commitment", 3), ("minimum order", 3),
        ("minimum volume", 2), ("purchase commitment", 2),
    ],
    "Volume Restriction": [
        ("volume restriction", 3), ("maximum volume", 2), ("volume cap", 2),
        ("quantity restriction", 2), ("volume limit", 2),
    ],
    "Ip Ownership Assignment": [
        ("assigns all right", 3), ("intellectual property ownership", 3),
        ("ip ownership", 3), ("work for hire", 2), ("assigns to", 2),
    ],
    "Joint Ip Ownership": [
        ("jointly own", 3), ("joint ownership", 3), ("co-ownership", 3),
        ("jointly developed", 2), ("joint intellectual property", 2),
    ],
    "License Grant": [
        ("hereby grants", 2), ("grants a license", 2), ("non-exclusive license", 2),
        ("exclusive license to", 2), ("royalty-free license", 2), ("license", 1),
    ],
    "Non-Transferable License": [
        ("non-transferable license", 3), ("not transferable", 3),
        ("may not sublicense", 3), ("non-sublicensable", 3),
        ("personal and non-transferable", 2), ("non-transferable", 1),
    ],
    "Affiliate License-Licensor": [
        ("licensor affiliates", 3), ("affiliates of licensor", 3),
        ("licensor and its affiliates", 2), ("affiliate of the licensor", 2),
    ],
    "Affiliate License-Licensee": [
        ("licensee affiliates", 3), ("affiliates of licensee", 3),
        ("licensee and its affiliates", 2), ("affiliate of the licensee", 2),
    ],
    "Unlimited/All-You-Can-Eat-License": [
        ("unlimited license", 3), ("all-you-can-eat", 3), ("unlimited use", 3),
        ("unrestricted license", 3), ("enterprise license", 2),
    ],
    "Irrevocable Or Perpetual License": [
        ("irrevocable license", 3), ("perpetual license", 3),
        ("irrevocable and perpetual", 3), ("license shall survive", 2),
    ],
    "Source Code Escrow": [
        ("source code escrow", 3), ("escrow agent", 3),
        ("deposit source code", 3), ("escrow agreement", 2), ("escrow", 1),
    ],
    "Post-Termination Services": [
        ("post-termination obligations", 2), ("wind-down services", 2),
        ("transition assistance", 2), ("after termination", 2),
        ("termination assistance", 2), ("post-termination", 1),
    ],
    "Audit Rights": [
        ("right to audit", 3), ("audit rights", 3), ("inspect records", 2),
        ("audit the books", 2), ("audit", 1),
    ],
    "Uncapped Liability": [
        ("unlimited liability", 3), ("no cap on liability", 3),
        ("liability shall not be limited", 3), ("uncapped liability", 3),
        ("no limitation on liability", 2), ("uncapped", 1),
    ],
    "Cap On Liability": [
        ("limitation of liability", 3), ("cap on liability", 3),
        ("shall not exceed", 2), ("aggregate liability", 2),
        ("maximum liability", 2), ("liability cap", 2),
    ],
    "Liquidated Damages": [
        ("liquidated damages", 3), ("agreed damages", 2),
        ("pre-agreed damages", 2), ("penalty clause", 2),
    ],
    "Warranty Duration": [
        ("warranty period", 3), ("warranty term", 3), ("warranty for", 2),
        ("warranty shall last", 2), ("warranty duration", 2), ("warranty", 1),
    ],
    "Insurance": [
        ("shall maintain insurance", 3), ("insurance coverage", 3),
        ("general liability insurance", 3), ("insurance policy", 2), ("insurance", 1),
    ],
    "Covenant Not To Sue": [
        ("covenant not to sue", 3), ("agrees not to sue", 3),
        ("waives right to sue", 3), ("release of claims", 2),
        ("shall not bring any action", 2),
    ],
    "Third Party Beneficiary": [
        ("no third party beneficiaries", 3), ("intended beneficiary", 3),
        ("third-party rights", 2), ("benefit of third parties", 2),
        ("third party beneficiary", 1),
    ],
    "Indemnification": [
        ("shall indemnify", 3), ("indemnification", 3), ("hold harmless", 3),
        ("defend and indemnify", 3), ("indemnify and hold", 2), ("indemnify", 1),
    ],
}

# Pre-compute max possible keyword score per category for normalisation
_MAX_KW_SCORE: Dict[str, float] = {}
for _cat, _kws in KEYWORD_MAP.items():
    _MAX_KW_SCORE[_cat] = float(sum(w for _, w in _kws))


def compute_keyword_scores(text: str) -> np.ndarray:
    """Return a [41] float array of normalised keyword scores in [0, 1]."""
    text_lower = text.lower()
    scores = np.zeros(NUM_LABELS, dtype=np.float32)
    for cat, kws in KEYWORD_MAP.items():
        idx = CAT_TO_IDX[cat]
        raw = 0.0
        for kw, w in kws:
            if kw in text_lower:
                raw += w
        max_score = _MAX_KW_SCORE.get(cat, 1.0)
        scores[idx] = min(1.0, raw / max(max_score, 1.0))
    return scores
# Fine-tuned model (mirrors train_cuad_multilabel_finetune.py)
class LegalBERTMultiLabel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "nlpaueb/legal-bert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_chunks: torch.Tensor,
    ) -> torch.Tensor:
        B, C, L = input_ids.shape
        ids_flat = input_ids.view(B * C, L)
        mask_flat = attention_mask.view(B * C, L)
        out = self.encoder(input_ids=ids_flat, attention_mask=mask_flat)
        cls_flat = out.last_hidden_state[:, 0, :]
        cls_3d = cls_flat.view(B, C, -1)
        chunk_mask = torch.zeros(B, C, device=input_ids.device)
        for b in range(B):
            chunk_mask[b, : int(n_chunks[b].item())] = 1.0
        chunk_mask = chunk_mask.unsqueeze(-1)
        pooled = (cls_3d * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def load_finetuned_model(
    model_dir: Path,
    device: torch.device,
) -> Tuple[LegalBERTMultiLabel, AutoTokenizer, dict]:
    """Load fine-tuned model, tokenizer, and config from model_dir."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    encoder_name = cfg.get("encoder_name", "nlpaueb/legal-bert-base-uncased")
    dropout = cfg.get("dropout", 0.1)

    print(f"  Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    print(f"  Building model ({encoder_name}) ...")
    model = LegalBERTMultiLabel(encoder_name=encoder_name, dropout=dropout).to(device)

    # Prefer best_model.pth, fall back to final_model.pth
    for fname in ("best_model.pth", "final_model.pth"):
        weights_path = model_dir / fname
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"  Loaded weights: {weights_path}")
            break
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    model.eval()
    return model, tokenizer, cfg


def encode_document(
    text: str,
    tokenizer,
    max_chunks: int,
    chunk_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a document into chunks; return (input_ids, attention_mask, n_chunks)."""
    cls_id = tokenizer.cls_token_id or 101
    sep_id = tokenizer.sep_token_id or 102
    pad_id = tokenizer.pad_token_id or 0

    token_ids: List[int] = tokenizer(
        text, add_special_tokens=False, truncation=False
    )["input_ids"]

    effective = chunk_size - 2
    raw_chunks = [token_ids[i: i + effective] for i in range(0, max(1, len(token_ids)), effective)]
    raw_chunks = raw_chunks[:max_chunks]
    n_real = len(raw_chunks)

    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []

    for chunk in raw_chunks:
        ids = [cls_id] + chunk + [sep_id]
        mask = [1] * len(ids)
        pad_len = chunk_size - len(ids)
        ids += [pad_id] * pad_len
        mask += [0] * pad_len
        input_ids_list.append(ids)
        attention_mask_list.append(mask)

    while len(input_ids_list) < max_chunks:
        input_ids_list.append([0] * chunk_size)
        attention_mask_list.append([0] * chunk_size)

    ids_t = torch.tensor([input_ids_list], dtype=torch.long, device=device)    # [1, C, L]
    mask_t = torch.tensor([attention_mask_list], dtype=torch.long, device=device)
    n_t = torch.tensor([n_real], dtype=torch.long, device=device)
    return ids_t, mask_t, n_t
# Ground-truth extraction
def extract_gt_labels(item: dict) -> np.ndarray:
    """Return a [41] binary array: 1 if contract has ≥1 non-empty annotation."""
    full_text: str = item.get("full_text", "")
    anns: list = item.get("clause_types", [])
    label_vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        raw_type = ann.get("clause_type", "")
        start = ann.get("start", -1)
        end = ann.get("end", -1)
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        if start < 0 or end <= start or end > len(full_text):
            continue
        span = full_text[start:end].strip()
        if not span:
            continue
        canonical = map_clause_type(raw_type)
        if canonical and canonical in CAT_TO_IDX:
            label_vec[CAT_TO_IDX[canonical]] = 1.0
    return label_vec
# Main evaluation
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid (fine-tuned Legal-BERT + keyword) CUAD evaluation"
    )
    p.add_argument("--test-dir", default="data/processed/cuad/test",
                   help="Path to CUAD test JSON files")
    p.add_argument("--model-dir", default="models/cuad_multilabel_finetuned",
                   help="Directory containing fine-tuned model (config.json + *.pth)")
    p.add_argument("--output-file",
                   default="experiments/results/cuad_hybrid_finetuned_results.json",
                   help="Path to save JSON results")
    p.add_argument("--neural-weight", type=float, default=0.65,
                   help="Weight for neural score in hybrid fusion (0=keyword-only, 1=neural-only)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Hybrid score threshold for positive prediction")
    p.add_argument("--max-chunks", type=int, default=6,
                   help="Max document chunks (must match training config)")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Tokens per chunk (must match training config)")
    p.add_argument("--keyword-only", action="store_true",
                   help="Skip neural model; use keyword scoring only")
    p.add_argument("--max-files", type=int, default=-1,
                   help="Limit number of test files (for quick smoke tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    neural_weight = 0.0 if args.keyword_only else args.neural_weight
    kw_weight = 1.0 - neural_weight

    print("=" * 70)
    print("LexiCache — Hybrid CUAD Document-Level Evaluation")
    print(f"  Neural weight : {neural_weight:.2f}  |  Keyword weight: {kw_weight:.2f}")
    print(f"  Threshold     : {args.threshold}")
    print(f"  Device        : {device}")
    print("=" * 70)
    # Load fine-tuned model (unless keyword-only)
    model: Optional[LegalBERTMultiLabel] = None
    tokenizer = None
    model_cfg: dict = {}

    if not args.keyword_only:
        if not model_dir.exists():
            print(f"\nERROR: Model directory not found: {model_dir}")
            print("Run training first:")
            print("  python scripts/training/train_cuad_multilabel_finetune.py")
            sys.exit(1)
        print(f"\n[1/3] Loading fine-tuned model from {model_dir} ...")
        model, tokenizer, model_cfg = load_finetuned_model(model_dir, device)
        # Override chunk settings from config if available
        args.max_chunks = model_cfg.get("max_chunks", args.max_chunks)
        args.chunk_size = model_cfg.get("chunk_size", args.chunk_size)
        print(f"  Chunk config: {args.max_chunks} chunks × {args.chunk_size} tokens")
    else:
        print("\n[1/3] Keyword-only mode — skipping neural model load.")
    # Load test contracts
    print(f"\n[2/3] Loading test contracts from {test_dir} ...")
    test_files = sorted(test_dir.glob("*.json"))
    if args.max_files > 0:
        test_files = test_files[: args.max_files]
    print(f"  Found {len(test_files)} test contracts")

    if len(test_files) == 0:
        print(f"ERROR: No JSON files found in {test_dir}")
        sys.exit(1)
    # Inference loop
    print(f"\n[3/3] Running inference ...")

    Y_true: List[np.ndarray] = []
    Y_pred: List[np.ndarray] = []
    Y_neural: List[np.ndarray] = []
    Y_kw: List[np.ndarray] = []
    per_contract: List[dict] = []

    for path in tqdm(test_files, desc="Contracts"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception as e:
            print(f"  WARN: Could not load {path.name}: {e}")
            continue

        full_text: str = item.get("full_text", "")
        if not full_text.strip():
            continue

        # Ground truth
        gt_vec = extract_gt_labels(item)

        # Keyword scores
        kw_scores = compute_keyword_scores(full_text)

        # Neural scores
        if model is not None and tokenizer is not None:
            with torch.no_grad():
                ids_t, mask_t, n_t = encode_document(
                    full_text, tokenizer, args.max_chunks, args.chunk_size, device
                )
                logits = model(ids_t, mask_t, n_t)
                neural_scores = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        else:
            neural_scores = np.zeros(NUM_LABELS, dtype=np.float32)

        # Hybrid fusion
        hybrid_scores = neural_weight * neural_scores + kw_weight * kw_scores
        pred_vec = (hybrid_scores >= args.threshold).astype(np.float32)

        Y_true.append(gt_vec)
        Y_pred.append(pred_vec)
        Y_neural.append(neural_scores)
        Y_kw.append(kw_scores)

        # Per-contract top-K recall
        gt_set = set(np.where(gt_vec > 0)[0])
        pred_set = set(np.where(pred_vec > 0)[0])
        found = len(gt_set & pred_set)
        topk_recall = found / len(gt_set) if gt_set else 0.0

        per_contract.append({
            "contract": path.stem,
            "gt_count": int(len(gt_set)),
            "pred_count": int(len(pred_set)),
            "found": int(found),
            "recall": round(topk_recall, 4),
        })
    # Aggregate metrics
    Y_true_arr = np.vstack(Y_true)   # [N, 41]
    Y_pred_arr = np.vstack(Y_pred)   # [N, 41]

    macro_p = precision_score(Y_true_arr, Y_pred_arr, average="macro", zero_division=0)
    macro_r = recall_score(Y_true_arr, Y_pred_arr, average="macro", zero_division=0)
    macro_f1 = f1_score(Y_true_arr, Y_pred_arr, average="macro", zero_division=0)
    micro_p = precision_score(Y_true_arr, Y_pred_arr, average="micro", zero_division=0)
    micro_r = recall_score(Y_true_arr, Y_pred_arr, average="micro", zero_division=0)
    micro_f1 = f1_score(Y_true_arr, Y_pred_arr, average="micro", zero_division=0)
    h_loss = hamming_loss(Y_true_arr, Y_pred_arr)

    topk_recalls = [c["recall"] for c in per_contract if c["gt_count"] > 0]
    topk_mean = float(np.mean(topk_recalls)) if topk_recalls else 0.0

    # Per-class metrics via classification_report
    report = classification_report(
        Y_true_arr, Y_pred_arr,
        target_names=CUAD_41_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    per_class: Dict[str, dict] = {}
    for cat in CUAD_41_CATEGORIES:
        r = report.get(cat, {})
        support = int(Y_true_arr[:, CAT_TO_IDX[cat]].sum())
        per_class[cat] = {
            "precision": round(float(r.get("precision", 0.0)), 6),
            "recall": round(float(r.get("recall", 0.0)), 6),
            "f1": round(float(r.get("f1-score", 0.0)), 6),
            "support": support,
        }
    # Print summary table
    print("\n" + "=" * 70)
    print("HYBRID CUAD DOCUMENT-LEVEL EVALUATION RESULTS")
    print(f"  Neural weight={neural_weight:.2f}  Keyword weight={kw_weight:.2f}  "
          f"Threshold={args.threshold}")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    print(f"{'Macro (41 classes)':<30} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f}")
    print(f"{'Micro (all instances)':<30} {micro_p:>10.4f} {micro_r:>10.4f} {micro_f1:>10.4f}")
    print(f"{'Hamming Loss':<30} {h_loss:>10.4f}")
    print(f"{'TopK Mean Recall':<30} {topk_mean:>10.4f}")

    print(f"\n{'Category':<40} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}")
    print("-" * 62)
    for cat in CUAD_41_CATEGORIES:
        m = per_class[cat]
        flag = " ✓" if m["f1"] >= 0.5 else ("  " if m["f1"] > 0 else " ✗")
        print(f"  {cat:<38} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>5}{flag}")

    # Zero-recall types
    zero_f1 = [c for c in CUAD_41_CATEGORIES if per_class[c]["f1"] == 0.0 and per_class[c]["support"] > 0]
    if zero_f1:
        print(f"\n  Zero-F1 types ({len(zero_f1)}): {', '.join(zero_f1)}")

    print("\n" + "=" * 70)
    # Save results to JSON
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Hybrid document-level CUAD evaluation: "
            f"fine-tuned Legal-BERT (w={neural_weight:.2f}) + "
            f"keyword scoring (w={kw_weight:.2f})"
        ),
        "config": {
            "test_dir": str(args.test_dir),
            "model_dir": str(args.model_dir),
            "n_test_contracts": len(Y_true),
            "n_cuad_categories": NUM_LABELS,
            "neural_weight": neural_weight,
            "keyword_weight": kw_weight,
            "threshold": args.threshold,
            "max_chunks": args.max_chunks,
            "chunk_size": args.chunk_size,
            "keyword_only": args.keyword_only,
            "model_config": model_cfg,
        },
        "aggregate_metrics": {
            "macro": {
                "precision": round(macro_p, 6),
                "recall": round(macro_r, 6),
                "f1": round(macro_f1, 6),
            },
            "micro": {
                "precision": round(micro_p, 6),
                "recall": round(micro_r, 6),
                "f1": round(micro_f1, 6),
            },
            "hamming_loss": round(h_loss, 6),
        },
        "top_k_accuracy": {
            "description": (
                "Per-contract recall: fraction of GT clause types correctly predicted. "
                "Contracts with zero GT annotations excluded."
            ),
            "aggregate": {
                "mean": topk_mean,
                "min": round(float(min(topk_recalls)), 4) if topk_recalls else 0.0,
                "max": round(float(max(topk_recalls)), 4) if topk_recalls else 0.0,
                "median": round(float(np.median(topk_recalls)), 4) if topk_recalls else 0.0,
                "n_contracts": len(topk_recalls),
            },
        },
        "per_class_metrics": per_class,
        "per_contract_top_k": per_contract,
        "cuad_41_categories": CUAD_41_CATEGORIES,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {output_path}")
    print("\nComparison with keyword-only baseline (Macro F1=0.2380):")
    delta = macro_f1 - 0.2380
    sign = "+" if delta >= 0 else ""
    print(f"  Hybrid Macro F1 = {macro_f1:.4f}  ({sign}{delta:.4f} vs baseline)")


if __name__ == "__main__":
    main()
