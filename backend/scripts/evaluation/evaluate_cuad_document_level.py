"""Document-level multi-label clause classification evaluation on CUAD test split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer

from src.ml_model import LexiCacheModel
# Official CUAD 41 clause categories
CUAD_41_CATEGORIES: List[str] = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "ROFR/ROFO/ROFN",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Indemnification",
]
# Explicit alias map: model type names → CUAD 41 canonical names
MODEL_TYPE_ALIASES: Dict[str, Optional[str]] = {
    # Model uses generic "Termination"; CUAD 41 has "Termination For Convenience"
    "termination": "Termination For Convenience",
    "termination for convenience": "Termination For Convenience",
    # Model uses "Cap on Liability"; CUAD 41 uses "Cap On Liability" (Title Case)
    "cap on liability": "Cap On Liability",
    # Model uses "Limitation of Liability"; closest CUAD 41 equivalent is "Cap On Liability"
    "limitation of liability": "Cap On Liability",
    # Model uses "Notice Period to Terminate Renewal"; CUAD uses different casing
    "notice period to terminate renewal": "Notice Period To Terminate Renewal",
    # Model uses "IP Ownership Assignment"; CUAD uses "Ip Ownership Assignment"
    "ip ownership assignment": "Ip Ownership Assignment",
    # Model uses "Joint IP Ownership"; CUAD uses "Joint Ip Ownership"
    "joint ip ownership": "Joint Ip Ownership",
    # Model uses "No-Solicit of Customers"; CUAD uses "No-Solicit Of Customers"
    "no-solicit of customers": "No-Solicit Of Customers",
    "no-solicit of employees": "No-Solicit Of Employees",
    # Model uses "Covenant Not to Sue" (lowercase 'to'); CUAD uses "Covenant Not To Sue"
    "covenant not to sue": "Covenant Not To Sue",
    # Model uses "ROFR/ROFO/ROFN" — exact match but ensure it resolves
    "rofr/rofo/rofn": "ROFR/ROFO/ROFN",
    # Types the model predicts that have NO CUAD 41 equivalent → discard
    "confidentiality": None,
    "intellectual property": None,
    "payment terms": None,
    "dispute resolution": None,
    "notice": None,
    "definitions": None,
    "amendment": None,
    "representations and warranties": None,
    "relationship of parties": None,
    "jurisdiction": None,
    "force majeure": None,
    "severability": None,
    "entire agreement": None,
    "waiver": None,
    "counterparts": None,
    "assignment": None,
}
# Type-name mapping helpers
def build_cuad_lower_map(cuad_categories: List[str]) -> Dict[str, str]:
    """Return a {lowercase_name: canonical_name} lookup for CUAD categories."""
    return {cat.lower(): cat for cat in cuad_categories}


def map_to_cuad_canonical(
    predicted_type: str,
    cuad_lower_map: Dict[str, str],
    mapping_cache: Dict[str, Optional[str]],
    fuzzy_cutoff: float = 0.72,
) -> Optional[str]:
    """Map a model-predicted clause type name to a CUAD canonical category name."""
    if predicted_type in mapping_cache:
        return mapping_cache[predicted_type]

    lower_pred = predicted_type.lower()

    # 0. Explicit alias map (handles known systematic mismatches)
    if lower_pred in MODEL_TYPE_ALIASES:
        result = MODEL_TYPE_ALIASES[lower_pred]
        mapping_cache[predicted_type] = result
        return result

    # 1. Exact case-insensitive match
    if lower_pred in cuad_lower_map:
        result = cuad_lower_map[lower_pred]
        mapping_cache[predicted_type] = result
        return result

    # 2. Fuzzy match
    candidates = list(cuad_lower_map.keys())
    matches = get_close_matches(lower_pred, candidates, n=1, cutoff=fuzzy_cutoff)
    if matches:
        result = cuad_lower_map[matches[0]]
        mapping_cache[predicted_type] = result
        return result

    # 3. No match -- outside CUAD 41
    mapping_cache[predicted_type] = None
    return None


def log_type_mapping(mapping_cache: Dict[str, Optional[str]]) -> None:
    """Print the resolved type-name mapping for transparency."""
    print("\n-- Predicted-type -> CUAD canonical mapping " + "-" * 31)
    mapped = {k: v for k, v in mapping_cache.items() if v is not None}
    unmapped = [k for k, v in mapping_cache.items() if v is None]
    for pred, canon in sorted(mapped.items()):
        marker = "  " if pred.lower() == canon.lower() else "~ "
        print(f"  {marker}{pred!r:45s} -> {canon!r}")
    if unmapped:
        print(f"\n  [outside CUAD 41 -- ignored in evaluation]")
        for u in sorted(unmapped):
            print(f"    x {u!r}")
    print("-" * 75)
# Ground-truth extraction
def load_contract_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ground_truth_cuad_types(
    contract: Dict,
    cuad_lower_map: Dict[str, str],
) -> Set[str]:
    """Return the set of CUAD canonical clause types that have at least one non-empty annotation in *contract*."""
    types: Set[str] = set()
    for item in contract.get("clause_types", []):
        if not isinstance(item, dict):
            continue
        clause_type = item.get("clause_type")
        if not isinstance(clause_type, str) or not clause_type.strip():
            continue
        canonical = cuad_lower_map.get(clause_type.lower())
        if canonical:
            types.add(canonical)
    return types
# Prediction extraction
def extract_predicted_cuad_types(
    model: LexiCacheModel,
    full_text: str,
    cuad_lower_map: Dict[str, str],
    mapping_cache: Dict[str, Optional[str]],
    min_conf: float = 0.55,
    fuzzy_cutoff: float = 0.72,
) -> Set[str]:
    """Run model.predict_cuad() on *full_text* and return the set of CUAD canonical clause types predicted with confidence >= *min_conf*."""
    predicted: Set[str] = set()
    results = model.predict_cuad(full_text)

    for row in results:
        clause_type = row.get("clause_type")
        confidence = float(row.get("confidence", 0.0))

        if not isinstance(clause_type, str):
            continue
        if clause_type == "Unknown clause":
            continue
        if confidence < min_conf:
            continue

        canonical = map_to_cuad_canonical(
            clause_type, cuad_lower_map, mapping_cache, fuzzy_cutoff=fuzzy_cutoff
        )
        if canonical:
            predicted.add(canonical)

    return predicted
# Top-K accuracy (per-contract recall)
def compute_top_k_accuracy(
    ground_truth: List[Set[str]],
    predicted: List[Set[str]],
    contract_names: List[str],
) -> Dict:
    """For each contract, compute the fraction of ground-truth CUAD clause types that were correctly predicted by the model (per-contract recall)."""
    per_contract: List[Dict] = []

    for gt, pred, name in zip(ground_truth, predicted, contract_names):
        if not gt:
            per_contract.append({
                "contract": name,
                "gt_count": 0,
                "pred_count": len(pred),
                "found": 0,
                "recall": None,
                "note": "no_gt_annotations",
            })
            continue

        found = len(gt & pred)
        recall = found / len(gt)
        per_contract.append({
            "contract": name,
            "gt_count": len(gt),
            "pred_count": len(pred),
            "found": found,
            "recall": round(recall, 4),
        })

    valid_recalls = [r["recall"] for r in per_contract if r["recall"] is not None]

    if not valid_recalls:
        agg = {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "n_contracts": 0}
    else:
        agg = {
            "mean": float(np.mean(valid_recalls)),
            "min": float(np.min(valid_recalls)),
            "max": float(np.max(valid_recalls)),
            "median": float(np.median(valid_recalls)),
            "n_contracts": len(valid_recalls),
        }

    return {"aggregate": agg, "per_contract": per_contract}
# Console output helpers
def print_summary_table(
    report: Dict,
    macro_p: float,
    macro_r: float,
    macro_f1: float,
    micro_p: float,
    micro_r: float,
    micro_f1: float,
    h_loss: float,
    top_k: Dict,
    n_contracts: int,
    n_categories: int,
) -> None:
    """Print a formatted summary table to stdout."""
    W = 75
    print("\n" + "=" * W)
    print("  CUAD DOCUMENT-LEVEL MULTI-LABEL EVALUATION RESULTS")
    print("=" * W)
    print(f"  Test contracts evaluated : {n_contracts}")
    print(f"  CUAD categories (classes): {n_categories}")
    print("-" * W)

    print(f"\n  {'Clause Type':<42} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>8}")
    print("  " + "-" * 73)

    for label in CUAD_41_CATEGORIES:
        if label not in report:
            continue
        row = report[label]
        p = row.get("precision", 0.0)
        r = row.get("recall", 0.0)
        f = row.get("f1-score", 0.0)
        s = int(row.get("support", 0))
        print(f"  {label:<42} {p:>9.4f} {r:>9.4f} {f:>9.4f} {s:>8d}")

    print("  " + "-" * 73)
    print(f"\n  {'Macro  avg':<42} {macro_p:>9.4f} {macro_r:>9.4f} {macro_f1:>9.4f}")
    print(f"  {'Micro  avg':<42} {micro_p:>9.4f} {micro_r:>9.4f} {micro_f1:>9.4f}")
    print(f"\n  Hamming Loss             : {h_loss:.6f}")

    agg = top_k["aggregate"]
    print(f"\n  Top-K Accuracy (per-contract recall of GT clause types)")
    print(f"    Mean   : {agg['mean']:.4f}")
    print(f"    Median : {agg['median']:.4f}")
    print(f"    Min    : {agg['min']:.4f}")
    print(f"    Max    : {agg['max']:.4f}")
    print(f"    N      : {agg['n_contracts']} contracts with >=1 GT annotation")
    print("\n" + "=" * W + "\n")
# Result persistence
def save_results_json(path: Path, payload: Dict) -> None:
    """Write full evaluation results to a JSON file (overwrites if exists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved -> {path}")
# Main evaluation loop
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Document-level multi-label CUAD clause classification evaluation. "
            "Evaluates which of the 41 CUAD clause types are present in each "
            "test contract (binary multi-label per contract). "
            "Model weights/thresholds are taken from LexiCacheModel defaults."
        )
    )
    parser.add_argument(
        "--projection-path",
        type=str,
        default="models/final_projection_head.pth",
        help="Path to the trained projection head weights.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/processed/cuad/test"),
        help="Directory containing test-split contract JSON files.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.20,
        help="Minimum model confidence to accept a predicted clause type (default 0.20 for recall-oriented eval).",
    )
    parser.add_argument(
        "--fuzzy-cutoff",
        type=float,
        default=0.60,
        help="difflib.get_close_matches cutoff for fuzzy type-name mapping (default 0.60).",
    )
    parser.add_argument(
        "--max-seed-per-type",
        type=int,
        default=50,
        help="CUAD train seeding cap per clause type (default 50 matches LexiCacheModel default).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of test files (0 = all). Useful for quick smoke runs.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("experiments/results/cuad_document_level_results.json"),
        help="Output path for the full JSON results file.",
    )
    args = parser.parse_args()

    # -- Discover test files ------------------------------------------------
    test_files = sorted(args.test_dir.glob("*.json"))
    if not test_files:
        raise RuntimeError(f"No test JSON files found in {args.test_dir}")

    if args.max_files > 0:
        test_files = test_files[: args.max_files]

    print(f"\n{'=' * 75}")
    print("  CUAD DOCUMENT-LEVEL EVALUATION - SETUP")
    print(f"{'=' * 75}")
    print(f"  Test directory  : {args.test_dir}")
    print(f"  Test contracts  : {len(test_files)}")
    print(f"  CUAD categories : {len(CUAD_41_CATEGORIES)}")
    print(f"  min_conf        : {args.min_conf}")
    print(f"  fuzzy_cutoff    : {args.fuzzy_cutoff}")
    print(f"  Model weights   : using LexiCacheModel __init__ defaults")
    print(f"{'-' * 75}")

    # -- Load model (train-only for fair baseline evaluation) ---------------
    print("\n  Loading LexiCacheModel (train-only support set)...")
    model = LexiCacheModel(
        projection_path=args.projection_path,
        use_train_only=True,
        support_set_path="models/eval_support_set.pkl",
        knowledge_path="models/eval_clause_knowledge.json",
        max_seed_examples_per_type=args.max_seed_per_type,
    )

    # Print the actual config the model loaded with (from __init__ print line)
    print(
        f"  Active config   : kw={model.kw_weight:.2f}, model={model.model_weight:.2f}, "
        f"include>={model.inclusion_conf_threshold:.2f}, "
        f"fusion>={model.fusion_score_threshold:.2f}, "
        f"agreement_bonus={model.hybrid_agreement_bonus:.2f}"
    )
    print(f"{'-' * 75}")

    # -- Build type-name lookup structures ----------------------------------
    cuad_lower_map: Dict[str, str] = build_cuad_lower_map(CUAD_41_CATEGORIES)
    mapping_cache: Dict[str, Optional[str]] = {}

    # -- Evaluation loop ----------------------------------------------------
    ground_truth_sets: List[Set[str]] = []
    predicted_sets: List[Set[str]] = []
    contract_names: List[str] = []

    print(f"\n  Evaluating {len(test_files)} contracts...\n")

    for idx, path in enumerate(test_files, start=1):
        contract = load_contract_json(path)
        full_text = contract.get("full_text", "")
        if not isinstance(full_text, str):
            full_text = ""

        gt = extract_ground_truth_cuad_types(contract, cuad_lower_map)
        pred = extract_predicted_cuad_types(
            model,
            full_text,
            cuad_lower_map,
            mapping_cache,
            min_conf=args.min_conf,
            fuzzy_cutoff=args.fuzzy_cutoff,
        )

        ground_truth_sets.append(gt)
        predicted_sets.append(pred)
        contract_names.append(path.stem)

        # Progress indicator every 10 contracts (and always on last)
        if idx % 10 == 0 or idx == len(test_files):
            print(
                f"  [{idx:>4}/{len(test_files)}] {path.name[:55]:<55} "
                f"GT={len(gt):>2}  Pred={len(pred):>2}"
            )

    # -- Log the resolved type-name mapping --------------------------------
    log_type_mapping(mapping_cache)

    # -- Binarise with fixed CUAD 41 class set -----------------------------
    mlb = MultiLabelBinarizer(classes=CUAD_41_CATEGORIES)
    mlb.fit([CUAD_41_CATEGORIES])  # fit on full class list so all 41 are present

    y_true = mlb.transform(ground_truth_sets)
    y_pred = mlb.transform(predicted_sets)

    # -- sklearn metrics ----------------------------------------------------
    report: Dict = classification_report(
        y_true,
        y_pred,
        target_names=CUAD_41_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    h_loss: float = float(hamming_loss(y_true, y_pred))

    top_k_results = compute_top_k_accuracy(
        ground_truth_sets, predicted_sets, contract_names
    )

    # -- Per-class summary dict (for JSON) ---------------------------------
    per_class_metrics: Dict[str, Dict] = {}
    for label in CUAD_41_CATEGORIES:
        if label in report:
            per_class_metrics[label] = {
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
        else:
            per_class_metrics[label] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }

    # -- Print summary table ------------------------------------------------
    print_summary_table(
        report=report,
        macro_p=float(macro_p),
        macro_r=float(macro_r),
        macro_f1=float(macro_f1),
        micro_p=float(micro_p),
        micro_r=float(micro_r),
        micro_f1=float(micro_f1),
        h_loss=h_loss,
        top_k=top_k_results,
        n_contracts=len(test_files),
        n_categories=len(CUAD_41_CATEGORIES),
    )

    # -- Build full JSON payload --------------------------------------------
    payload: Dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Document-level multi-label CUAD clause classification evaluation. "
            "Binary label per (contract, clause_type): 1 if >=1 annotation exists / predicted."
        ),
        "config": {
            "test_dir": str(args.test_dir),
            "n_test_contracts": len(test_files),
            "n_cuad_categories": len(CUAD_41_CATEGORIES),
            "min_conf": args.min_conf,
            "fuzzy_cutoff": args.fuzzy_cutoff,
            "max_seed_per_type": args.max_seed_per_type,
            # Actual model weights (read from loaded model, not hardcoded)
            "kw_weight": model.kw_weight,
            "model_weight": model.model_weight,
            "inclusion_conf_threshold": model.inclusion_conf_threshold,
            "fusion_score_threshold": model.fusion_score_threshold,
            "hybrid_agreement_bonus": model.hybrid_agreement_bonus,
            "context_promote_threshold": model.context_promote_threshold,
            "model_distance_scale": model.model_distance_scale,
        },
        "aggregate_metrics": {
            "macro": {
                "precision": round(float(macro_p), 6),
                "recall": round(float(macro_r), 6),
                "f1": round(float(macro_f1), 6),
            },
            "micro": {
                "precision": round(float(micro_p), 6),
                "recall": round(float(micro_r), 6),
                "f1": round(float(micro_f1), 6),
            },
            "hamming_loss": round(h_loss, 6),
        },
        "top_k_accuracy": {
            "description": (
                "Per-contract recall: fraction of ground-truth CUAD clause types "
                "correctly predicted by the model. Contracts with zero GT annotations "
                "are excluded from aggregate statistics."
            ),
            "aggregate": top_k_results["aggregate"],
        },
        "per_class_metrics": per_class_metrics,
        "type_name_mapping": {
            "mapped": {k: v for k, v in mapping_cache.items() if v is not None},
            "unmapped_outside_cuad41": [k for k, v in mapping_cache.items() if v is None],
        },
        "cuad_41_categories": CUAD_41_CATEGORIES,
        "per_contract_top_k": top_k_results["per_contract"],
    }

    # -- Save to JSON -------------------------------------------------------
    save_results_json(args.result_json, payload)


if __name__ == "__main__":
    main()
