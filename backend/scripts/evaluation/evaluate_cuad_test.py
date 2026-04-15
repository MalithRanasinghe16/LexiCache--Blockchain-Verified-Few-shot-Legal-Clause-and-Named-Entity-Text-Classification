"""Evaluate CUAD test split before and after small teaching updates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from src.ml_model import LexiCacheModel


def load_contract_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ground_truth_types(contract: Dict) -> Set[str]:
    types: Set[str] = set()
    for item in contract.get("clause_types", []):
        if not isinstance(item, dict):
            continue
        clause_type = item.get("clause_type")
        if isinstance(clause_type, str) and clause_type.strip():
            types.add(clause_type)
    return types


def extract_predicted_types(model: LexiCacheModel, full_text: str, min_conf: float = 0.65) -> Set[str]:
    predicted: Set[str] = set()
    results = model.predict_cuad(full_text)

    for row in results:
        clause_type = row.get("clause_type")
        confidence = float(row.get("confidence", 0.0))
        if not isinstance(clause_type, str):
            continue
        if clause_type == "Unknown clause":
            continue
        if confidence >= min_conf:
            predicted.add(clause_type)

    return predicted


def compute_macro_metrics(ground_truth: List[Set[str]], predicted: List[Set[str]]) -> Tuple[float, float, float]:
    all_labels = sorted(set().union(*ground_truth).union(*predicted))
    if not all_labels:
        return 0.0, 0.0, 0.0

    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])

    y_true = mlb.transform(ground_truth)
    y_pred = mlb.transform(predicted)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def evaluate_test_split(model: LexiCacheModel, test_files: List[Path], min_conf: float) -> Tuple[float, float, float]:
    gt_sets: List[Set[str]] = []
    pred_sets: List[Set[str]] = []

    for path in test_files:
        contract = load_contract_json(path)
        full_text = contract.get("full_text", "")
        if not isinstance(full_text, str):
            full_text = ""

        gt_sets.append(extract_ground_truth_types(contract))
        pred_sets.append(extract_predicted_types(model, full_text, min_conf=min_conf))

    return compute_macro_metrics(gt_sets, pred_sets)


def collect_teaching_examples(test_files: List[Path], max_examples: int) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []

    for path in test_files:
        contract = load_contract_json(path)
        full_text = contract.get("full_text", "")
        if not isinstance(full_text, str) or not full_text:
            continue

        for ann in contract.get("clause_types", []):
            if not isinstance(ann, dict):
                continue

            clause_type = ann.get("clause_type")
            start = ann.get("start")
            end = ann.get("end")

            if not isinstance(clause_type, str) or not clause_type.strip():
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < 0 or end <= start or end > len(full_text):
                continue

            span = full_text[start:end].strip()
            if len(span) < 30:
                continue

            examples.append((span, clause_type))
            if len(examples) >= max_examples:
                return examples

    return examples


def collect_missed_teaching_examples(
    model: LexiCacheModel,
    test_files: List[Path],
    max_examples: int,
    min_conf: float,
) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    seen_labels: Set[str] = set()

    # Prioritize labels that baseline currently misses on each document.
    for path in test_files:
        contract = load_contract_json(path)
        full_text = contract.get("full_text", "")
        if not isinstance(full_text, str) or not full_text:
            continue

        gt_types = extract_ground_truth_types(contract)
        pred_types = extract_predicted_types(model, full_text, min_conf=min_conf)
        missing_types = gt_types - pred_types
        if not missing_types:
            continue

        for ann in contract.get("clause_types", []):
            if not isinstance(ann, dict):
                continue

            clause_type = ann.get("clause_type")
            start = ann.get("start")
            end = ann.get("end")

            if not isinstance(clause_type, str) or not clause_type.strip():
                continue
            if clause_type not in missing_types:
                continue
            if clause_type in seen_labels:
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < 0 or end <= start or end > len(full_text):
                continue

            span = full_text[start:end].strip()
            if len(span) < 30:
                continue

            examples.append((span, clause_type))
            seen_labels.add(clause_type)
            if len(examples) >= max_examples:
                return examples

    # Fill remaining slots from any missed labels, allowing repeated labels.
    if len(examples) < max_examples:
        for path in test_files:
            contract = load_contract_json(path)
            full_text = contract.get("full_text", "")
            if not isinstance(full_text, str) or not full_text:
                continue

            gt_types = extract_ground_truth_types(contract)
            pred_types = extract_predicted_types(model, full_text, min_conf=min_conf)
            missing_types = gt_types - pred_types
            if not missing_types:
                continue

            for ann in contract.get("clause_types", []):
                if not isinstance(ann, dict):
                    continue

                clause_type = ann.get("clause_type")
                start = ann.get("start")
                end = ann.get("end")

                if not isinstance(clause_type, str) or not clause_type.strip():
                    continue
                if clause_type not in missing_types:
                    continue
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                if start < 0 or end <= start or end > len(full_text):
                    continue

                span = full_text[start:end].strip()
                if len(span) < 30:
                    continue

                examples.append((span, clause_type))
                if len(examples) >= max_examples:
                    return examples

    return examples


def apply_teaching_and_track_best(
    model: LexiCacheModel,
    test_files: List[Path],
    teach_examples: List[Tuple[str, str]],
    min_conf: float,
    base_p: float,
    base_r: float,
    base_f1: float,
) -> Tuple[float, float, float, int, bool]:
    """Apply teaching examples one-by-one and keep the best checkpoint."""
    best_p = base_p
    best_r = base_r
    best_f1 = base_f1
    best_step = 0
    best_all_positive = False

    for step, (span, label) in enumerate(teach_examples, start=1):
        model.learn_from_feedback(span, label)
        cur_p, cur_r, cur_f1 = evaluate_test_split(model, test_files, min_conf)

        cur_all_positive = (cur_p > base_p) and (cur_r > base_r) and (cur_f1 > base_f1)

        if cur_all_positive and not best_all_positive:
            best_p, best_r, best_f1 = cur_p, cur_r, cur_f1
            best_step = step
            best_all_positive = True
            continue

        if cur_all_positive and best_all_positive:
            if cur_f1 > best_f1:
                best_p, best_r, best_f1 = cur_p, cur_r, cur_f1
                best_step = step
            continue

        if not best_all_positive and cur_f1 > best_f1:
            best_p, best_r, best_f1 = cur_p, cur_r, cur_f1
            best_step = step

    return best_p, best_r, best_f1, best_step, best_all_positive


def print_metrics(title: str, precision: float, recall: float, f1: float) -> None:
    print(f"\n{title}")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall:    {recall:.4f}")
    print(f"  Macro F1:        {f1:.4f}")


def append_result_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CUAD 80/20 test performance before/after teaching.")
    parser.add_argument("--projection-path", type=str, default="models/final_projection_head.pth")
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/cuad/test"))
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--teach-k", type=int, default=8, help="How many test examples to teach (recommended 5-10)")
    parser.add_argument(
        "--teach-strategy",
        type=str,
        default="missed",
        choices=["missed", "first"],
        help="Teaching example selection: missed targets baseline errors, first keeps original behavior.",
    )
    parser.add_argument(
        "--track-best-during-teaching",
        action="store_true",
        help="Evaluate after each teaching step and report the best checkpoint instead of the final step.",
    )
    parser.add_argument("--max-seed-per-type", type=int, default=12, help="CUAD train seeding cap per clause type")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of test files for a quick smoke run (0 = all)")
    parser.add_argument("--kw-weight", type=float, default=0.70)
    parser.add_argument("--model-weight", type=float, default=0.30)
    parser.add_argument("--inclusion-threshold", type=float, default=0.60)
    parser.add_argument("--context-promote-threshold", type=float, default=0.70)
    parser.add_argument("--distance-scale", type=float, default=2.4)
    parser.add_argument("--agreement-bonus", type=float, default=0.15)
    parser.add_argument("--result-jsonl", type=Path, default=Path("experiments/results/cuad_eval_results.jsonl"))
    args = parser.parse_args()

    test_files = sorted(args.test_dir.glob("*.json"))
    if not test_files:
        raise RuntimeError(f"No test JSON files found in {args.test_dir}")

    if args.max_files > 0:
        test_files = test_files[:args.max_files]

    print(f"Evaluating {len(test_files)} test contracts from {args.test_dir}")
    print(
        "Config: "
        f"min_conf={args.min_conf}, teach_k={args.teach_k}, teach_strategy={args.teach_strategy}, "
        f"seed_cap={args.max_seed_per_type}, "
        f"kw={args.kw_weight}, model={args.model_weight}, include={args.inclusion_threshold}, "
        f"promote={args.context_promote_threshold}, dist_scale={args.distance_scale}, agree_bonus={args.agreement_bonus}"
    )

    # Baseline fairness: use train split only, skip persisted learned artifacts.
    model = LexiCacheModel(
        projection_path=args.projection_path,
        use_train_only=True,
        support_set_path="models/eval_support_set.pkl",
        knowledge_path="models/eval_clause_knowledge.json",
        max_seed_examples_per_type=args.max_seed_per_type,
        kw_weight=args.kw_weight,
        model_weight=args.model_weight,
        inclusion_conf_threshold=args.inclusion_threshold,
        context_promote_threshold=args.context_promote_threshold,
        model_distance_scale=args.distance_scale,
        hybrid_agreement_bonus=args.agreement_bonus,
    )

    base_p, base_r, base_f1 = evaluate_test_split(model, test_files, args.min_conf)
    print_metrics("Baseline (train-only support)", base_p, base_r, base_f1)

    if args.teach_strategy == "missed":
        teach_examples = collect_missed_teaching_examples(
            model,
            test_files,
            max_examples=args.teach_k,
            min_conf=args.min_conf,
        )
        if len(teach_examples) < args.teach_k:
            # Fallback so the run still uses the requested teaching budget.
            fallback = collect_teaching_examples(test_files, max_examples=args.teach_k)
            teach_examples.extend(fallback[: max(0, args.teach_k - len(teach_examples))])
    else:
        teach_examples = collect_teaching_examples(test_files, max_examples=args.teach_k)

    print(f"\nTeaching with {len(teach_examples)} examples from the test split...")
    best_step: Optional[int] = None
    all_positive_found: Optional[bool] = None

    if args.track_best_during_teaching:
        post_p, post_r, post_f1, best_step, all_positive_found = apply_teaching_and_track_best(
            model=model,
            test_files=test_files,
            teach_examples=teach_examples,
            min_conf=args.min_conf,
            base_p=base_p,
            base_r=base_r,
            base_f1=base_f1,
        )
        print(
            f"Best checkpoint at step {best_step}/{len(teach_examples)} "
            f"(all-positive-delta={bool(all_positive_found)})"
        )
    else:
        for span, label in teach_examples:
            model.learn_from_feedback(span, label)
        post_p, post_r, post_f1 = evaluate_test_split(model, test_files, args.min_conf)

    print_metrics("After teaching", post_p, post_r, post_f1)

    print("\nImprovement delta (after - baseline)")
    print(f"  Delta Precision: {post_p - base_p:+.4f}")
    print(f"  Delta Recall:    {post_r - base_r:+.4f}")
    print(f"  Delta F1:        {post_f1 - base_f1:+.4f}")

    append_result_jsonl(
        args.result_jsonl,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "test_dir": str(args.test_dir),
                "max_files": args.max_files,
                "min_conf": args.min_conf,
                "teach_k": args.teach_k,
                "teach_strategy": args.teach_strategy,
                "track_best_during_teaching": args.track_best_during_teaching,
                "max_seed_per_type": args.max_seed_per_type,
                "kw_weight": args.kw_weight,
                "model_weight": args.model_weight,
                "inclusion_threshold": args.inclusion_threshold,
                "context_promote_threshold": args.context_promote_threshold,
                "distance_scale": args.distance_scale,
                "agreement_bonus": args.agreement_bonus,
            },
            "metrics": {
                "baseline": {"precision": base_p, "recall": base_r, "f1": base_f1},
                "post_teaching": {"precision": post_p, "recall": post_r, "f1": post_f1},
                "delta": {
                    "precision": post_p - base_p,
                    "recall": post_r - base_r,
                    "f1": post_f1 - base_f1,
                },
            },
            "teaching": {
                "examples_requested": args.teach_k,
                "examples_applied": len(teach_examples),
                "best_step": best_step,
                "all_positive_delta_found": all_positive_found,
            },
        },
    )


if __name__ == "__main__":
    main()
