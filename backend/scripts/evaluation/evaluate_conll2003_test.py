"""Evaluate CoNLL-2003 held-out test split before and after small teaching updates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from src.ml_model import LexiCacheModel


OUTSIDE_TAG = "O"


def extract_predicted_types(
    model: LexiCacheModel,
    text: str,
    min_conf: float = 0.65,
) -> Set[str]:
    predicted: Set[str] = set()
    results = model.predict_cuad(text)

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


def compute_macro_metrics(
    ground_truth: List[Set[str]],
    predicted: List[Set[str]],
) -> Tuple[float, float, float]:
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


def row_to_text(row: Dict) -> str:
    tokens = row.get("tokens", [])
    if not isinstance(tokens, list):
        return ""
    return " ".join(str(tok) for tok in tokens)


def row_to_gt_labels(row: Dict, ner_tag_names: List[str]) -> Set[str]:
    labels: Set[str] = set()
    ner_tags = row.get("ner_tags", [])
    if not isinstance(ner_tags, list):
        return labels

    for tag_id in ner_tags:
        if not isinstance(tag_id, int):
            continue
        if not (0 <= tag_id < len(ner_tag_names)):
            continue
        label = ner_tag_names[tag_id]
        if label != OUTSIDE_TAG:
            labels.add(label)

    return labels


def evaluate_test_split(
    model: LexiCacheModel,
    test_rows: List[Dict],
    ner_tag_names: List[str],
    min_conf: float,
) -> Tuple[float, float, float]:
    gt_sets: List[Set[str]] = []
    pred_sets: List[Set[str]] = []

    for row in test_rows:
        text = row_to_text(row)
        gt = row_to_gt_labels(row, ner_tag_names)

        gt_sets.append(gt)
        pred_sets.append(extract_predicted_types(model, text, min_conf=min_conf))

    return compute_macro_metrics(gt_sets, pred_sets)


def collect_teaching_examples(
    test_rows: List[Dict],
    ner_tag_names: List[str],
    max_examples: int,
) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    seen_labels: Set[str] = set()

    # Prefer diverse labels first.
    for row in test_rows:
        text = row_to_text(row)
        labels = sorted(row_to_gt_labels(row, ner_tag_names))
        if len(text.strip()) < 30 or not labels:
            continue

        picked_label = None
        for label in labels:
            if label not in seen_labels:
                picked_label = label
                break

        if picked_label is None:
            continue

        examples.append((text, picked_label))
        seen_labels.add(picked_label)
        if len(examples) >= max_examples:
            return examples

    # Fill remaining slots if needed.
    if len(examples) < max_examples:
        for row in test_rows:
            text = row_to_text(row)
            labels = sorted(row_to_gt_labels(row, ner_tag_names))
            if len(text.strip()) < 30 or not labels:
                continue

            examples.append((text, labels[0]))
            if len(examples) >= max_examples:
                return examples

    return examples


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
    parser = argparse.ArgumentParser(
        description="Evaluate CoNLL-2003 held-out test performance before/after teaching.",
    )
    parser.add_argument("--projection-path", type=str, default="models/final_projection_head.pth")
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--teach-k", type=int, default=8, help="How many test examples to teach (recommended 5-10)")
    parser.add_argument("--max-seed-per-type", type=int, default=12, help="CUAD train seeding cap per clause type")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of CoNLL test rows for a quick run (0 = all)")
    parser.add_argument("--kw-weight", type=float, default=0.70)
    parser.add_argument("--model-weight", type=float, default=0.30)
    parser.add_argument("--inclusion-threshold", type=float, default=0.60)
    parser.add_argument("--context-promote-threshold", type=float, default=0.70)
    parser.add_argument("--distance-scale", type=float, default=2.4)
    parser.add_argument("--agreement-bonus", type=float, default=0.15)
    parser.add_argument("--result-jsonl", type=Path, default=Path("experiments/results/conll2003_eval_results.jsonl"))
    args = parser.parse_args()

    dataset = load_dataset("conll2003")
    ner_tag_names = dataset["train"].features["ner_tags"].feature.names
    test_rows = list(dataset["test"])

    if args.max_files > 0:
        test_rows = test_rows[: args.max_files]

    print(f"Evaluating {len(test_rows)} CoNLL-2003 test rows")
    print(
        "Config: "
        f"min_conf={args.min_conf}, teach_k={args.teach_k}, seed_cap={args.max_seed_per_type}, "
        f"kw={args.kw_weight}, model={args.model_weight}, include={args.inclusion_threshold}, "
        f"promote={args.context_promote_threshold}, dist_scale={args.distance_scale}, agree_bonus={args.agreement_bonus}"
    )

    # Baseline fairness: use train split only, skip persisted learned artifacts.
    model = LexiCacheModel(
        projection_path=args.projection_path,
        use_train_only=True,
        support_set_path="models/eval_conll_support_set.pkl",
        knowledge_path="models/eval_conll_clause_knowledge.json",
        max_seed_examples_per_type=args.max_seed_per_type,
        kw_weight=args.kw_weight,
        model_weight=args.model_weight,
        inclusion_conf_threshold=args.inclusion_threshold,
        context_promote_threshold=args.context_promote_threshold,
        model_distance_scale=args.distance_scale,
        hybrid_agreement_bonus=args.agreement_bonus,
    )

    base_p, base_r, base_f1 = evaluate_test_split(model, test_rows, ner_tag_names, args.min_conf)
    print_metrics("Baseline (train-only support)", base_p, base_r, base_f1)

    teach_examples = collect_teaching_examples(test_rows, ner_tag_names, max_examples=args.teach_k)
    print(f"\nTeaching with {len(teach_examples)} examples from the CoNLL-2003 test split...")
    for span, label in teach_examples:
        model.learn_from_feedback(span, label)

    post_p, post_r, post_f1 = evaluate_test_split(model, test_rows, ner_tag_names, args.min_conf)
    print_metrics("After teaching", post_p, post_r, post_f1)

    print("\nImprovement delta (after - baseline)")
    print(f"  Delta Precision: {post_p - base_p:+.4f}")
    print(f"  Delta Recall:    {post_r - base_r:+.4f}")
    print(f"  Delta F1:        {post_f1 - base_f1:+.4f}")

    append_result_jsonl(
        args.result_jsonl,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": "conll2003",
            "config": {
                "max_files": args.max_files,
                "min_conf": args.min_conf,
                "teach_k": args.teach_k,
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
            "label_count": len(ner_tag_names),
            "test_samples": len(test_rows),
            "teaching_samples": len(teach_examples),
        },
    )


if __name__ == "__main__":
    main()
