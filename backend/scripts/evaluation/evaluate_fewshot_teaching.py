"""
Few-Shot Teaching Accuracy Evaluation for LexiCache
=====================================================
Evaluates how well the model adapts to unknown/rare clause types
when users provide N teaching examples (1-10 shot).

Experimental Design
-------------------
- Six target clause types are treated as completely unknown:
    4 with F1=0.0 in standard evaluation  +  2 borderline (F1<0.35)
- Each target type is stripped from the seed support set before the experiment
  (zero-knowledge baseline — model starts with NO examples of these types).
- Teaching examples are drawn from the CUAD *train* split.
- Test examples are drawn from the CUAD *test* split (never seen during teaching).
- For each shot count N in SHOT_COUNTS:
    - N_TRIALS independent trials are run (different random teach subsets each time)
    - Per trial: teach N spans per class → evaluate on ALL test spans
- Metrics: per-class F1 / Precision / Recall, Macro F1, Micro F1, avg confidence
- Charts saved to experiments/results/fewshot_teaching/

Usage
-----
    python scripts/evaluation/evaluate_fewshot_teaching.py
    python scripts/evaluation/evaluate_fewshot_teaching.py --train-dir data/processed/cuad/train
                                                           --test-dir  data/processed/cuad/test
                                                           --max-shots 10 --trials 5
"""

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")          # headless — safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
# project root on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ml_model import LexiCacheModel           # noqa: E402
from src.data import normalize_text               # noqa: E402
# Experiment configuration
TARGET_TYPES: List[str] = [
    # F1 = 0.0 in full evaluation (true zero-shot failures)
    "Most Favored Nation",
    "Price Restrictions",
    "Unlimited/All-You-Can-Eat-License",
    "Source Code Escrow",
    # Borderline (F1 < 0.35) — included for stronger thesis evidence
    "Non-Disparagement",
    "Notice Period To Terminate Renewal",
]

SHOT_COUNTS: List[int] = [0, 1, 2, 3, 4, 5, 7, 10]
N_TRIALS: int = 5
RANDOM_SEED: int = 42
# Data helpers
def load_spans(data_dir: Path, target_types: List[str]) -> Dict[str, List[str]]:
    """Return {clause_type: [span_text, ...]} for every target type found."""
    type_to_spans: Dict[str, List[str]] = {t: [] for t in target_types}
    for path in sorted(data_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            continue

        full_text: str = doc.get("full_text", "")
        for ann in doc.get("clause_types", []):
            ct = ann.get("clause_type", "")
            if ct not in type_to_spans:
                continue
            start = ann.get("start", 0)
            end = ann.get("end", 0)
            span = full_text[start:end].strip()
            if len(span) >= 30:
                type_to_spans[ct].append(span)

    return type_to_spans
# Model surgery helpers
def strip_target_types(model: LexiCacheModel, target_types: List[str]) -> None:
    """Remove all seed support examples for the target clause types in-place."""
    keep = [
        i for i, lbl in enumerate(model.support_labels)
        if lbl not in target_types
    ]
    model.support_embeddings = [model.support_embeddings[i] for i in keep]
    model.support_labels     = [model.support_labels[i]     for i in keep]
    model.support_texts      = [model.support_texts[i]      for i in keep]
    model.support_sources    = [model.support_sources[i]    for i in keep]


def snapshot(model: LexiCacheModel) -> Tuple[list, list, list, list, dict, dict, int]:
    """Capture a cheap copy of the mutable support-set state."""
    return (
        list(model.support_embeddings),
        list(model.support_labels),
        list(model.support_texts),
        list(model.support_sources),
        dict(model.label_to_id),
        copy.deepcopy(model.learned_types),
        model.next_label_id,
    )


def restore(model: LexiCacheModel,
            state: Tuple[list, list, list, list, dict, dict, int]) -> None:
    """Restore model to a previously snapshotted state (no disk I/O)."""
    (model.support_embeddings,
     model.support_labels,
     model.support_texts,
     model.support_sources,
     model.label_to_id,
     model.learned_types,
     model.next_label_id) = state


def teach_in_memory(model: LexiCacheModel, text: str, label: str) -> None:
    """
    Teach one example without touching disk.
    Mirrors learn_from_feedback() but skips _save_support_set / _save_knowledge_base.
    """
    normalized = normalize_text(text)
    with torch.no_grad():
        emb  = model.model([normalized], batch_size=1)
        proj = model.projection(emb.to(model.model.device))

    if label not in model.label_to_id:
        model.label_to_id[label] = model.next_label_id
        model.next_label_id += 1

    if label not in model.learned_types:
        model.learned_types[label] = {
            "examples": [], "count": 0,
            "first_learned": datetime.now().isoformat()
        }

    model.support_embeddings.append(proj.squeeze(0).cpu())
    model.support_labels.append(label)
    model.support_texts.append(text)
    model.support_sources.append("learned")

    entry = model.learned_types[label]
    entry["count"] = entry.get("count", 0) + 1
    entry.setdefault("examples", []).append(text[:200])
# Evaluation helper
def evaluate(model: LexiCacheModel,
             test_spans: Dict[str, List[str]]) -> Dict:
    """
    Classify every test span for every target type and compute metrics.

    Returns a dict with:
        per_class[type] = {precision, recall, f1, avg_confidence, tp, fp, fn}
        macro_f1, macro_precision, macro_recall
        micro_f1, micro_precision, micro_recall
        avg_confidence   (over all correctly predicted spans)
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []

    for true_label, spans in test_spans.items():
        for span in spans:
            result = model._classify_segment(span)
            y_true.append(true_label)
            y_pred.append(result["clause_type"])
            confidences.append(result["confidence"])

    target_types = list(test_spans.keys())
    per_class: Dict[str, Dict] = {}

    total_tp = total_fp = total_fn = 0

    for t in target_types:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == t and b == t)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != t and b == t)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == t and b != t)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        correct_confs = [
            c for yt, yp, c in zip(y_true, y_pred, confidences)
            if yt == t and yp == t
        ]
        avg_conf = float(np.mean(correct_confs)) if correct_confs else 0.0

        per_class[t] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "avg_confidence": round(avg_conf, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro
    macro_p  = float(np.mean([per_class[t]["precision"] for t in target_types]))
    macro_r  = float(np.mean([per_class[t]["recall"]    for t in target_types]))
    macro_f1 = float(np.mean([per_class[t]["f1"]        for t in target_types]))

    # Micro
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    all_correct_confs = [
        c for yt, yp, c in zip(y_true, y_pred, confidences) if yt == yp
    ]
    avg_conf_all = float(np.mean(all_correct_confs)) if all_correct_confs else 0.0

    return {
        "per_class":        per_class,
        "macro_f1":         round(macro_f1, 4),
        "macro_precision":  round(macro_p,  4),
        "macro_recall":     round(macro_r,  4),
        "micro_f1":         round(micro_f1, 4),
        "micro_precision":  round(micro_p,  4),
        "micro_recall":     round(micro_r,  4),
        "avg_confidence":   round(avg_conf_all, 4),
    }
# Charting
def _shot_axis(shot_counts: List[int]) -> List[str]:
    return [str(s) for s in shot_counts]


def plot_macro_f1(results: Dict, out_dir: Path) -> None:
    """Chart 1: Shot-count vs Macro F1 with ±1 std-dev error bands."""
    shots = sorted(results.keys())
    means, stds = [], []
    for n in shots:
        vals = [r["macro_f1"] for r in results[n]]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shots, means, "o-", color="#2563EB", linewidth=2, markersize=7, label="Macro F1")
    ax.fill_between(shots,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.18, color="#2563EB")
    ax.axhline(means[0], linestyle="--", color="gray", linewidth=1, label="0-shot baseline")
    ax.set_xlabel("Teaching Examples per Class (N-shot)", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Few-Shot Teaching: Macro F1 vs Shot Count\n(zero-knowledge start, 6 rare clause types)", fontsize=13)
    ax.set_xticks(shots)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "chart1_macro_f1_vs_shots.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'chart1_macro_f1_vs_shots.png'}")


def plot_precision_recall(results: Dict, out_dir: Path) -> None:
    """Chart 2: Macro Precision & Recall vs Shot count."""
    shots = sorted(results.keys())
    prec_means, prec_stds = [], []
    rec_means,  rec_stds  = [], []

    for n in shots:
        p_vals = [r["macro_precision"] for r in results[n]]
        r_vals = [r["macro_recall"]    for r in results[n]]
        prec_means.append(float(np.mean(p_vals)))
        prec_stds.append(float(np.std(p_vals)))
        rec_means.append(float(np.mean(r_vals)))
        rec_stds.append(float(np.std(r_vals)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shots, prec_means, "s-", color="#16A34A", linewidth=2,
            markersize=7, label="Macro Precision")
    ax.fill_between(shots,
                    [m - s for m, s in zip(prec_means, prec_stds)],
                    [m + s for m, s in zip(prec_means, prec_stds)],
                    alpha=0.15, color="#16A34A")
    ax.plot(shots, rec_means, "^-", color="#DC2626", linewidth=2,
            markersize=7, label="Macro Recall")
    ax.fill_between(shots,
                    [m - s for m, s in zip(rec_means, rec_stds)],
                    [m + s for m, s in zip(rec_means, rec_stds)],
                    alpha=0.15, color="#DC2626")
    ax.set_xlabel("Teaching Examples per Class (N-shot)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Few-Shot Teaching: Precision & Recall vs Shot Count", fontsize=13)
    ax.set_xticks(shots)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "chart2_precision_recall_vs_shots.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'chart2_precision_recall_vs_shots.png'}")


def plot_per_class_f1_heatmap(results: Dict, target_types: List[str], out_dir: Path) -> None:
    """Chart 3: Heatmap — class (row) × shot-count (col) → mean F1."""
    shots = sorted(results.keys())
    matrix = np.zeros((len(target_types), len(shots)))

    for j, n in enumerate(shots):
        for i, t in enumerate(target_types):
            vals = [r["per_class"][t]["f1"] for r in results[n]]
            matrix[i, j] = float(np.mean(vals))

    short_labels = [
        t.replace("Unlimited/All-You-Can-Eat-License", "Unlimited License")
         .replace("Notice Period To Terminate Renewal", "Notice Period")
        for t in target_types
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(shots)))
    ax.set_xticklabels([str(s) for s in shots], fontsize=11)
    ax.set_yticks(range(len(target_types)))
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("N-shot (teaching examples per class)", fontsize=12)
    ax.set_title("Per-Class F1 Heatmap vs Shot Count", fontsize=13)

    for i in range(len(target_types)):
        for j in range(len(shots)):
            val = matrix[i, j]
            color = "black" if 0.2 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="F1 Score")
    fig.tight_layout()
    fig.savefig(out_dir / "chart3_per_class_f1_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'chart3_per_class_f1_heatmap.png'}")


def plot_confidence(results: Dict, out_dir: Path) -> None:
    """Chart 4: Average prediction confidence vs Shot count (overall + micro F1)."""
    shots = sorted(results.keys())
    conf_means, conf_stds = [], []
    micro_means = []

    for n in shots:
        c_vals = [r["avg_confidence"] for r in results[n]]
        m_vals = [r["micro_f1"]       for r in results[n]]
        conf_means.append(float(np.mean(c_vals)))
        conf_stds.append(float(np.std(c_vals)))
        micro_means.append(float(np.mean(m_vals)))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_conf  = "#7C3AED"
    color_micro = "#0891B2"

    ax1.plot(shots, conf_means, "D-", color=color_conf, linewidth=2,
             markersize=7, label="Avg Confidence (correct preds)")
    ax1.fill_between(shots,
                     [m - s for m, s in zip(conf_means, conf_stds)],
                     [m + s for m, s in zip(conf_means, conf_stds)],
                     alpha=0.15, color=color_conf)
    ax1.set_xlabel("Teaching Examples per Class (N-shot)", fontsize=12)
    ax1.set_ylabel("Confidence", fontsize=12, color=color_conf)
    ax1.tick_params(axis="y", labelcolor=color_conf)
    ax1.set_xticks(shots)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(shots, micro_means, "o--", color=color_micro, linewidth=2,
             markersize=7, label="Micro F1")
    ax2.set_ylabel("Micro F1", fontsize=12, color=color_micro)
    ax2.tick_params(axis="y", labelcolor=color_micro)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")
    ax1.set_title("Confidence & Micro F1 Progression vs Shot Count", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "chart4_confidence_micro_f1.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'chart4_confidence_micro_f1.png'}")
# Main experiment
def run(args: argparse.Namespace) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = ROOT / "experiments" / "results" / "fewshot_teaching"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = ROOT / args.train_dir
    test_dir  = ROOT / args.test_dir
    max_shots = args.max_shots
    n_trials  = args.trials

    shot_counts = [s for s in SHOT_COUNTS if s <= max_shots]

    print("=" * 72)
    print("LexiCache — Few-Shot Teaching Accuracy Evaluation")
    print(f"  Train dir : {train_dir}")
    print(f"  Test  dir : {test_dir}")
    print(f"  Shots     : {shot_counts}")
    print(f"  Trials    : {n_trials}")
    print(f"  Targets   : {len(TARGET_TYPES)} clause types")
    print("=" * 72)
    # Load CUAD spans
    print("\nLoading CUAD spans...")
    train_spans = load_spans(train_dir, TARGET_TYPES)
    test_spans  = load_spans(test_dir,  TARGET_TYPES)

    print("\nData availability:")
    print(f"  {'Clause Type':<45} {'Train':>6} {'Test':>6}")
    print("  " + "-" * 60)
    usable_types = []
    for t in TARGET_TYPES:
        n_tr = len(train_spans[t])
        n_te = len(test_spans[t])
        flag = ""
        if n_tr < max_shots:
            flag = f"  *** only {n_tr} train examples — max shots capped at {n_tr}"
        if n_te == 0:
            flag = "  *** SKIPPED (no test examples)"
        print(f"  {t:<45} {n_tr:>6} {n_te:>6}{flag}")
        if n_te > 0:
            usable_types.append(t)

    # Filter test_spans to only usable types
    test_spans_filtered = {t: test_spans[t] for t in usable_types}
    # Load model (use_train_only=True → no saved learned examples)
    print("\nLoading LexiCacheModel...")
    model = LexiCacheModel(use_train_only=True)

    # Strip target types from the CUAD seed support set
    before = len(model.support_labels)
    strip_target_types(model, TARGET_TYPES)
    after = len(model.support_labels)
    print(f"\nStripped {before - after} seed examples for target types "
          f"({after} seed examples remain).")

    # Save the baseline (stripped) state — restored between every trial
    baseline = snapshot(model)
    # Experiment loop
    # results[shot_count] = List[metrics_dict]  (one entry per trial)
    results: Dict[int, List[Dict]] = {n: [] for n in shot_counts}

    for n_shot in shot_counts:
        print(f"\n{'─'*60}")
        print(f"  N-SHOT = {n_shot}  ({n_trials} trials)")
        print(f"{'─'*60}")

        for trial in range(n_trials):
            restore(model, baseline)

            # Teach N examples per target type
            if n_shot > 0:
                for t in usable_types:
                    pool = train_spans[t]
                    cap  = min(n_shot, len(pool))
                    chosen = random.sample(pool, cap)
                    for span in chosen:
                        teach_in_memory(model, span, t)

            # Evaluate on test spans
            metrics = evaluate(model, test_spans_filtered)
            results[n_shot].append(metrics)

            # Print trial summary
            print(f"    trial {trial+1}/{n_trials} → "
                  f"macro_f1={metrics['macro_f1']:.4f}  "
                  f"micro_f1={metrics['micro_f1']:.4f}  "
                  f"conf={metrics['avg_confidence']:.4f}")

        # Aggregate across trials
        trial_macro = [r["macro_f1"] for r in results[n_shot]]
        print(f"  → mean macro F1 = {np.mean(trial_macro):.4f} "
              f"± {np.std(trial_macro):.4f}")
    # Print final summary table
    print("\n" + "=" * 90)
    print("FEW-SHOT TEACHING RESULTS — SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Shots':>6}  {'Macro F1':>10}  {'Macro P':>10}  "
    header += f"{'Macro R':>10}  {'Micro F1':>10}  {'Avg Conf':>10}"
    print(header)
    print("-" * 90)
    for n in shot_counts:
        mf1  = np.mean([r["macro_f1"]        for r in results[n]])
        mp   = np.mean([r["macro_precision"]  for r in results[n]])
        mr   = np.mean([r["macro_recall"]     for r in results[n]])
        mif1 = np.mean([r["micro_f1"]         for r in results[n]])
        conf = np.mean([r["avg_confidence"]   for r in results[n]])
        print(f"{n:>6}  {mf1:>10.4f}  {mp:>10.4f}  {mr:>10.4f}  {mif1:>10.4f}  {conf:>10.4f}")

    # Per-class breakdown at max shot
    max_n = max(shot_counts)
    print(f"\nPer-class F1 at {max_n}-shot (mean over {n_trials} trials):")
    print(f"  {'Clause Type':<45} {'F1':>6}  {'P':>6}  {'R':>6}  {'Conf':>6}")
    print("  " + "-" * 70)
    for t in usable_types:
        f1s = [r["per_class"][t]["f1"]        for r in results[max_n]]
        ps  = [r["per_class"][t]["precision"] for r in results[max_n]]
        rs  = [r["per_class"][t]["recall"]    for r in results[max_n]]
        cs  = [r["per_class"][t]["avg_confidence"] for r in results[max_n]]
        print(f"  {t:<45} {np.mean(f1s):>6.4f}  "
              f"{np.mean(ps):>6.4f}  {np.mean(rs):>6.4f}  {np.mean(cs):>6.4f}")
    # Save JSON results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Few-shot teaching evaluation: zero-knowledge start, "
            "6 rare/unknown clause types, N-shot teach from CUAD train, "
            "test on CUAD test."
        ),
        "config": {
            "target_types": usable_types,
            "shot_counts":  shot_counts,
            "n_trials":     n_trials,
            "random_seed":  RANDOM_SEED,
        },
        "data_counts": {
            t: {"train": len(train_spans[t]), "test": len(test_spans[t])}
            for t in TARGET_TYPES
        },
        "results_by_shot": {
            str(n): {
                "mean_macro_f1":    round(float(np.mean([r["macro_f1"]       for r in results[n]])), 4),
                "std_macro_f1":     round(float(np.std( [r["macro_f1"]       for r in results[n]])), 4),
                "mean_macro_prec":  round(float(np.mean([r["macro_precision"] for r in results[n]])), 4),
                "mean_macro_rec":   round(float(np.mean([r["macro_recall"]    for r in results[n]])), 4),
                "mean_micro_f1":    round(float(np.mean([r["micro_f1"]        for r in results[n]])), 4),
                "mean_avg_conf":    round(float(np.mean([r["avg_confidence"]  for r in results[n]])), 4),
                "per_class_mean_f1": {
                    t: round(float(np.mean([r["per_class"][t]["f1"] for r in results[n]])), 4)
                    for t in usable_types
                },
                "trials": results[n],
            }
            for n in shot_counts
        },
    }

    json_path = out_dir / "fewshot_teaching_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nJSON results saved to: {json_path}")
    # Generate charts
    print("\nGenerating charts...")
    plot_macro_f1(results, out_dir)
    plot_precision_recall(results, out_dir)
    plot_per_class_f1_heatmap(results, usable_types, out_dir)
    plot_confidence(results, out_dir)

    print(f"\nAll charts saved to: {out_dir}")
    print("\nDone.")
# Entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate few-shot teaching accuracy for rare/unknown clause types"
    )
    parser.add_argument(
        "--train-dir", default="data/processed/cuad/train",
        help="CUAD train split directory (teaching examples)"
    )
    parser.add_argument(
        "--test-dir", default="data/processed/cuad/test",
        help="CUAD test split directory (evaluation examples)"
    )
    parser.add_argument(
        "--max-shots", type=int, default=10,
        help="Maximum N-shot to test (default: 10)"
    )
    parser.add_argument(
        "--trials", type=int, default=N_TRIALS,
        help=f"Number of random trials per shot count (default: {N_TRIALS})"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
