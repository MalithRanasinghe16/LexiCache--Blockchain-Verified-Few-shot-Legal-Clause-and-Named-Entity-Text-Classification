"""generate_eval_report.py Reads all existing LexiCache evaluation result JSONs and generates a comprehensive set of charts saved to experiments/results/eval_report/."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
# Paths
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
RESULTS_DIR = BASE_DIR / "experiments" / "results"
OUT_DIR = RESULTS_DIR / "eval_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINETUNED_JSON  = RESULTS_DIR / "cuad_hybrid_finetuned_results.json"
ABLATION_JSON   = RESULTS_DIR / "cuad_hybrid_ablation_results.json"
FEWSHOT_JSON    = RESULTS_DIR / "fewshot_teaching" / "fewshot_teaching_results.json"
FULLSHOT_JSON   = RESULTS_DIR / "cuad_hybrid_fullshot_results.json"
# Colour palette
TIER_COLORS = {
    "excellent": "#2ecc71",   # F1 >= 0.70
    "good":      "#f39c12",   # F1 0.45 – 0.70
    "poor":      "#e74c3c",   # F1 0.20 – 0.45
    "zero":      "#95a5a6",   # F1 == 0
}
ACCENT   = "#3498db"
ACCENT2  = "#e74c3c"
ACCENT3  = "#2ecc71"
BG_COLOR = "#f8f9fa"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#cccccc",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#555555",
    "ytick.color":      "#555555",
    "text.color":       "#333333",
    "grid.color":       "#dddddd",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "font.size":        10,
})
# Helpers
def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tier_color(f1: float) -> str:
    if f1 == 0.0:
        return TIER_COLORS["zero"]
    if f1 < 0.45:
        return TIER_COLORS["poor"]
    if f1 < 0.70:
        return TIER_COLORS["good"]
    return TIER_COLORS["excellent"]


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(BASE_DIR)}")
# Chart 1 — Per-class F1 horizontal bar (sorted)
def chart_per_class_f1(data: dict) -> None:
    pcm: Dict[str, dict] = data["per_class_metrics"]
    classes = list(pcm.keys())
    f1s     = [pcm[c]["f1"] for c in classes]
    supports= [pcm[c]["support"] for c in classes]

    # Sort by F1 ascending
    order = sorted(range(len(classes)), key=lambda i: f1s[i])
    classes_s = [classes[i] for i in order]
    f1s_s     = [f1s[i]     for i in order]
    sup_s     = [supports[i] for i in order]
    colors_s  = [tier_color(v) for v in f1s_s]

    fig, ax = plt.subplots(figsize=(12, 14))
    y = np.arange(len(classes_s))
    bars = ax.barh(y, f1s_s, color=colors_s, height=0.72, edgecolor="none")

    # Annotate F1 value + support
    for i, (bar, f, s) in enumerate(zip(bars, f1s_s, sup_s)):
        ax.text(f + 0.005, i, f"{f:.3f}  (n={s})",
                va="center", fontsize=8, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(classes_s, fontsize=8.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("Per-class F1 Score — Fine-tuned Hybrid Model (102 test contracts)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.axvline(0.50, color="#555", lw=1.2, linestyle=":", alpha=0.7)
    ax.axvline(0.70, color="#555", lw=1.2, linestyle=":", alpha=0.7)
    ax.text(0.503, -0.8, "0.50", color="#555", fontsize=8)
    ax.text(0.703, -0.8, "0.70", color="#555", fontsize=8)
    ax.grid(axis="x", alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=TIER_COLORS["excellent"], label="Excellent  F1 ≥ 0.70"),
        mpatches.Patch(color=TIER_COLORS["good"],      label="Good       F1 0.45–0.70"),
        mpatches.Patch(color=TIER_COLORS["poor"],      label="Poor       F1 0.20–0.45"),
        mpatches.Patch(color=TIER_COLORS["zero"],      label="Zero       F1 = 0"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "chart1_per_class_f1.png")
# Chart 2 — Precision vs Recall scatter with F1 iso-curves
def chart_precision_recall_scatter(data: dict) -> None:
    pcm = data["per_class_metrics"]
    classes   = list(pcm.keys())
    precisions= [pcm[c]["precision"] for c in classes]
    recalls   = [pcm[c]["recall"]    for c in classes]
    f1s       = [pcm[c]["f1"]        for c in classes]
    supports  = [pcm[c]["support"]   for c in classes]

    fig, ax = plt.subplots(figsize=(10, 9))

    # F1 iso-curves
    t = np.linspace(0.01, 1.0, 200)
    for f_target in [0.2, 0.4, 0.6, 0.8]:
    # F1 iso-curve formula
        with np.errstate(divide="ignore", invalid="ignore"):
            p_iso = f_target * t / (2 * t - f_target)
        mask = (p_iso >= 0) & (p_iso <= 1)
        ax.plot(t[mask], p_iso[mask], color="#aaaaaa", lw=0.9, linestyle="--", alpha=0.8)
        # Label the iso-curve near top-right
        valid = np.where(mask)[0]
        if len(valid):
            xi = t[valid[-1]]
            yi = p_iso[valid[-1]]
            if 0 <= yi <= 1:
                ax.text(xi + 0.01, yi, f"F1={f_target:.1f}",
                        fontsize=7.5, color="#888888", va="bottom")

    # Scatter — size proportional to sqrt(support)
    max_sup = max(supports) if supports else 1
    sizes   = [max(20, 400 * (s / max_sup) ** 0.5) for s in supports]
    scatter_colors = [tier_color(f) for f in f1s]

    sc = ax.scatter(recalls, precisions, s=sizes, c=scatter_colors,
                    alpha=0.82, edgecolors="#ffffff", linewidths=0.6, zorder=5)

    # Label top 10 and bottom 5 by support
    labeled = set()
    # Always label F1=0 classes and support > 40
    for i, (cls, p, r, f, s) in enumerate(zip(classes, precisions, recalls, f1s, supports)):
        if f == 0.0 or s >= 40:
            ax.annotate(
                cls.replace(" ", "\n") if len(cls) > 18 else cls,
                (r, p), fontsize=6.5, ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points",
                color="#444444",
            )
            labeled.add(i)

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Recall per Class\n(bubble size ∝ support count)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.4)

    legend_patches = [
        mpatches.Patch(color=TIER_COLORS["excellent"], label="Excellent  F1 ≥ 0.70"),
        mpatches.Patch(color=TIER_COLORS["good"],      label="Good       F1 0.45–0.70"),
        mpatches.Patch(color=TIER_COLORS["poor"],      label="Poor       F1 0.20–0.45"),
        mpatches.Patch(color=TIER_COLORS["zero"],      label="Zero       F1 = 0"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save(fig, "chart2_precision_recall_scatter.png")
# Chart 3 — Ablation study: neural weight vs metrics
def chart_ablation(data: dict) -> None:
    results = data["ablation_results"]
    neural_w  = [r["neural_weight"]   for r in results]
    macro_f1  = [r["macro_f1"]        for r in results]
    micro_f1  = [r["micro_f1"]        for r in results]
    macro_p   = [r["macro_precision"] for r in results]
    macro_r   = [r["macro_recall"]    for r in results]
    hamming   = [r["hamming_loss"]    for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hybrid Weight Ablation Study (102 test contracts)",
                 fontsize=13, fontweight="bold", y=1.01)

    # Left: F1 + P + R
    ax = axes[0]
    ax.plot(neural_w, macro_f1, "o-", color=ACCENT,  lw=2.2, ms=7, label="Macro F1")
    ax.plot(neural_w, micro_f1, "s-", color=ACCENT3, lw=2.2, ms=7, label="Micro F1")
    ax.plot(neural_w, macro_p,  "^--", color="#9b59b6", lw=1.6, ms=6, label="Macro Precision", alpha=0.8)
    ax.plot(neural_w, macro_r,  "v--", color="#e67e22", lw=1.6, ms=6, label="Macro Recall", alpha=0.8)

    # Highlight best Macro F1
    best_idx = int(np.argmax(macro_f1))
    ax.axvline(neural_w[best_idx], color="#aaaaaa", lw=1.2, linestyle=":")
    ax.scatter([neural_w[best_idx]], [macro_f1[best_idx]],
               color=ACCENT, s=160, zorder=10, marker="*", label=f"Best F1={macro_f1[best_idx]:.4f}")
    ax.set_xlabel("Neural Weight  →  (1 - Neural = Keyword Weight)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("F1 / Precision / Recall vs Neural Weight", fontsize=11)
    ax.set_xticks(neural_w)
    ax.set_xticklabels([f"{w:.2f}" for w in neural_w], rotation=30, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.5)

    # Right: Hamming Loss
    ax2 = axes[1]
    bars = ax2.bar([f"{w:.2f}" for w in neural_w], hamming,
                   color=[tier_color(1 - h) for h in hamming], edgecolor="none", width=0.5)
    for bar, v in zip(bars, hamming):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                 f"{v:.4f}", ha="center", fontsize=8.5)
    ax2.set_xlabel("Neural Weight", fontsize=10)
    ax2.set_ylabel("Hamming Loss  (lower is better)", fontsize=10)
    ax2.set_title("Hamming Loss vs Neural Weight", fontsize=11)
    ax2.set_ylim(0, max(hamming) * 1.25)
    ax2.grid(axis="y", alpha=0.5)

    fig.tight_layout()
    save(fig, "chart3_ablation_study.png")
# Chart 4 — Training convergence (loss + F1 per epoch)
def chart_training_convergence(data: dict) -> None:
    history = data["config"]["model_config"].get("training_history", [])
    if not history:
        print("  [skip] No training history in finetuned results")
        return

    epochs    = [h["epoch"]         for h in history]
    losses    = [h["train_loss"]     for h in history]
    macro_f1s = [h["macro_f1"]       for h in history]
    micro_f1s = [h["micro_f1"]       for h in history]
    macro_rs  = [h["macro_recall"]   for h in history]
    macro_ps  = [h["macro_precision"] for h in history]

    best_epoch = max(history, key=lambda h: h["macro_f1"])["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fine-tuning Convergence — Legal-BERT Multilabel",
                 fontsize=13, fontweight="bold")

    # Training loss
    ax1.plot(epochs, losses, "o-", color=ACCENT2, lw=2.2, ms=7, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("BCE Loss", fontsize=10)
    ax1.set_title("Training Loss per Epoch", fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(alpha=0.5)
    ax1.legend(fontsize=9)

    # Metrics
    ax2.plot(epochs, macro_f1s, "o-", color=ACCENT,  lw=2.2, ms=7, label="Macro F1")
    ax2.plot(epochs, micro_f1s, "s-", color=ACCENT3, lw=2.2, ms=7, label="Micro F1")
    ax2.plot(epochs, macro_ps,  "^--", color="#9b59b6", lw=1.6, ms=6, label="Macro Precision", alpha=0.8)
    ax2.plot(epochs, macro_rs,  "v--", color="#e67e22", lw=1.6, ms=6, label="Macro Recall", alpha=0.8)
    ax2.axvline(best_epoch, color="#aaaaaa", lw=1.2, linestyle=":")
    ax2.text(best_epoch + 0.05, min(macro_f1s) - 0.01, f"Best epoch={best_epoch}",
             fontsize=8, color="#555555")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Validation Metrics per Epoch", fontsize=11)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.3, 0.9)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.5)

    fig.tight_layout()
    save(fig, "chart4_training_convergence.png")
# Chart 5 — Per-contract recall distribution
def chart_per_contract_recall(data: dict) -> None:
    top_k = data.get("per_contract_top_k", [])
    if not top_k:
        print("  [skip] per_contract_top_k not found")
        return

    recalls = [c["recall"] for c in top_k if c.get("recall") is not None]
    if not recalls:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Per-contract Recall Distribution (102 test contracts)",
                 fontsize=13, fontweight="bold")

    # Histogram
    ax = axes[0]
    counts, bins, patches = ax.hist(recalls, bins=15, color=ACCENT, edgecolor="white", alpha=0.85)
    mean_r  = np.mean(recalls)
    median_r = np.median(recalls)
    ax.axvline(mean_r,   color=ACCENT2, lw=2, linestyle="--", label=f"Mean = {mean_r:.3f}")
    ax.axvline(median_r, color=ACCENT3, lw=2, linestyle=":",  label=f"Median = {median_r:.3f}")
    ax.set_xlabel("Recall (fraction of GT clause types found)", fontsize=10)
    ax.set_ylabel("Number of Contracts", fontsize=10)
    ax.set_title("Recall Distribution Histogram", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.5)

    # CDF
    ax2 = axes[1]
    sorted_r = np.sort(recalls)
    cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax2.plot(sorted_r, cdf, color=ACCENT, lw=2.2)
    ax2.fill_between(sorted_r, cdf, alpha=0.12, color=ACCENT)
    for thr in [0.5, 0.7, 0.9]:
        idx = np.searchsorted(sorted_r, thr)
        frac = cdf[min(idx, len(cdf) - 1)]
        ax2.plot([thr, thr], [0, frac], color="#999999", lw=1, linestyle=":")
        ax2.plot([0, thr], [frac, frac], color="#999999", lw=1, linestyle=":")
        ax2.text(thr + 0.01, frac - 0.04, f"{frac*100:.0f}% ≤ {thr}", fontsize=8, color="#555")
    ax2.set_xlabel("Recall threshold", fontsize=10)
    ax2.set_ylabel("Fraction of contracts", fontsize=10)
    ax2.set_title("Cumulative Distribution of Contract Recall", fontsize=11)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.5)

    fig.tight_layout()
    save(fig, "chart5_per_contract_recall.png")
# Chart 6 — Support distribution (class frequency)
def chart_support_distribution(data: dict) -> None:
    pcm      = data["per_class_metrics"]
    classes  = list(pcm.keys())
    supports = [pcm[c]["support"] for c in classes]

    order    = sorted(range(len(classes)), key=lambda i: supports[i], reverse=True)
    classes_s= [classes[i] for i in order]
    sup_s    = [supports[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 9))
    colors  = [ACCENT if s >= 20 else "#f39c12" if s >= 5 else "#e74c3c" for s in sup_s]
    y = np.arange(len(classes_s))
    bars = ax.barh(y, sup_s, color=colors, height=0.72, edgecolor="none")
    for bar, s in zip(bars, sup_s):
        ax.text(s + 0.5, bar.get_y() + bar.get_height() / 2,
                str(s), va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(classes_s, fontsize=8.5)
    ax.set_xlabel("Support (number of positive test contracts)", fontsize=11)
    ax.set_title("Class Frequency in Test Set (102 contracts)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color=ACCENT,     label="Common  (≥ 20)"),
        mpatches.Patch(color="#f39c12",  label="Moderate (5–19)"),
        mpatches.Patch(color="#e74c3c",  label="Rare     (< 5)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "chart6_support_distribution.png")
# Chart 7 — Few-shot learning curve
def chart_fewshot_curve(data: dict) -> None:
    shots_data = data["results_by_shot"]
    shots  = sorted(int(k) for k in shots_data.keys())
    f1s    = [shots_data[str(s)]["mean_macro_f1"]   for s in shots]
    stds   = [shots_data[str(s)]["std_macro_f1"]    for s in shots]
    precs  = [shots_data[str(s)]["mean_macro_prec"] for s in shots]
    recs   = [shots_data[str(s)]["mean_macro_rec"]  for s in shots]
    confs  = [shots_data[str(s)]["mean_avg_conf"]   for s in shots]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Few-shot Teaching Evaluation — 6 Rare Clause Types",
                 fontsize=13, fontweight="bold")

    # F1 with std band
    ax = axes[0]
    shots_arr = np.array(shots)
    f1_arr    = np.array(f1s)
    std_arr   = np.array(stds)
    ax.fill_between(shots_arr, f1_arr - std_arr, f1_arr + std_arr,
                    alpha=0.18, color=ACCENT, label="±1 std dev")
    ax.plot(shots_arr, f1_arr, "o-", color=ACCENT,  lw=2.2, ms=8,  label="Macro F1")
    ax.plot(shots_arr, precs,  "^--", color="#9b59b6", lw=1.8, ms=7, label="Macro Precision")
    ax.plot(shots_arr, recs,   "v--", color=ACCENT2, lw=1.8, ms=7, label="Macro Recall")
    ax.axhline(f1s[0], color="#888", lw=1.2, linestyle=":", label=f"0-shot baseline = {f1s[0]:.3f}")

    for s, f in zip(shots, f1s):
        ax.text(s, f + 0.015, f"{f:.3f}", ha="center", fontsize=8, color=ACCENT)

    ax.set_xlabel("Number of teaching examples (shots)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Macro F1 / P / R vs Shot Count", fontsize=11)
    ax.set_xticks(shots)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.5)

    # Confidence + Micro F1
    ax2 = axes[1]
    micro_f1s = [shots_data[str(s)]["mean_micro_f1"] for s in shots]
    color_conf  = "#9b59b6"
    color_micro = "#1abc9c"
    ax2.plot(shots, confs,     "D-", color=color_conf,  lw=2.0, ms=7, label="Avg Confidence")
    ax2.plot(shots, micro_f1s, "o-", color=color_micro, lw=2.0, ms=7, label="Micro F1")
    ax2.set_xlabel("Number of teaching examples (shots)", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Avg Confidence & Micro F1 vs Shot Count", fontsize=11)
    ax2.set_xticks(shots)
    ax2.set_ylim(0, 0.85)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.5)

    fig.tight_layout()
    save(fig, "chart7_fewshot_learning_curve.png")
# Chart 8 — Per-class few-shot heatmap
def chart_fewshot_heatmap(data: dict) -> None:
    shots_data   = data["results_by_shot"]
    target_types = data["config"]["target_types"]
    shots        = sorted(int(k) for k in shots_data.keys())

    # Build matrix [n_types x n_shots]
    short_labels = {
        "Most Favored Nation":                "Most Favored Nation",
        "Price Restrictions":                 "Price Restrictions",
        "Unlimited/All-You-Can-Eat-License":  "Unlimited License",
        "Source Code Escrow":                 "Source Code Escrow",
        "Non-Disparagement":                  "Non-Disparagement",
        "Notice Period To Terminate Renewal": "Notice Period (Renewal)",
    }
    matrix = np.zeros((len(target_types), len(shots)))
    for j, s in enumerate(shots):
        per_class = shots_data[str(s)].get("per_class_mean_f1", {})
        for i, t in enumerate(target_types):
            matrix[i, j] = per_class.get(t, 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean F1 Score", shrink=0.85)

    ax.set_xticks(range(len(shots)))
    ax.set_xticklabels([f"{s}-shot" for s in shots], fontsize=10)
    ax.set_yticks(range(len(target_types)))
    ax.set_yticklabels([short_labels.get(t, t) for t in target_types], fontsize=10)

    for i in range(len(target_types)):
        for j in range(len(shots)):
            val = matrix[i, j]
            color = "black" if 0.2 < val < 0.75 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_title("Per-class F1 by Shot Count — Few-shot Teaching (5 trials avg)",
                 fontsize=12, fontweight="bold", pad=12)
    # Data count annotations on right
    counts = data["data_counts"]
    for i, t in enumerate(target_types):
        tr = counts[t]["train"]
        te = counts[t]["test"]
        ax.text(len(shots) + 0.15, i, f"train={tr} test={te}",
                va="center", fontsize=8, color="#666")
    ax.set_xlim(-0.5, len(shots) - 0.5 + 1.5)

    fig.tight_layout()
    save(fig, "chart8_fewshot_heatmap.png")
# Chart 9 — Metrics dashboard: summary tiles
def chart_metrics_dashboard(finetuned: dict, ablation: dict, fewshot: dict) -> None:
    agg = finetuned["aggregate_metrics"]
    top_k = finetuned.get("top_k_accuracy", {}).get("aggregate", {})
    best_ablation = max(ablation["ablation_results"], key=lambda r: r["macro_f1"])
    shots_data = fewshot["results_by_shot"]
    best_shot  = max(shots_data.items(), key=lambda kv: kv[1]["mean_macro_f1"])

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("LexiCache Model — Evaluation Dashboard", fontsize=16,
                 fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    tile_data = [
        ("Macro F1\n(finetuned hybrid)",
         f"{agg['macro']['f1']:.4f}",
         "Higher is better",  TIER_COLORS["good"]),
        ("Micro F1\n(finetuned hybrid)",
         f"{agg['micro']['f1']:.4f}",
         "Higher is better",  TIER_COLORS["excellent"]),
        ("Macro Precision\n(finetuned hybrid)",
         f"{agg['macro']['precision']:.4f}",
         "Higher is better",  TIER_COLORS["good"]),
        ("Macro Recall\n(finetuned hybrid)",
         f"{agg['macro']['recall']:.4f}",
         "Higher is better",  TIER_COLORS["poor"]),
        ("Hamming Loss\n(finetuned hybrid)",
         f"{agg['hamming_loss']:.4f}",
         "Lower is better",  TIER_COLORS["good"]),
        ("Mean Contract Recall\n(top-K accuracy)",
         f"{top_k.get('mean', 0):.4f}",
         f"Median={top_k.get('median',0):.3f}",  TIER_COLORS["excellent"]),
        ("Best Macro F1\n(ablation sweep)",
         f"{best_ablation['macro_f1']:.4f}",
         f"Neural={best_ablation['neural_weight']:.2f} KW={best_ablation['kw_weight']:.2f}",
         TIER_COLORS["good"]),
        (f"Few-shot F1\n({best_shot[0]}-shot)",
         f"{best_shot[1]['mean_macro_f1']:.4f}",
         "Rare clause types only",  TIER_COLORS["excellent"]),
    ]

    for idx, (title, value, subtitle, color) in enumerate(tile_data):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(color + "22")  # light tint
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor=color,
            facecolor=color + "20",
            transform=ax.transAxes,
        ))
        ax.text(0.5, 0.68, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.35, title, ha="center", va="center",
                fontsize=9, color="#333333", transform=ax.transAxes)
        ax.text(0.5, 0.13, subtitle, ha="center", va="center",
                fontsize=7.5, color="#777777", transform=ax.transAxes)
        ax.axis("off")

    save(fig, "chart9_metrics_dashboard.png")
# Main
def main() -> None:
    print("=" * 60)
    print("LexiCache Evaluation Report Generator")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    finetuned = load_json(FINETUNED_JSON)
    ablation  = load_json(ABLATION_JSON)
    fewshot   = load_json(FEWSHOT_JSON)

    if finetuned is None:
        print("ERROR: cuad_hybrid_finetuned_results.json is required. Run evaluate_cuad_multilabel_finetuned.py first.")
        sys.exit(1)

    print("\n[1/9] Per-class F1 bar chart ...")
    chart_per_class_f1(finetuned)

    print("[2/9] Precision-Recall scatter ...")
    chart_precision_recall_scatter(finetuned)

    if ablation:
        print("[3/9] Ablation study ...")
        chart_ablation(ablation)
    else:
        print("[3/9] Ablation: skipped (no data)")

    print("[4/9] Training convergence ...")
    chart_training_convergence(finetuned)

    print("[5/9] Per-contract recall distribution ...")
    chart_per_contract_recall(finetuned)

    print("[6/9] Support distribution ...")
    chart_support_distribution(finetuned)

    if fewshot:
        print("[7/9] Few-shot learning curve ...")
        chart_fewshot_curve(fewshot)

        print("[8/9] Few-shot per-class heatmap ...")
        chart_fewshot_heatmap(fewshot)
    else:
        print("[7/9] Few-shot: skipped (no data)")
        print("[8/9] Few-shot heatmap: skipped (no data)")

    if ablation and fewshot:
        print("[9/9] Metrics dashboard ...")
        chart_metrics_dashboard(finetuned, ablation, fewshot)
    else:
        print("[9/9] Dashboard: skipped (missing data)")

    print(f"\nDone! All charts saved to:\n  {OUT_DIR}")

    # Print quick summary table
    pcm = finetuned["per_class_metrics"]
    f1_vals = sorted([(c, pcm[c]["f1"]) for c in pcm], key=lambda x: -x[1])
    print("\n--- Top 5 classes by F1 ---")
    for cls, f1 in f1_vals[:5]:
        print(f"  {cls:<40} F1={f1:.3f}")
    print("--- Bottom 5 classes by F1 ---")
    for cls, f1 in f1_vals[-5:]:
        print(f"  {cls:<40} F1={f1:.3f}")
    agg = finetuned["aggregate_metrics"]
    print(f"\nAggregate  Macro F1={agg['macro']['f1']:.4f}  "
          f"Micro F1={agg['micro']['f1']:.4f}  "
          f"Hamming Loss={agg['hamming_loss']:.4f}")


if __name__ == "__main__":
    main()
