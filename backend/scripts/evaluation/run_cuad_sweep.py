"""Run a quick CUAD tuning sweep and rank configurations by post-teaching macro F1."""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple


def parse_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Macro Precision:") and "baseline_precision" not in metrics:
            metrics["baseline_precision"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Macro Recall:") and "baseline_recall" not in metrics:
            metrics["baseline_recall"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Macro F1:") and "baseline_f1" not in metrics:
            metrics["baseline_f1"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Macro Precision:") and "post_precision" not in metrics:
            metrics["post_precision"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Macro Recall:") and "post_recall" not in metrics:
            metrics["post_recall"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Macro F1:") and "post_f1" not in metrics:
            metrics["post_f1"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Delta Precision:"):
            metrics["delta_precision"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Delta Recall:"):
            metrics["delta_recall"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Delta F1:"):
            metrics["delta_f1"] = float(line.split(":", 1)[1].strip())
    return metrics


def run_one(
    py_exec: str,
    cfg: Dict[str, float | int],
    test_dir: Path,
    max_files: int,
    teach_k: int,
    result_jsonl: Path,
    teach_strategy: str,
    track_best_during_teaching: bool,
) -> Tuple[int, str]:
    cmd = [
        py_exec,
        "-m",
        "scripts.evaluation.evaluate_cuad_test",
        "--test-dir",
        str(test_dir),
        "--max-files",
        str(max_files),
        "--teach-k",
        str(teach_k),
        "--teach-strategy",
        str(teach_strategy),
        "--min-conf",
        str(cfg["min_conf"]),
        "--max-seed-per-type",
        str(cfg["seed_cap"]),
        "--kw-weight",
        str(cfg["kw"]),
        "--model-weight",
        str(cfg["model"]),
        "--inclusion-threshold",
        str(cfg["include"]),
        "--context-promote-threshold",
        str(cfg["promote"]),
        "--distance-scale",
        str(cfg["dist"]),
        "--agreement-bonus",
        str(cfg["agree"]),
        "--result-jsonl",
        str(result_jsonl),
    ]

    if track_best_during_teaching:
        cmd.append("--track-best-during-teaching")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick sweep runner for CUAD evaluator")
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/cuad/test"))
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--teach-k", type=int, default=5)
    parser.add_argument("--result-jsonl", type=Path, default=Path("experiments/results/cuad_eval_results.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--teach-strategy", type=str, default="missed", choices=["missed", "first"])
    parser.add_argument("--track-best-during-teaching", action="store_true")
    args = parser.parse_args()

    py_exec = str(Path(".venv") / "Scripts" / "python.exe")

    min_conf_values = [0.60, 0.65]
    include_values = [0.60, 0.65, 0.70]
    seed_caps = [3, 8, 12]
    kw_model_pairs = [(0.70, 0.30), (0.60, 0.40), (0.80, 0.20)]

    fixed = {
        "promote": 0.70,
        "dist": 2.4,
        "agree": 0.15,
    }

    configs: List[Dict[str, float | int]] = []
    for min_conf, include, seed_cap, (kw, model) in product(min_conf_values, include_values, seed_caps, kw_model_pairs):
        configs.append(
            {
                "min_conf": min_conf,
                "include": include,
                "seed_cap": seed_cap,
                "kw": kw,
                "model": model,
                "promote": fixed["promote"],
                "dist": fixed["dist"],
                "agree": fixed["agree"],
            }
        )

    print(f"Running {len(configs)} sweep configs...")
    ranked: List[Dict] = []

    for idx, cfg in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] cfg={cfg}")
        code, output = run_one(
            py_exec=py_exec,
            cfg=cfg,
            test_dir=args.test_dir,
            max_files=args.max_files,
            teach_k=args.teach_k,
            result_jsonl=args.result_jsonl,
            teach_strategy=args.teach_strategy,
            track_best_during_teaching=args.track_best_during_teaching,
        )

        if code != 0:
            print(f"  FAILED (exit={code})")
            continue

        metrics = parse_metrics(output)
        row = {**cfg, **metrics}
        ranked.append(row)
        print(
            "  baseline_f1="
            + f"{row.get('baseline_f1', float('nan')):.4f}"
            + ", post_f1="
            + f"{row.get('post_f1', float('nan')):.4f}"
            + ", delta_f1="
            + f"{row.get('delta_f1', float('nan')):+.4f}"
        )

    ranked.sort(key=lambda r: (r.get("post_f1", -1.0), r.get("delta_f1", -1.0)), reverse=True)

    out_path = Path("experiments") / "results" / "cuad_sweep_ranked.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)

    print("\nTop configs:")
    for i, row in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{i}. post_f1={row.get('post_f1', float('nan')):.4f}, "
            f"delta_f1={row.get('delta_f1', float('nan')):+.4f}, "
            f"min_conf={row['min_conf']}, include={row['include']}, seed_cap={row['seed_cap']}, "
            f"kw/model={row['kw']}/{row['model']}"
        )

    print(f"\nRanked results saved to {out_path}")


if __name__ == "__main__":
    main()
