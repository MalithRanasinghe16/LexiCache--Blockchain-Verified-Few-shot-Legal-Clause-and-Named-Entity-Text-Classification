"""
LexiCache Hybrid Ablation Study
Compares:
- Pure Keyword-only (neural_weight=0.0)
- Pure Neural-only (neural_weight=1.0)
- Hybrid with different KW weights (0.3, 0.4, 0.5, 0.6, 0.7)

Perfect for Table 8.2 in Chapter 8
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import necessary functions from your original evaluation file
from scripts.evaluation.evaluate_cuad_multilabel_finetuned import (
    CUAD_41_CATEGORIES,
    CAT_TO_IDX,
    compute_keyword_scores,
    map_clause_type,
    extract_gt_labels,
    LegalBERTMultiLabel,
    load_finetuned_model,
    encode_document,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="LexiCache Hybrid Ablation Study")
    parser.add_argument("--test-dir", default="data/processed/cuad/test")
    parser.add_argument("--model-dir", default="models/cuad_multilabel_finetuned")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)

    print("=" * 85)
    print("LexiCache — Hybrid Ablation Study")
    print(f"Test directory : {test_dir}")
    print(f"Device         : {device}")
    print("=" * 85)

    # Load test files
    test_files = sorted(test_dir.glob("*.json"))
    if args.max_files > 0:
        test_files = test_files[:args.max_files]
    print(f"Loaded {len(test_files)} test contracts\n")

    # Load fine-tuned model
    model = None
    tokenizer = None
    if model_dir.exists():
        print(f"Loading fine-tuned Legal-BERT model from {model_dir} ...")
        model, tokenizer, _ = load_finetuned_model(model_dir, device)
    else:
        print("Warning: Model directory not found. Neural-only runs will be skipped.")

    # Pre-compute scores to save massive amounts of time
    print("Pre-computing Neural & Keyword scores for all documents...")
    all_gt_vecs = []
    all_kw_scores = []
    all_neural_scores = []
    
    valid_files = 0
    for path in tqdm(test_files, desc="Encoding Documents"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue

        full_text: str = item.get("full_text", "")
        if not full_text.strip():
            continue

        gt_vec = extract_gt_labels(item)
        kw_scores = compute_keyword_scores(full_text)
        
        if model is not None and tokenizer is not None:
            with torch.no_grad():
                ids_t, mask_t, n_t = encode_document(full_text, tokenizer, 6, 512, device)
                logits = model(ids_t, mask_t, n_t)
                neural_scores = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        else:
            neural_scores = np.zeros(len(CUAD_41_CATEGORIES), dtype=np.float32)
            
        all_gt_vecs.append(gt_vec)
        all_kw_scores.append(kw_scores)
        all_neural_scores.append(neural_scores)
        valid_files += 1

    # Configurations to test
    weights_to_test = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    results = []
    
    print(f"\nRunning ablation study on {valid_files} valid files...\n")

    for nw in weights_to_test:
        kw_weight = 1.0 - nw
        print(f"Testing → Neural weight = {nw:.2f} | KW weight = {kw_weight:.2f}")
        
        Y_pred = []
        for i in range(valid_files):
            n_scores = all_neural_scores[i]
            k_scores = all_kw_scores[i]
            
            if nw == 1.0 and kw_weight == 0.0:
                hybrid_scores = n_scores
            elif nw == 0.0 and kw_weight == 1.0:
                hybrid_scores = k_scores
            else:
                hybrid_scores = np.maximum(n_scores * nw, k_scores * kw_weight)
                hybrid_scores = hybrid_scores / max(nw, kw_weight)
                
            pred_vec = (hybrid_scores >= args.threshold).astype(np.float32)
            Y_pred.append(pred_vec)
            
        Y_true_arr = np.vstack(all_gt_vecs)
        Y_pred_arr = np.vstack(Y_pred)

        macro_f1 = f1_score(Y_true_arr, Y_pred_arr, average="macro", zero_division=0)
        macro_p = precision_score(Y_true_arr, Y_pred_arr, average="macro", zero_division=0)
        macro_r = recall_score(Y_true_arr, Y_pred_arr, average="macro", zero_division=0)
        micro_f1 = f1_score(Y_true_arr, Y_pred_arr, average="micro", zero_division=0)
        h_loss = hamming_loss(Y_true_arr, Y_pred_arr)

        res_dict = {
            "configuration": f"Neural={nw:.2f} / KW={kw_weight:.2f}",
            "neural_weight": nw,
            "kw_weight": kw_weight,
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "hamming_loss": round(h_loss, 4),
        }
        results.append(res_dict)
        print(f"   Macro F1 = {res_dict['macro_f1']:.4f} | Macro Prec = {res_dict['macro_precision']:.4f} | Macro Rec = {res_dict['macro_recall']:.4f}\n")

    # Save results
    output_path = Path("experiments/results/cuad_hybrid_ablation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": "Hybrid ablation study: Pure KW vs Pure Neural vs multiple hybrid weights",
        "test_contracts": valid_files,
        "threshold": args.threshold,
        "ablation_results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print clean table for Chapter 8
    print("\n" + "="*95)
    print("HYBRID ABLATION RESULTS — READY FOR TABLE 8.2 IN CHAPTER 8")
    print("="*95)
    print(f"{'Configuration':<32} {'Macro F1':>10} {'Macro P':>10} {'Macro R':>10} {'Micro F1':>10}")
    print("-" * 95)
    for r in results:
        print(f"{r['configuration']:<32} {r['macro_f1']:>10.4f} {r['macro_precision']:>10.4f} "
              f"{r['macro_recall']:>10.4f} {r['micro_f1']:>10.4f}")

if __name__ == "__main__":
    main()