"""Multi-task episodic fine-tuning on CUAD, LEDGAR, and CoNLL-2003.

This trainer uses real labels (no synthetic/random targets) and samples
few-shot episodes from each dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from src.data import normalize_text
from src.modeling import PrototypicalNetwork


Example = Tuple[str, str]


def load_cuad_examples(train_dir: Path, min_span_len: int = 30) -> List[Example]:
    examples: List[Example] = []
    files = sorted(train_dir.glob("*.json"))
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue

        full_text = item.get("full_text", "")
        anns = item.get("clause_types", [])
        if not isinstance(full_text, str) or not isinstance(anns, list):
            continue

        for ann in anns:
            if not isinstance(ann, dict):
                continue
            label = ann.get("clause_type")
            start = ann.get("start")
            end = ann.get("end")
            if not isinstance(label, str) or not label.strip():
                continue
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start < 0 or end <= start or end > len(full_text):
                continue

            span = full_text[start:end].strip()
            if len(span) < min_span_len:
                continue
            examples.append((normalize_text(span), f"CUAD::{label.strip()}"))

    return examples


def load_ledgar_examples(max_rows: int = 12000) -> List[Example]:
    ds = load_dataset("lex_glue", "ledgar", split=f"train[:{max_rows}]")
    label_names = ds.features["label"].names
    examples: List[Example] = []
    for row in ds:
        text = row.get("text", "")
        label_idx = row.get("label")
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        if not isinstance(label_idx, int) or not (0 <= label_idx < len(label_names)):
            continue
        label = label_names[label_idx]
        examples.append((normalize_text(text), f"LEDGAR::{label}"))
    return examples


def _dominant_non_o_tag(tag_ids: Sequence[int], tag_names: Sequence[str]) -> Optional[str]:
    names = [tag_names[t] for t in tag_ids if isinstance(t, int) and 0 <= t < len(tag_names)]
    names = [n for n in names if n != "O"]
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


def load_conll_examples(max_rows: int = 8000) -> List[Example]:
    ds = load_dataset("conll2003", split=f"train[:{max_rows}]")
    tag_names = ds.features["ner_tags"].feature.names

    examples: List[Example] = []
    for row in ds:
        tokens = row.get("tokens", [])
        ner_tags = row.get("ner_tags", [])
        if not isinstance(tokens, list) or not isinstance(ner_tags, list):
            continue
        text = " ".join(str(t) for t in tokens).strip()
        if len(text) < 20:
            continue

        label = _dominant_non_o_tag(ner_tags, tag_names)
        if not label:
            continue

        examples.append((normalize_text(text), f"CONLL::{label}"))

    return examples


def build_label_index(examples: Sequence[Example]) -> DefaultDict[str, List[int]]:
    idx: DefaultDict[str, List[int]] = defaultdict(list)
    for i, (_, label) in enumerate(examples):
        idx[label].append(i)
    return idx


def sample_episode(
    examples: Sequence[Example],
    label_to_indices: Dict[str, List[int]],
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: random.Random,
) -> Optional[Tuple[List[str], torch.Tensor, List[str], torch.Tensor]]:
    min_count = k_shot + q_query
    valid_labels = [label for label, ids in label_to_indices.items() if len(ids) >= min_count]
    if len(valid_labels) < n_way:
        return None

    chosen = rng.sample(valid_labels, n_way)

    support_texts: List[str] = []
    query_texts: List[str] = []
    support_local_labels: List[int] = []
    query_local_labels: List[int] = []

    for local_id, label in enumerate(chosen):
        ids = list(label_to_indices[label])
        rng.shuffle(ids)
        s_ids = ids[:k_shot]
        q_ids = ids[k_shot:k_shot + q_query]

        for i in s_ids:
            support_texts.append(examples[i][0])
            support_local_labels.append(local_id)

        for i in q_ids:
            query_texts.append(examples[i][0])
            query_local_labels.append(local_id)

    return (
        support_texts,
        torch.tensor(support_local_labels, dtype=torch.long),
        query_texts,
        torch.tensor(query_local_labels, dtype=torch.long),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-task episodic finetuning for LexiCache")
    parser.add_argument("--cuad-train-dir", type=Path, default=Path("data/processed/cuad/train"))
    parser.add_argument("--projection-in", type=Path, default=Path("models/projection_head.pth"))
    parser.add_argument("--projection-out", type=Path, default=Path("models/final_projection_head.pth"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episodes-per-epoch", type=int, default=150)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-proj", type=float, default=8e-5)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ledgar-max-rows", type=int, default=12000)
    parser.add_argument("--conll-max-rows", type=int, default=8000)
    parser.add_argument("--task-weights", type=float, nargs=3, default=[0.5, 0.3, 0.2], metavar=("CUAD", "LEDGAR", "CONLL"))
    parser.add_argument("--unfreeze-encoder", action="store_true", help="Enable full encoder finetuning (slow on CPU)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("LexiCache - Multi-Task Fine-Tuning (Labeled Episodic)")
    print("Datasets: CUAD + LEDGAR + CoNLL-2003")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data once (important for CPU performance)
    print("Loading datasets...")
    cuad_examples = load_cuad_examples(args.cuad_train_dir)
    ledgar_examples = load_ledgar_examples(args.ledgar_max_rows)
    conll_examples = load_conll_examples(args.conll_max_rows)

    print(f"  CUAD examples:   {len(cuad_examples)}")
    print(f"  LEDGAR examples: {len(ledgar_examples)}")
    print(f"  CoNLL examples:  {len(conll_examples)}")

    task_examples: Dict[str, List[Example]] = {
        "cuad": cuad_examples,
        "ledgar": ledgar_examples,
        "conll": conll_examples,
    }
    task_indices: Dict[str, Dict[str, List[int]]] = {
        task: build_label_index(examples)
        for task, examples in task_examples.items()
    }

    model = PrototypicalNetwork()
    projection = nn.Linear(model.hidden_size, model.hidden_size).to(device)

    if args.projection_in.exists():
        projection.load_state_dict(torch.load(args.projection_in, map_location=device))
        print(f"Loaded initial projection: {args.projection_in}")
    else:
        print(f"Projection input not found, training from fresh projection: {args.projection_in}")

    if args.unfreeze_encoder:
        model.train()
        for p in model.encoder.parameters():
            p.requires_grad = True
        print("Encoder finetuning: ENABLED")
        optim_params = [
            {"params": projection.parameters(), "lr": args.lr_proj},
            {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        ]
    else:
        model.eval()
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder finetuning: DISABLED (projection-only)")
        optim_params = [{"params": projection.parameters(), "lr": args.lr_proj}]

    projection.train()
    optimizer = torch.optim.Adam(optim_params)
    criterion = nn.CrossEntropyLoss()

    tasks = ["cuad", "ledgar", "conll"]
    weights = args.task_weights
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    rng = random.Random(args.seed)
    best_loss = float("inf")

    print(
        f"Starting training: epochs={args.epochs}, episodes/epoch={args.episodes_per_epoch}, "
        f"n_way={args.n_way}, k_shot={args.k_shot}, q_query={args.q_query}, batch_size={args.batch_size}"
    )

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        trained_eps = 0
        print(f"\nEpoch {epoch}/{args.epochs}")

        for _ in tqdm(range(args.episodes_per_epoch), desc=f"Epoch {epoch}"):
            task = random.choices(tasks, weights=probs, k=1)[0]
            episode = sample_episode(
                examples=task_examples[task],
                label_to_indices=task_indices[task],
                n_way=args.n_way,
                k_shot=args.k_shot,
                q_query=args.q_query,
                rng=rng,
            )
            if episode is None:
                continue

            support_texts, support_labels_local, query_texts, query_labels_local = episode
            support_labels_local = support_labels_local.to(device)
            query_labels_local = query_labels_local.to(device)

            support_emb = model(
                support_texts,
                batch_size=args.batch_size,
                enable_grad=args.unfreeze_encoder,
            )
            query_emb = model(
                query_texts,
                batch_size=args.batch_size,
                enable_grad=args.unfreeze_encoder,
            )

            support_proj = projection(support_emb.to(device))
            query_proj = projection(query_emb.to(device))

            prototypes, _ = model.compute_prototypes(support_proj, support_labels_local)
            dists = torch.cdist(query_proj, prototypes)
            loss = criterion(-dists, query_labels_local)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            trained_eps += 1

        if trained_eps == 0:
            print("  No trainable episodes this epoch. Check dataset sizes and n_way/k_shot/q_query.")
            continue

        avg_loss = total_loss / trained_eps
        print(f"  Epoch {epoch} completed | trained_episodes={trained_eps} | avg_loss={avg_loss:.4f}")

        args.projection_out.parent.mkdir(parents=True, exist_ok=True)
        epoch_ckpt = args.projection_out.parent / f"epoch_{epoch}_projection_head.pth"
        torch.save(projection.state_dict(), epoch_ckpt)
        print(f"  Saved epoch checkpoint: {epoch_ckpt}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(projection.state_dict(), args.projection_out)
            print(f"  New best projection saved: {args.projection_out}")

    print("\n" + "=" * 80)
    print("MULTI-TASK FINE-TUNING COMPLETED")
    print(f"Best avg loss: {best_loss:.4f}")
    print(f"Best model path: {args.projection_out}")
    print("=" * 80)


if __name__ == "__main__":
    main()