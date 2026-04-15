"""Fine-tune Legal-BERT for document-level multi-label CUAD clause classification."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# CUAD 41 canonical categories (must match evaluate_cuad_document_level.py)
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
# Label aliases: maps variant names → canonical CUAD 41 name

LABEL_ALIASES: Dict[str, str] = {
    # Termination
    "termination": "Termination For Convenience",
    "termination for convenience": "Termination For Convenience",
    "termination for cause": "Termination For Convenience",
    # Liability
    "limitation of liability": "Cap On Liability",
    "cap on liability": "Cap On Liability",
    "liability cap": "Cap On Liability",
    "uncapped liability": "Uncapped Liability",
    "unlimited liability": "Uncapped Liability",
    # IP
    "ip ownership assignment": "Ip Ownership Assignment",
    "intellectual property assignment": "Ip Ownership Assignment",
    "joint ip ownership": "Joint Ip Ownership",
    "joint intellectual property": "Joint Ip Ownership",
    # License
    "license grant": "License Grant",
    "non-transferable license": "Non-Transferable License",
    "irrevocable or perpetual license": "Irrevocable Or Perpetual License",
    "perpetual license": "Irrevocable Or Perpetual License",
    "irrevocable license": "Irrevocable Or Perpetual License",
    "unlimited license": "Unlimited/All-You-Can-Eat-License",
    "affiliate license-licensor": "Affiliate License-Licensor",
    "affiliate license-licensee": "Affiliate License-Licensee",
    # Solicitation
    "no-solicit of customers": "No-Solicit Of Customers",
    "no-solicit of employees": "No-Solicit Of Employees",
    "non-solicitation": "No-Solicit Of Employees",
    # Other
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
    """Map a raw clause type string to a canonical CUAD 41 category."""
    if not raw:
        return None
    # Direct match
    if raw in CAT_TO_IDX:
        return raw
    # Alias lookup (case-insensitive)
    lower = raw.strip().lower()
    if lower in LABEL_ALIASES:
        return LABEL_ALIASES[lower]
    # Case-insensitive direct match
    for cat in CUAD_41_CATEGORIES:
        if cat.lower() == lower:
            return cat
    return None
# Dataset
class CUADMultiLabelDataset(Dataset):
    """Document-level multi-label dataset for CUAD."""

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        max_chunks: int = 6,
        chunk_size: int = 512,
        verbose: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.samples: List[Tuple[str, List[float]]] = []

        files = sorted(data_dir.glob("*.json"))
        if verbose:
            print(f"  Loading {len(files)} contracts from {data_dir} ...")

        skipped = 0
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    item = json.load(f)
            except Exception:
                skipped += 1
                continue

            full_text: str = item.get("full_text", "")
            anns: list = item.get("clause_types", [])
            if not isinstance(full_text, str) or not full_text.strip():
                skipped += 1
                continue

            # Build binary label vector
            label_vec = [0.0] * NUM_LABELS
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

            self.samples.append((full_text, label_vec))

        if verbose:
            n_pos = sum(sum(lv) for _, lv in self.samples)
            print(f"  Loaded {len(self.samples)} contracts | "
                  f"skipped={skipped} | total positive labels={int(n_pos)}")
            # Per-class statistics
            counts = defaultdict(int)
            for _, lv in self.samples:
                for i, v in enumerate(lv):
                    if v > 0:
                        counts[CUAD_41_CATEGORIES[i]] += 1
            print(f"  Label distribution (top 10):")
            for cat, cnt in sorted(counts.items(), key=lambda x: -x[1])[:10]:
                print(f"    {cat}: {cnt}/{len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, labels = self.samples[idx]

        # Tokenize without special tokens, then split into chunks
        token_ids: List[int] = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        effective = self.chunk_size - 2  # reserve for [CLS] and [SEP]
        raw_chunks = [
            token_ids[i: i + effective]
            for i in range(0, max(1, len(token_ids)), effective)
        ]
        raw_chunks = raw_chunks[: self.max_chunks]

        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []

        cls_id = self.tokenizer.cls_token_id or 101
        sep_id = self.tokenizer.sep_token_id or 102
        pad_id = self.tokenizer.pad_token_id or 0

        for chunk in raw_chunks:
            ids = [cls_id] + chunk + [sep_id]
            mask = [1] * len(ids)
            # Pad to chunk_size
            pad_len = self.chunk_size - len(ids)
            ids += [pad_id] * pad_len
            mask += [0] * pad_len
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(mask, dtype=torch.long))

        # Pad to max_chunks with empty (all-pad) chunks
        while len(input_ids_list) < self.max_chunks:
            input_ids_list.append(torch.zeros(self.chunk_size, dtype=torch.long))
            attention_mask_list.append(torch.zeros(self.chunk_size, dtype=torch.long))

        return {
            "input_ids": torch.stack(input_ids_list),          # [max_chunks, chunk_size]
            "attention_mask": torch.stack(attention_mask_list), # [max_chunks, chunk_size]
            "labels": torch.tensor(labels, dtype=torch.float),  # [41]
            "n_chunks": torch.tensor(len(raw_chunks), dtype=torch.long),
        }
# Model
class LegalBERTMultiLabel(nn.Module):
    """Legal-BERT encoder with a 41-class sigmoid classification head."""

    def __init__(
        self,
        encoder_name: str = "nlpaueb/legal-bert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Enable gradient checkpointing to drastically reduce memory usage
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, max_chunks, chunk_size]
        attention_mask: torch.Tensor,  # [B, max_chunks, chunk_size]
        n_chunks: torch.Tensor,        # [B]  actual number of non-padding chunks
    ) -> torch.Tensor:                 # [B, num_labels]
        B, C, L = input_ids.shape
        # Flatten batch × chunks for parallel encoding
        ids_flat = input_ids.view(B * C, L)
        mask_flat = attention_mask.view(B * C, L)

        out = self.encoder(input_ids=ids_flat, attention_mask=mask_flat)
        cls_flat = out.last_hidden_state[:, 0, :]  # [B*C, hidden]
        cls_3d = cls_flat.view(B, C, -1)           # [B, C, hidden]

        # Mask out padding chunks (n_chunks tells us how many are real)
        chunk_mask = torch.zeros(B, C, device=input_ids.device)
        for b in range(B):
            chunk_mask[b, : n_chunks[b].item()] = 1.0
        chunk_mask = chunk_mask.unsqueeze(-1)       # [B, C, 1]

        pooled = (cls_3d * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)               # [B, hidden]
        return self.classifier(pooled)              # [B, num_labels]
# Training helpers
def compute_pos_weights(dataset: CUADMultiLabelDataset) -> torch.Tensor:
    """Compute per-class positive weights for BCEWithLogitsLoss."""
    N = len(dataset)
    counts = torch.zeros(NUM_LABELS)
    for _, lv in dataset.samples:
        counts += torch.tensor(lv)
    pos_weight = ((N - counts) / counts.clamp(min=1)).clamp(1.0, 20.0)
    return pos_weight


def train_one_epoch(
    model: LegalBERTMultiLabel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        n_chunks = batch["n_chunks"].to(device)

        logits = model(ids, mask, n_chunks)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def collect_probs(
    model: LegalBERTMultiLabel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect sigmoid probabilities and ground-truth labels over a DataLoader."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(loader, desc="  eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        n_chunks = batch["n_chunks"].to(device)
        labels = batch["labels"].numpy()

        logits = model(ids, mask, n_chunks)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    return np.vstack(all_probs), np.vstack(all_labels)


def evaluate_split(
    model: LegalBERTMultiLabel,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    per_class_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate on a DataLoader; return macro/micro F1."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    Y_prob, Y_true = collect_probs(model, loader, device)

    if per_class_thresholds is not None:
        Y_pred = (Y_prob >= per_class_thresholds[None, :]).astype(int)
    else:
        Y_pred = (Y_prob >= threshold).astype(int)

    macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro_p = precision_score(Y_true, Y_pred, average="macro", zero_division=0)
    macro_r = recall_score(Y_true, Y_pred, average="macro", zero_division=0)

    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
    }


def tune_per_class_thresholds(
    Y_prob: np.ndarray,
    Y_true: np.ndarray,
    candidates: Optional[List[float]] = None,
) -> np.ndarray:
    """Find the F1-optimal threshold for each class independently."""
    from sklearn.metrics import f1_score

    if candidates is None:
        candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                      0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    num_labels = Y_prob.shape[1]
    best_thresholds = np.full(num_labels, 0.50)

    for j in range(num_labels):
        best_f1 = -1.0
        for t in candidates:
            preds = (Y_prob[:, j] >= t).astype(int)
            f1 = f1_score(Y_true[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[j] = t

    return best_thresholds
# Main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Legal-BERT for CUAD document-level multi-label classification"
    )
    p.add_argument("--train-dir", default="data/processed/cuad/train",
                   help="Path to CUAD training JSON files")
    p.add_argument("--val-dir", default="data/processed/cuad/test",
                   help="Path to CUAD validation/test JSON files (used for early stopping)")
    p.add_argument("--output-dir", default="models/cuad_multilabel_finetuned",
                   help="Directory to save fine-tuned model")
    p.add_argument("--encoder-name", default="nlpaueb/legal-bert-base-uncased",
                   help="HuggingFace encoder model name")
    p.add_argument("--epochs", type=int, default=15,
                   help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Training batch size (reduce if OOM, default 2)")
    p.add_argument("--max-chunks", type=int, default=6,
                   help="Max document chunks (6 × 512 = 3072 tokens per doc)")
    p.add_argument("--chunk-size", type=int, default=512,
                   help="Tokens per chunk (including [CLS] and [SEP])")
    p.add_argument("--lr-encoder", type=float, default=2e-5,
                   help="Learning rate for Legal-BERT encoder layers")
    p.add_argument("--lr-head", type=float, default=1e-4,
                   help="Learning rate for classification head")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate on pooled representation")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binary prediction during eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-encoder-epochs", type=int, default=1,
                   help="Freeze encoder for first N epochs (train head only)")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a checkpoint .pth file to resume training from")
    p.add_argument("--lr-schedule", choices=["linear", "cosine"], default="cosine",
                   help="LR schedule after warmup: cosine decay (default) or linear decay")
    p.add_argument("--label-smoothing", type=float, default=0.05,
                   help="Label smoothing for BCE loss — targets become [s/2, 1-s/2]. "
                        "Reduces overconfidence on rare classes (default 0.05)")
    p.add_argument("--tune-thresholds", action="store_true", default=True,
                   help="After training, sweep per-class thresholds on val set to maximise F1")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("LexiCache — Fine-tune Legal-BERT for CUAD Multi-Label Classification")
    print(f"Device : {device}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  "
          f"Chunks: {args.max_chunks}×{args.chunk_size}")
    print("=" * 70)
    # Tokenizer & datasets
    print("\n[1/4] Loading tokenizer and datasets ...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    train_ds = CUADMultiLabelDataset(
        Path(args.train_dir), tokenizer,
        max_chunks=args.max_chunks, chunk_size=args.chunk_size,
    )
    val_ds = CUADMultiLabelDataset(
        Path(args.val_dir), tokenizer,
        max_chunks=args.max_chunks, chunk_size=args.chunk_size,
    )

    if len(train_ds) == 0:
        print(f"ERROR: No training samples found in {args.train_dir}")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    # Model
    print("\n[2/4] Building model ...")
    model = LegalBERTMultiLabel(
        encoder_name=args.encoder_name,
        num_labels=NUM_LABELS,
        dropout=args.dropout,
    ).to(device)

    # Positive class weights for imbalanced labels
    pos_weight = compute_pos_weights(train_ds).to(device)

    # Label smoothing: convert hard {0,1} targets to {s/2, 1-s/2} to reduce
    label_smoothing = args.label_smoothing

    class SmoothedBCELoss(nn.Module):
        def __init__(self, pos_weight: torch.Tensor, smoothing: float) -> None:
            super().__init__()
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.smoothing = smoothing

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if self.smoothing > 0:
                targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
            return self.loss_fn(logits, targets)

    criterion = SmoothedBCELoss(pos_weight, label_smoothing)

    start_epoch = 1
    if args.resume_from:
        print(f"\nRestoring model weights from checkpoint: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        
        # Try to infer starting epoch from the filename (e.g., epoch_01)
        import re
        match = re.search(r"epoch_(\d+)", str(args.resume_from))
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"Resuming training from Epoch {start_epoch}")
    # Optimizer & scheduler
    print("\n[3/4] Setting up optimizer ...")
    # Separate learning rates: encoder (small) vs. head (larger)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.classifier.parameters(), "lr": args.lr_head},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Training loop
    print("\n[4/4] Training ...")
    best_macro_f1 = 0.0
    best_epoch = 0
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Optionally freeze encoder for first N epochs
        if epoch <= args.freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = False
            print("  [encoder FROZEN — training head only]")
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
            if epoch == args.freeze_encoder_epochs + 1:
                print("  [encoder UNFROZEN — full fine-tuning]")

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print(f"  Train loss: {avg_loss:.4f}")

        # Validation
        metrics = evaluate_split(model, val_loader, device, threshold=args.threshold)
        print(
            f"  Val  Macro F1={metrics['macro_f1']:.4f}  "
            f"Micro F1={metrics['micro_f1']:.4f}  "
            f"P={metrics['macro_precision']:.4f}  "
            f"R={metrics['macro_recall']:.4f}"
        )

        history.append({"epoch": epoch, "train_loss": avg_loss, **metrics})

        # Save checkpoint
        ckpt_path = output_dir / f"epoch_{epoch:02d}_f1_{metrics['macro_f1']:.4f}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # Track best model
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best model saved (Macro F1={best_macro_f1:.4f})")

    # Save final model and config
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # Per-class threshold tuning on validation set using best model weights
    per_class_thresholds_list: Optional[List[float]] = None
    if args.tune_thresholds:
        print("\n[Threshold Tuning] Loading best model and sweeping per-class thresholds on val set ...")
        best_state = torch.load(output_dir / "best_model.pth", map_location=device)
        model.load_state_dict(best_state)
        Y_prob_val, Y_true_val = collect_probs(model, val_loader, device)
        per_class_thresholds = tune_per_class_thresholds(Y_prob_val, Y_true_val)
        per_class_thresholds_list = per_class_thresholds.tolist()

        # Evaluate with tuned thresholds vs global threshold
        from sklearn.metrics import f1_score
        Y_pred_global = (Y_prob_val >= args.threshold).astype(int)
        Y_pred_tuned = (Y_prob_val >= per_class_thresholds[None, :]).astype(int)
        f1_global = f1_score(Y_true_val, Y_pred_global, average="macro", zero_division=0)
        f1_tuned = f1_score(Y_true_val, Y_pred_tuned, average="macro", zero_division=0)
        print(f"  Global threshold ({args.threshold:.2f}) Macro F1 = {f1_global:.4f}")
        print(f"  Per-class tuned thresholds  Macro F1 = {f1_tuned:.4f}  (+{f1_tuned - f1_global:+.4f})")

        per_class_info = {
            cat: round(float(t), 3)
            for cat, t in zip(CUAD_41_CATEGORIES, per_class_thresholds)
        }
        print("  Per-class optimal thresholds:")
        for cat, t in per_class_info.items():
            print(f"    {cat}: {t}")

        np.save(str(output_dir / "per_class_thresholds.npy"), per_class_thresholds)
        with open(output_dir / "per_class_thresholds.json", "w") as f:
            json.dump(per_class_info, f, indent=2)
        print(f"  Saved: {output_dir / 'per_class_thresholds.json'}")

    config = {
        "encoder_name": args.encoder_name,
        "num_labels": NUM_LABELS,
        "cuad_41_categories": CUAD_41_CATEGORIES,
        "max_chunks": args.max_chunks,
        "chunk_size": args.chunk_size,
        "dropout": args.dropout,
        "threshold": args.threshold,
        "lr_schedule": args.lr_schedule,
        "label_smoothing": args.label_smoothing,
        "per_class_thresholds": per_class_thresholds_list,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "training_history": history,
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save tokenizer alongside model
    tokenizer.save_pretrained(str(output_dir))

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print(f"Best epoch : {best_epoch}  |  Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved: {output_dir / 'best_model.pth'}")
    print(f"Config     : {output_dir / 'config.json'}")
    print("=" * 70)
    print("\nNext step — run hybrid evaluation:")
    print("  python scripts/evaluation/evaluate_cuad_multilabel_finetuned.py")


if __name__ == "__main__":
    main()
