"""Split CUAD JSON contracts into reproducible 80/20 train/test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def split_cuad(
    source_dir: Path,
    train_dir: Path,
    test_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    clean: bool = False,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for path in train_dir.glob("*.json"):
            path.unlink()
        for path in test_dir.glob("*.json"):
            path.unlink()

    files = sorted(source_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No .json files found in {source_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for src in train_files:
        shutil.copy2(src, train_dir / src.name)

    for src in test_files:
        shutil.copy2(src, test_dir / src.name)

    print("CUAD split complete")
    print(f"  Source files: {len(files)}")
    print(f"  Train files:  {len(train_files)} -> {train_dir}")
    print(f"  Test files:   {len(test_files)} -> {test_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split CUAD JSON files into train/test folders.")
    parser.add_argument("--source", type=Path, default=Path("data/processed/cuad/full"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/cuad/train"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/cuad/test"))
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Remove existing JSON files in target folders first")
    args = parser.parse_args()

    split_cuad(
        source_dir=args.source,
        train_dir=args.train,
        test_dir=args.test,
        train_ratio=args.ratio,
        seed=args.seed,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
