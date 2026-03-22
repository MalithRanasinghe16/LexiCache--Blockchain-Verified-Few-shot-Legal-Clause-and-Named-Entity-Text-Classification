"""Prepare per-contract CUAD JSON files expected by split/evaluation scripts."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List


CLAUSE_RE = re.compile(r'related to\s+"([^"]+)"', re.IGNORECASE)


def safe_name(name: str, idx: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    if not cleaned:
        cleaned = f"contract_{idx:04d}"
    return f"{cleaned}.json"


def parse_clause_type(question: str, fallback: str) -> str:
    match = CLAUSE_RE.search(question or "")
    if match:
        return match.group(1).strip()
    return fallback


def prepare_cuad_full(source_zip: Path, out_dir: Path) -> None:
    if not source_zip.exists():
        raise FileNotFoundError(f"Source zip not found: {source_zip}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.json"):
        p.unlink()

    with zipfile.ZipFile(source_zip) as zf:
        payload = json.loads(zf.read("CUADv1.json"))

    entries: List[Dict] = payload.get("data", [])
    total_annotations = 0

    for idx, item in enumerate(entries):
        title = item.get("title", f"contract_{idx:04d}")
        paragraphs = item.get("paragraphs", [])
        if not paragraphs:
            continue

        para = paragraphs[0]
        full_text = para.get("context", "")
        qas = para.get("qas", [])

        clause_types = []
        for qa in qas:
            question = qa.get("question", "")
            qa_id = str(qa.get("id", ""))
            fallback = qa_id.split("__")[-1] if "__" in qa_id else qa_id or "Unknown"
            clause_type = parse_clause_type(question, fallback)

            for ans in qa.get("answers", []):
                text = ans.get("text", "")
                start = ans.get("answer_start")
                if not isinstance(start, int) or not isinstance(text, str):
                    continue
                end = start + len(text)
                if start < 0 or end > len(full_text) or end <= start:
                    continue

                clause_types.append(
                    {
                        "clause_type": clause_type,
                        "start": start,
                        "end": end,
                    }
                )

        total_annotations += len(clause_types)
        out = {
            "file_name": title,
            "full_text": full_text,
            "clause_types": clause_types,
        }

        file_path = out_dir / safe_name(title, idx)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print("CUAD full preparation complete")
    print(f"  Contracts written: {len(list(out_dir.glob('*.json')))}")
    print(f"  Total clause spans: {total_annotations}")
    print(f"  Output directory: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CUADv1.json archive into per-contract JSON files.")
    parser.add_argument("--source-zip", type=Path, default=Path("data/raw/cuad/data_from_repo.zip"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/cuad/full"))
    args = parser.parse_args()

    prepare_cuad_full(args.source_zip, args.out_dir)


if __name__ == "__main__":
    main()
