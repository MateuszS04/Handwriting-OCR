from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

from src.utils.io import (
    image_exists_for_record,
    page_id_from_record,
    read_json_records,
    setup_logging,
    split_counts,
    write_jsonl,
)

log = logging.getLogger(__name__)


def load_training_records(
    manifest_path: Path,
    *,
    lines_dir: Path,
    min_confidence: float = 0.0,
    require_images: bool = True,
) -> list[dict]:
    """Load and validate records before splitting."""

    records: list[dict] = []
    for record in read_json_records(manifest_path):
        text = (record.get("text") or "").strip()
        if not text:
            continue
        confidence = record.get("confidence")
        if confidence is not None and float(confidence) < min_confidence:
            continue
        if require_images and not image_exists_for_record(record, lines_dir):
            log.warning("skip missing image: %s", record.get("file"))
            continue

        normalized = dict(record)
        normalized["text"] = text
        normalized["page_id"] = page_id_from_record(record)
        normalized.setdefault("weight", float(confidence) if confidence is not None else 1.0)
        records.append(normalized)

    return records


def build_splits(
    manifest_path: Path,
    lines_dir: Path,
    out_dir: Path,
    *,
    min_confidence: float = 0.0,
    by: str = "page",
    seed: int = 42,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    require_images: bool = True,
) -> dict[str, list[dict]]:
    """Split one training manifest into train/val/test JSONL files."""

    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_training_records(
        manifest_path,
        lines_dir=lines_dir,
        min_confidence=min_confidence,
        require_images=require_images,
    )
    log.info("loaded %d usable records from %s", len(records), manifest_path)

    rng = random.Random(seed)

    if by == "page":
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            groups[record["page_id"]].append(record)

        ids = sorted(groups.keys())
        rng.shuffle(ids)
        n_train, n_val, _ = split_counts(len(ids), train, val, test)
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train : n_train + n_val])

        buckets = {"train": [], "val": [], "test": []}
        for page_id, page_records in groups.items():
            if page_id in train_ids:
                buckets["train"].extend(page_records)
            elif page_id in val_ids:
                buckets["val"].extend(page_records)
            else:
                buckets["test"].extend(page_records)

    elif by == "line":
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_train, n_val, _ = split_counts(len(shuffled), train, val, test)
        buckets = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
    else:
        raise ValueError(f"Unsupported split grouping: {by}")

    for name, split_records in buckets.items():
        out_path = out_dir / f"{name}.jsonl"
        write_jsonl(split_records, out_path)
        pages = {r["page_id"] for r in split_records}
        log.info("wrote %s: %d records from %d pages", out_path, len(split_records), len(pages))

    return buckets


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--lines-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--by", choices=["page", "line"], default="page")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--allow-missing-images", action="store_true")
    args = ap.parse_args()

    build_splits(
        manifest_path=args.manifest,
        lines_dir=args.lines_dir,
        out_dir=args.out_dir,
        min_confidence=args.min_confidence,
        by=args.by,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
        require_images=not args.allow_missing_images,
    )


if __name__ == "__main__":
    main()