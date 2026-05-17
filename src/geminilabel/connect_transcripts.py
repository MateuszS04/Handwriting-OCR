"""Connect line-crop metadata with transcript JSON into one training manifest.

This module does NOT split into train/val/test. Splitting is handled only by:

    python -m src.train.dataset_prep ...

Use this module only when you need to create one combined manifest, usually
``data/splits/all.jsonl``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.io import (
    LineRecord,
    TranscriptRecord,
    load_lines,
    load_transcripts,
    setup_logging,
    write_jsonl,
)

log = logging.getLogger(__name__)


def _transcript_keys(t: TranscriptRecord) -> list[str]:
    keys: list[str] = []
    if t.file:
        p = Path(t.file)
        keys.extend([t.file, p.name, p.stem])
    if t.page_id and t.line_no is not None:
        keys.append(f"{t.page_id}#{t.line_no:03d}")
        keys.append(f"{t.page_id}#{t.line_no}")
    return keys


def _line_keys(line: LineRecord) -> list[str]:
    return [
        line.rel_path,
        line.file,
        Path(line.file).stem,
        f"{line.page_id}#{line.line_no:03d}",
        f"{line.page_id}#{line.line_no}",
    ]


def build_training_manifest(
    *,
    lines_dir: Path,
    transcripts_path: Path,
    out_jsonl: Path,
    require_images: bool = True,
    min_confidence: float | None = None,
) -> list[dict]:
    """Join line crops with transcript JSON and write one JSONL manifest."""

    lines = load_lines(lines_dir, require_images=require_images)
    transcripts = load_transcripts(transcripts_path)

    transcript_index: dict[str, TranscriptRecord] = {}
    duplicate_keys: set[str] = set()
    for transcript in transcripts:
        if min_confidence is not None and transcript.confidence is not None:
            if transcript.confidence < min_confidence:
                continue
        for key in _transcript_keys(transcript):
            if key in transcript_index:
                duplicate_keys.add(key)
            else:
                transcript_index[key] = transcript

    if duplicate_keys:
        log.warning("duplicate transcript keys ignored after first match: %d", len(duplicate_keys))

    manifest: list[dict] = []
    unmatched_lines: list[str] = []
    for line in lines:
        transcript = None
        for key in _line_keys(line):
            transcript = transcript_index.get(key)
            if transcript:
                break

        if transcript is None:
            unmatched_lines.append(line.rel_path)
            continue

        manifest.append(
            {
                "file": line.rel_path,
                "image_path": line.image_path.as_posix(),
                "text": transcript.text,
                "page_id": line.page_id,
                "line_no": line.line_no,
                "confidence": transcript.confidence,
                "source_json": transcript.source_json,
                "polygon": line.polygon,
                "baseline": line.baseline,
                "pad_px": line.pad_px,
            }
        )

    write_jsonl(manifest, out_jsonl)

    log.info("loaded lines: %d", len(lines))
    log.info("loaded transcripts: %d", len(transcripts))
    log.info("matched training records: %d -> %s", len(manifest), out_jsonl)
    if unmatched_lines:
        log.warning("unmatched lines: %d (first: %s)", len(unmatched_lines), unmatched_lines[:5])

    return manifest


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines-dir", required=True, type=Path)
    ap.add_argument("--transcripts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="include records even when the image file is missing on disk",
    )
    args = ap.parse_args()

    build_training_manifest(
        lines_dir=args.lines_dir,
        transcripts_path=args.transcripts,
        out_jsonl=args.out,
        require_images=not args.allow_missing_images,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()

