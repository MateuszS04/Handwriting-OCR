import argparse
from pathlib import Path

from src.train.dataset_prep import build_splits


def _cli_build_splits(args: argparse.Namespace) -> None:
    build_splits(
        manifest_path=Path(args.manifest),
        lines_dir=Path(args.lines_dir),
        out_dir=Path(args.out_dir),
        min_confidence=args.min_confidence,
        by=args.by,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
        require_images=not args.allow_missing_images,
    )