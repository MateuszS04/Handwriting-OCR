from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from PIL import Image

from src.utils.io import read_jsonl


log = logging.getLogger(__name__)


@dataclass
class LineDatasetConfig:
    lines_dir: Path
    max_target_length: int = 128
    is_train: bool = False


class LineDataset:

    def __init__(
        self,
        jsonl_path: Path,
        processor,
        cfg: LineDatasetConfig,
        transform=None,
    ) -> None:
        self.records = list(read_jsonl(jsonl_path))
        self.processor = processor
        self.cfg = cfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, record: dict) -> Path:
        if record.get("image_path"):
            path = Path(record["image_path"])
            if path.exists():
                return path
        return self.cfg.lines_dir / record["file"]

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        path = self._image_path(r)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        labels = self.processor.tokenizer(
            r["text"],
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_target_length,
        ).input_ids

        pad_id = self.processor.tokenizer.pad_token_id

        labels = [tok if tok != pad_id else -100 for tok in labels]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "weight": float(r.get("weight", 1.0)),
            "file": r["file"],
        }