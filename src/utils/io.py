"""Small I/O helpers shared across the pipeline."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

TEXT_FIELDS = ("text", "transcription", "transcript", "ground_truth", "label")
FILE_FIELDS = ("file", "rel_path", "image", "image_path", "filename", "path")
LIST_FIELDS = ("lines", "items", "data", "transcripts", "records")
CONF_FIELDS = ("confidence", "score", "probability")


@dataclass(frozen=True)
class LineRecord:
    """One segmented crop from a line sidecar JSON."""

    page_id: str
    line_no: int
    file: str
    rel_path: str
    image_path: Path
    polygon: list | None
    baseline: list | None
    pad_px: int | None


@dataclass(frozen=True)
class TranscriptRecord:
    """One normalized transcript entry loaded from local/Gemini JSON."""

    text: str
    file: str | None = None
    page_id: str | None = None
    line_no: int | None = None
    confidence: float | None = None
    source_json: str | None = None


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: str | Path, *, append: bool = False) -> None:
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def page_id_from_filename(name: str) -> str:
    """Strip page suffix (`_p0001`) and extension to get the source-document id."""
    stem = Path(name).stem
    if "_p" in stem:
        stem = stem.rsplit("_p", 1)[0]
    return stem


def read_json_records(path: str | Path) -> list[dict]:
    """Read records from JSONL or a JSON list file."""

    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return list(read_jsonl(p))

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p} must contain a JSON list or JSONL records")
    return data


def line_no_from_name(name: str) -> int | None:
    match = re.search(r"_l(\d+)", Path(name).stem)
    return int(match.group(1)) if match else None


def page_id_from_line_name(name: str) -> str | None:
    stem = Path(name).stem
    match = re.match(r"(.+)_l\d+$", stem)
    return match.group(1) if match else None


def page_id_from_record(record: dict) -> str:
    """Prefer explicit page_id, then nested line folder, then filename stem."""

    if record.get("page_id"):
        return str(record["page_id"])

    file_value = Path(record["file"])
    if file_value.parent != Path("."):
        return file_value.parent.as_posix()

    inferred = page_id_from_line_name(file_value.name)
    return inferred or file_value.stem


def image_exists_for_record(record: dict, lines_dir: str | Path) -> bool:
    if record.get("image_path") and Path(record["image_path"]).exists():
        return True
    return (Path(lines_dir) / record["file"]).exists()


def split_counts(n: int, train: float, val: float, test: float) -> tuple[int, int, int]:
    """Return robust train/val/test counts, keeping val/test non-empty when possible."""

    if n <= 0:
        return 0, 0, 0

    n_val = int(n * val)
    n_test = int(n * test)
    if n >= 3 and val > 0 and n_val == 0:
        n_val = 1
    if n >= 3 and test > 0 and n_test == 0:
        n_test = 1

    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = max(1, n - n_val - n_test)
    return n_train, n_val, n - n_train - n_val


def load_lines(lines_dir: str | Path, *, require_images: bool = True) -> list[LineRecord]:
    """Load line metadata from all ``*.lines.json`` files under ``lines_dir``."""

    lines_root = Path(lines_dir)
    records: list[LineRecord] = []
    for sidecar in sorted(lines_root.rglob("*.lines.json")):
        page_id = sidecar.stem.replace(".lines", "")
        entries = json.loads(sidecar.read_text(encoding="utf-8"))
        for index, entry in enumerate(entries, start=1):
            file_name = entry["file"]
            rel_path = entry.get("rel_path")
            if not rel_path:
                candidate = sidecar.parent / file_name
                try:
                    rel_path = candidate.relative_to(lines_root).as_posix()
                except ValueError:
                    rel_path = file_name

            image_path = lines_root / rel_path
            if require_images and not image_path.exists():
                logging.getLogger(__name__).warning(
                    "missing image for sidecar entry: %s", image_path
                )
                continue

            records.append(
                LineRecord(
                    page_id=page_id,
                    line_no=line_no_from_name(file_name) or index,
                    file=file_name,
                    rel_path=rel_path,
                    image_path=image_path,
                    polygon=entry.get("polygon"),
                    baseline=entry.get("baseline"),
                    pad_px=entry.get("pad_px"),
                )
            )
    return records


def text_from_record(record: dict) -> str | None:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return None


def file_from_record(record: dict) -> str | None:
    for field in FILE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def confidence_from_record(record: dict) -> float | None:
    for field in CONF_FIELDS:
        value = record.get(field)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def line_no_from_record(record: dict, fallback_index: int | None = None) -> int | None:
    for field in ("line_no", "line_number", "line", "index", "idx", "number"):
        value = record.get(field)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    file_value = file_from_record(record)
    if file_value:
        from_file = line_no_from_name(file_value)
        if from_file is not None:
            return from_file
    return fallback_index


def iter_json_files(path: str | Path) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
    else:
        for item in sorted(p.rglob("*.json")):
            if not item.name.endswith(".lines.json"):
                yield item


def normalize_line_dicts(
    records: Iterable[dict],
    *,
    page_id: str | None,
    source_json: Path,
) -> Iterator[TranscriptRecord]:
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue

        text = text_from_record(record)
        if text is None:
            continue

        file_value = file_from_record(record)
        line_no = line_no_from_record(record, fallback_index=index)
        record_page_id = record.get("page_id") or record.get("page") or page_id
        if file_value and not record_page_id:
            record_page_id = page_id_from_line_name(file_value)

        yield TranscriptRecord(
            text=text,
            file=file_value,
            page_id=str(record_page_id) if record_page_id else None,
            line_no=line_no,
            confidence=confidence_from_record(record),
            source_json=source_json.as_posix(),
        )


def load_transcripts(transcripts_path: str | Path) -> list[TranscriptRecord]:
    """Load transcript records from a JSON file or directory of JSON files."""

    transcripts: list[TranscriptRecord] = []
    for path in iter_json_files(transcripts_path):
        data = json.loads(path.read_text(encoding="utf-8"))
        page_id = path.stem

        if isinstance(data, list):
            transcripts.extend(normalize_line_dicts(data, page_id=page_id, source_json=path))
            continue

        if isinstance(data, dict):
            if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
                for file_value, text in data.items():
                    transcripts.append(
                        TranscriptRecord(
                            text=text.strip(),
                            file=file_value,
                            page_id=page_id_from_line_name(file_value),
                            line_no=line_no_from_name(file_value),
                            source_json=path.as_posix(),
                        )
                    )
                continue

            for field in LIST_FIELDS:
                if isinstance(data.get(field), list):
                    record_page_id = data.get("page_id") or data.get("page") or page_id
                    transcripts.extend(
                        normalize_line_dicts(
                            data[field], page_id=str(record_page_id), source_json=path
                        )
                    )
                    break
            else:
                transcripts.extend(normalize_line_dicts([data], page_id=page_id, source_json=path))

    return transcripts
