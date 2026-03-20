"""JSONL append/resume storage so no API work is lost on failures."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt JSONL line in %s: %s", path, line[:100])
    return records


def load_all_deduped(
    path: Path,
    key_fields: tuple[str, ...] = ("question_id", "sample_idx"),
    prefer: str = "latest",
) -> list[dict]:
    """Load JSONL and deduplicate, keeping one record per unique key.

    prefer: 'latest' keeps the last record by file order,
            'highest' keeps the record with the highest pass_score.
    """
    records = load_all(path)
    lookup: dict[tuple, dict] = {}
    for r in records:
        k = tuple(r.get(f) for f in key_fields)
        if k in lookup and prefer == "highest":
            if r.get("pass_score", 0) > lookup[k].get("pass_score", 0):
                lookup[k] = r
        else:
            lookup[k] = r
    return list(lookup.values())


def already_done_ids(path: Path) -> set[str]:
    """Return set of question_ids that already have k entries in the JSONL."""
    from collections import Counter
    from .config import K
    records = load_all(path)
    counts = Counter(r["question_id"] for r in records)
    return {qid for qid, cnt in counts.items() if cnt >= K}
