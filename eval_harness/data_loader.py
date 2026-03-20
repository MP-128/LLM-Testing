"""Load and normalize all 5 xlsx datasets into a unified schema."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import openpyxl

from .config import DATA_DIR, DATASETS

ProblemType = Literal["binary", "numeric", "rubric"]

# Insurance dataset answers that are binary approve/deny
_BINARY_ANSWERS = {"approve claim", "deny claim", "approve", "deny"}


@dataclass
class Problem:
    id: str
    dataset: str          # short key (e.g. "insurance")
    type: ProblemType
    scenario: str
    question: str
    correct_answer: str   # always stored as string, even for numeric
    rubric: str
    category: str
    reference: str = ""


def _detect_type(dataset_key: str, correct_answer) -> ProblemType:
    if dataset_key == "insurance":
        return "binary"
    if dataset_key == "multistep":
        # Check if the answer is numeric or can be parsed as one
        if isinstance(correct_answer, (int, float)):
            return "numeric"
        try:
            float(str(correct_answer).replace(",", "").replace("x", "").strip())
            return "numeric"
        except ValueError:
            pass
    return "rubric"


def _to_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def load_dataset(key: str) -> list[Problem]:
    filename = DATASETS[key]
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}\nRun: cp '<xlsx>' {path}")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)

    # Read and normalise header
    raw_header = next(rows, None)
    if raw_header is None:
        return []

    # Build a name→index map, normalising header strings to lowercase snake_case
    _norm = {
        "question_id": "qid",
        "scenario": "scenario",
        "question": "question",
        "tasks": "question",
        "correct_answer": "correct_answer",
        "rubric": "rubric",
        # accept both column name variants for rubric
        "rubric (deterministic decision rules)": "rubric",
        "category": "category",
        # accept the verbose category header used by ca_claims
        "category (choose one: water damage, fire damage, wind/hail damage, theft/burglary, liability, wildfire damage, auto damage, vandalism, earthquake, flood)": "category",
        "reference docs": "reference",
        "reference_docs": "reference",
    }
    col_idx: dict[str, int] = {}
    for i, h in enumerate(raw_header):
        if h is None:
            continue
        key_norm = str(h).strip().lower()
        mapped = _norm.get(key_norm)
        if mapped and mapped not in col_idx:
            col_idx[mapped] = i

    def _get(row, field: str):
        idx = col_idx.get(field)
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    problems: list[Problem] = []
    for row in rows:
        if not any(row):
            continue
        qid           = _get(row, "qid")
        scenario      = _get(row, "scenario")
        question      = _get(row, "question")
        correct_answer = _get(row, "correct_answer")
        rubric        = _get(row, "rubric")
        category      = _get(row, "category")
        reference     = _get(row, "reference")

        if not qid or not question:
            continue

        ptype = _detect_type(key, correct_answer)
        problems.append(Problem(
            id=_to_str(qid),
            dataset=key,
            type=ptype,
            scenario=_to_str(scenario),
            question=_to_str(question),
            correct_answer=_to_str(correct_answer),
            rubric=_to_str(rubric),
            category=_to_str(category),
            reference=_to_str(reference),
        ))

    wb.close()
    return problems


def load_all() -> list[Problem]:
    problems = []
    for key in DATASETS:
        problems.extend(load_dataset(key))
    return problems


def load_selected(dataset_keys: list[str]) -> list[Problem]:
    problems = []
    for key in dataset_keys:
        problems.extend(load_dataset(key))
    return problems


def extract_numeric(text: str) -> float | None:
    """Pull the first number (int or float, possibly with trailing x/%) from text."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[xX%])?", text.replace(",", ""))
    if match:
        return float(match.group().rstrip("xX%"))
    return None
