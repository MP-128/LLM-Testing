"""Compute pass@k using the unbiased estimator from Chen et al. 2021."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import K
from .storage import load_all


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator for pass@k.
    n = total samples generated per problem
    c = number of passing samples
    k = target k
    """
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_metrics(judgment_path: Path, k: int = K) -> dict[str, Any]:
    """
    Returns:
        {
          "overall": {"pass@k": float, "n_problems": int},
          "by_category": {"Finance": {"pass@k": float, ...}, ...},
          "per_problem": [{"question_id": ..., "pass@k": float, ...}, ...]
        }
    """
    records = load_all(judgment_path)
    if not records:
        return {}

    # Group: question_id -> list of judgment records
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_qid[r["question_id"]].append(r)

    per_problem = []
    for qid, judgments in by_qid.items():
        n = len(judgments)
        c = sum(1 for j in judgments if j["passed"])
        pk = pass_at_k(n, c, min(k, n))
        per_problem.append({
            "question_id": qid,
            "dataset": judgments[0].get("dataset", ""),
            "category": judgments[0].get("category", "unknown"),
            "n_samples": n,
            "n_passing": c,
            f"pass@{k}": pk,
        })

    overall_pk = sum(p[f"pass@{k}"] for p in per_problem) / len(per_problem)

    # By category (need category info from judgments if stored, else skip)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for p in per_problem:
        cat = p.get("category", "unknown")
        by_cat[cat].append(p[f"pass@{k}"])

    by_category = {
        cat: {f"pass@{k}": sum(scores) / len(scores), "n_problems": len(scores)}
        for cat, scores in by_cat.items()
    }

    return {
        "overall": {f"pass@{k}": overall_pk, "n_problems": len(per_problem)},
        "by_category": by_category,
        "per_problem": per_problem,
    }


def compute_all_metrics(
    model_dataset_paths: list[tuple[str, str, Path]],  # (model, dataset, judgment_path)
    k: int = K,
) -> dict[str, Any]:
    """Aggregate metrics across all model x dataset combinations."""
    results: dict[str, dict] = {}
    for model, dataset, path in model_dataset_paths:
        key = f"{model}__{dataset}"
        results[key] = {"model": model, "dataset": dataset, **compute_metrics(path, k)}
    return results


def print_summary_table(results: dict[str, Any], k: int = K) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"pass@{k} Results")
    table.add_column("Model", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column(f"pass@{k}", style="green", justify="right")
    table.add_column("N Problems", justify="right")

    for key, data in sorted(results.items()):
        if "overall" not in data:
            continue
        table.add_row(
            data["model"],
            data["dataset"],
            f"{data['overall'][f'pass@{k}']:.3f}",
            str(data["overall"]["n_problems"]),
        )

    console.print(table)
