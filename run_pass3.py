"""
Pass@3 evaluation on a configurable dataset.

For each question:
  - Generate 3 independent responses (temperature=0.8, no shared context)
  - Judge each response against the rubric
  - pass_at_3 = 1 if ANY response passes, else 0

Output: results/metrics/pass3_{dataset}_results.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

# Ensure the package root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import config, storage
from eval_harness.data_loader import load_dataset
from eval_harness.generator import generate_dataset, _gen_path
from eval_harness.judge import judge_dataset, _judge_path

MODELS = list(config.MODELS.keys())
K = config.K  # 3

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="multistep", choices=list(config.DATASETS.keys()),
                    help="Dataset key to evaluate (default: multistep)")
_args, _ = parser.parse_known_args()
DATASET_KEY = _args.dataset


async def run_model(model: str, problems: list, judge_semaphore: asyncio.Semaphore) -> dict:
    """Run the full pass@3 pipeline for a single model. Returns summary stats."""
    display = config.MODELS[model].get("display", model)

    # ── Generate ─────────────────────────────────────────────────────────────
    print(f"\n[{display}] Generating {K} responses per problem ...")
    gen_path = await generate_dataset(model, problems)

    gen_records = storage.load_all(gen_path)
    print(f"[{display}]   {len(gen_records)} generation records")

    # ── Judge ────────────────────────────────────────────────────────────────
    print(f"[{display}] Judging with {config.JUDGE_MODEL} ...")
    judge_path = await judge_dataset(model, problems, gen_records, semaphore=judge_semaphore)

    judge_records = storage.load_all(judge_path)
    print(f"[{display}]   {len(judge_records)} judgment records")

    # ── Build lookup tables ──────────────────────────────────────────────────
    gen_lookup: dict[tuple[str, int], str] = {
        (r["question_id"], r["sample_idx"]): r["response"]
        for r in gen_records
    }
    judge_lookup: dict[tuple[str, int], bool] = {
        (r["question_id"], r["sample_idx"]): bool(r["passed"])
        for r in judge_records
    }

    # ── Build per-question rows ──────────────────────────────────────────────
    rows: list[dict] = []
    for problem in problems:
        qid = problem.id
        row: dict = {"model": display, "Question_id": qid}
        any_pass = False
        for i in range(K):
            key = (qid, i)
            response = gen_lookup.get(key, "")
            passed = judge_lookup.get(key, False)
            row[f"response_{i + 1}"] = response
            row[f"response_{i + 1}_judgment"] = "PASS" if passed else "FAIL"
            if passed:
                any_pass = True
        row["pass_at_3"] = 1 if any_pass else 0
        rows.append(row)

    total = len(rows)
    passes = sum(r["pass_at_3"] for r in rows)
    return {"model": display, "rows": rows, "total": total, "passes": passes}


async def main() -> None:
    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset: {DATASET_KEY}")
    problems = load_dataset(DATASET_KEY)
    print(f"  {len(problems)} problems loaded")

    # ── 2. Run all models in parallel ────────────────────────────────────────
    # All models share the same judge endpoint (claude-sonnet-4-6), so a global
    # semaphore caps total concurrent judge calls across all parallel model runs.
    judge_semaphore = asyncio.Semaphore(config.MAX_JUDGE_CONCURRENCY)

    results = await asyncio.gather(
        *[run_model(model, problems, judge_semaphore) for model in MODELS],
        return_exceptions=True,
    )

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for model, result in zip(MODELS, results):
        if isinstance(result, BaseException):
            display = config.MODELS[model].get("display", model)
            print(f"[WARNING] {display} failed: {result}")
            continue
        all_rows.extend(result["rows"])
        summaries.append(result)

    # ── 3. Save combined CSV ─────────────────────────────────────────────────
    output_csv = config.METRICS_DIR / f"pass3_{DATASET_KEY}_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "Question_id",
        "response_1", "response_1_judgment",
        "response_2", "response_2_judgment",
        "response_3", "response_3_judgment",
        "pass_at_3",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to: {output_csv}")

    # ── 4. Summary table ─────────────────────────────────────────────────────
    col = 24
    print(f"\n{'=' * 52}")
    print(f"  PASS@3 EVALUATION SUMMARY  —  {DATASET_KEY}")
    print(f"{'=' * 52}")
    print(f"  {'Model':<{col}} {'Passed':>7}  {'pass@3':>8}")
    print(f"  {'-' * (col + 18)}")
    for s in summaries:
        pct = s["passes"] / s["total"] if s["total"] else 0.0
        print(f"  {s['model']:<{col}} {s['passes']:>4}/{s['total']:<3}  {pct:.4f}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    asyncio.run(main())
