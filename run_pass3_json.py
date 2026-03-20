"""
Pass@3 evaluation on a standalone JSON dataset.

For each question:
  - Generate 3 independent responses (temperature=0.8, no shared context)
  - Judge each response against the rubric
  - pass_at_3 = 1 if ANY response passes, else 0

Output:
  results/metrics/pass3_<dataset_key>_summary.json   – each model's pass@3 %
  results/metrics/pass3_<dataset_key>_details.json    – per-question attempt outputs + pass/fail
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure the package root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import config, storage
from eval_harness.data_loader import Problem
from eval_harness.generator import generate_for_problem, _gen_path
from eval_harness.judge import judge_response, _judge_path
from eval_harness.storage import append, load_all, already_done_ids

import argparse

MODELS = list(config.MODELS.keys())
K = config.K  # 3

parser = argparse.ArgumentParser()
parser.add_argument("--json-file", required=True, help="Path to the JSON dataset file")
parser.add_argument("--dataset-key", default=None, help="Short key for output files (derived from filename if omitted)")
_args, _ = parser.parse_known_args()

JSON_FILE = Path(_args.json_file).resolve()
DATASET_KEY = _args.dataset_key or JSON_FILE.stem.lower().replace(" ", "_").replace("&", "and")


def load_json_dataset(path: Path) -> list[Problem]:
    """Load a JSON dataset into Problem objects, handling various field name conventions."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    problems = []
    for item in data:
        # Support both lowercase and capitalized field names
        qid = item.get("Question_ID") or item.get("question_id") or ""
        scenario = item.get("Scenario") or item.get("scenario") or ""
        tasks = item.get("Tasks") or item.get("tasks") or ""
        correct = item.get("Correct_Answer") or item.get("correct_answer") or ""
        rubric = item.get("Rubric") or item.get("rubric") or ""
        category = item.get("Category") or item.get("category") or ""

        problems.append(Problem(
            id=str(qid).strip(),
            dataset=DATASET_KEY,
            type="rubric",
            scenario=str(scenario).strip(),
            question=str(tasks).strip(),
            correct_answer=str(correct).strip(),
            rubric=str(rubric).strip(),
            category=str(category).strip(),
        ))
    return problems


async def run_model(model: str, problems: list[Problem], judge_semaphore: asyncio.Semaphore) -> dict:
    """Run the full pass@3 pipeline for a single model. Returns summary stats."""
    display = config.MODELS[model].get("display", model)

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\n[{display}] Generating {K} responses per problem ...")
    gen_path = _gen_path(model, DATASET_KEY)
    done = already_done_ids(gen_path)
    remaining = [p for p in problems if p.id not in done]

    if remaining:
        gen_semaphore = asyncio.Semaphore(5)
        async def _gen(problem):
            records = await generate_for_problem(model, problem, gen_semaphore)
            for r in records:
                append(gen_path, r)
        await asyncio.gather(*[_gen(p) for p in remaining])

    gen_records = load_all(gen_path)
    print(f"[{display}]   {len(gen_records)} generation records")

    # ── Judge ─────────────────────────────────────────────────────────────────
    print(f"[{display}] Judging with {config.JUDGE_MODEL} ...")
    judge_path = _judge_path(model, DATASET_KEY)
    judged = {(r["question_id"], r["sample_idx"]) for r in load_all(judge_path)}
    problem_map = {p.id: p for p in problems}

    async def _judge(rec):
        qid = rec["question_id"]
        idx = rec["sample_idx"]
        if (qid, idx) in judged:
            return
        problem = problem_map.get(qid)
        if not problem:
            return
        result = await judge_response(problem, rec["response"], model, idx, judge_semaphore, config.JUDGE_MODEL)
        append(judge_path, {
            "question_id": result.question_id,
            "dataset": result.dataset,
            "model": result.model,
            "sample_idx": result.sample_idx,
            "category": problem.category,
            "criteria_scores": result.criteria_scores,
            "pass_score": result.pass_score,
            "passed": result.passed,
            "reasoning": result.reasoning,
            "judgment_type": result.judgment_type,
            "timestamp": result.timestamp,
        })

    await asyncio.gather(*[_judge(r) for r in gen_records])

    judge_records = load_all(judge_path)
    print(f"[{display}]   {len(judge_records)} judgment records")

    # ── Build lookup tables ───────────────────────────────────────────────────
    gen_lookup: dict[tuple[str, int], str] = {
        (r["question_id"], r["sample_idx"]): r["response"]
        for r in gen_records
    }
    judge_lookup: dict[tuple[str, int], dict] = {
        (r["question_id"], r["sample_idx"]): r
        for r in judge_records
    }

    # ── Build per-question rows ───────────────────────────────────────────────
    rows: list[dict] = []
    for problem in problems:
        qid = problem.id
        row: dict = {
            "model": display,
            "question_id": qid,
            "category": problem.category,
            "attempts": [],
        }
        any_pass = False
        for i in range(K):
            key = (qid, i)
            response = gen_lookup.get(key, "")
            judge_rec = judge_lookup.get(key, {})
            passed = bool(judge_rec.get("passed", False))
            attempt = {
                "attempt_number": i + 1,
                "response": response,
                "passed": passed,
                "pass_score": judge_rec.get("pass_score", 0.0),
                "reasoning": judge_rec.get("reasoning", ""),
            }
            row["attempts"].append(attempt)
            if passed:
                any_pass = True
        row["pass_at_3"] = 1 if any_pass else 0
        rows.append(row)

    total = len(rows)
    passes = sum(r["pass_at_3"] for r in rows)
    return {"model": display, "rows": rows, "total": total, "passes": passes}


async def main() -> None:
    # ── 1. Load JSON dataset ─────────────────────────────────────────────────
    print(f"Loading questions from: {JSON_FILE}")
    problems = load_json_dataset(JSON_FILE)
    print(f"  {len(problems)} problems loaded")

    # ── 2. Run all models in parallel ─────────────────────────────────────────
    judge_semaphore = asyncio.Semaphore(config.MAX_JUDGE_CONCURRENCY)

    results = await asyncio.gather(
        *[run_model(model, problems, judge_semaphore) for model in MODELS],
        return_exceptions=True,
    )

    all_details: list[dict] = []
    summaries: list[dict] = []
    for model, result in zip(MODELS, results):
        if isinstance(result, BaseException):
            display = config.MODELS[model].get("display", model)
            print(f"[WARNING] {display} failed: {result}")
            continue
        all_details.extend(result["rows"])
        pct = result["passes"] / result["total"] if result["total"] else 0.0
        summaries.append({
            "model": result["model"],
            "total_questions": result["total"],
            "passed": result["passes"],
            "pass_at_3_pct": round(pct * 100, 2),
        })

    # ── 3. Save summary JSON ─────────────────────────────────────────────────
    summary_path = config.METRICS_DIR / f"pass3_{DATASET_KEY}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    # ── 4. Save details JSON ─────────────────────────────────────────────────
    details_path = config.METRICS_DIR / f"pass3_{DATASET_KEY}_details.json"
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2, ensure_ascii=False)
    print(f"Details saved to: {details_path}")

    # ── 5. Summary table ─────────────────────────────────────────────────────
    col = 24
    print(f"\n{'=' * 52}")
    print(f"  PASS@3 EVALUATION SUMMARY  —  {DATASET_KEY}")
    print(f"{'=' * 52}")
    print(f"  {'Model':<{col}} {'Passed':>7}  {'pass@3':>8}")
    print(f"  {'-' * (col + 18)}")
    for s in summaries:
        pct = s["pass_at_3_pct"]
        print(f"  {s['model']:<{col}} {s['passed']:>4}/{s['total_questions']:<3}  {pct:>7.2f}%")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    asyncio.run(main())
