"""
Pass@3 evaluation on P&C Insurance Sample v3.json.

Splits the questions into 3 shards and runs them in parallel.
For each question x model:
  - Generate 3 independent responses (temperature=0.8)
  - Judge each response against the rubric
  - pass@3 = 1 if ANY response passes, else 0

Outputs 2 JSON files:
  1. pass3_v3_model_summary.json       - pass@3 % per model
  2. pass3_v3_question_attempts.json    - per-question detail with all 3 attempts and pass/fail
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import config, storage
from eval_harness.data_loader import Problem
from eval_harness.generator import generate_dataset, _gen_path
from eval_harness.judge import judge_dataset, _judge_path
from eval_harness.storage import load_all, load_all_deduped

# ── Configuration ────────────────────────────────────────────────────────────
V3_JSON = Path(__file__).parent.parent / "P&C Insurance Sample v3.json"
DATASET_KEY = "pc_insurance_v3"
K = 3
MODELS = list(config.MODELS.keys())
OUTPUT_DIR = config.METRICS_DIR


def load_v3_questions() -> list[Problem]:
    """Load P&C Insurance Sample v3.json into Problem objects."""
    with open(V3_JSON, encoding="utf-8") as f:
        data = json.load(f)

    problems = []
    for item in data:
        rows = item.get("rows", {})
        problems.append(Problem(
            id=item["question_id"],
            dataset=DATASET_KEY,
            type="rubric",
            scenario=rows.get("Scenario", ""),
            question=rows.get("Tasks", ""),
            correct_answer=rows.get("Correct_Answer", ""),
            rubric=rows.get("Rubric", ""),
            category=rows.get("Category", ""),
            reference="",
        ))
    return problems


def split_into_shards(problems: list[Problem], n_shards: int = 3) -> list[list[Problem]]:
    shards = [[] for _ in range(n_shards)]
    for i, p in enumerate(problems):
        shards[i % n_shards].append(p)
    return shards


async def run_model_on_shard(
    model: str,
    shard_problems: list[Problem],
    shard_idx: int,
    judge_semaphore: asyncio.Semaphore,
):
    display = config.MODELS[model].get("display", model)
    print(f"  [Shard {shard_idx+1}] [{display}] Generating {K} responses for {len(shard_problems)} questions...")

    try:
        gen_path = await generate_dataset(model, shard_problems, concurrency=5)
        gen_records = load_all(gen_path)
        shard_qids = {p.id for p in shard_problems}
        shard_gen = [r for r in gen_records if r["question_id"] in shard_qids]
        print(f"  [Shard {shard_idx+1}] [{display}] {len(shard_gen)} generations, now judging...")

        judge_path = await judge_dataset(
            model, shard_problems, shard_gen,
            semaphore=judge_semaphore,
        )
        judge_records = load_all(judge_path)
        shard_judge = [r for r in judge_records if r["question_id"] in shard_qids]
        print(f"  [Shard {shard_idx+1}] [{display}] {len(shard_judge)} judgments done")
    except Exception as e:
        print(f"  [Shard {shard_idx+1}] [{display}] ERROR: {e}")


async def run_shard(
    shard_idx: int,
    shard_problems: list[Problem],
    judge_semaphore: asyncio.Semaphore,
):
    print(f"\n{'='*50}")
    print(f"  SHARD {shard_idx+1}/3 - {len(shard_problems)} questions")
    print(f"{'='*50}")

    for model in MODELS:
        await run_model_on_shard(model, shard_problems, shard_idx, judge_semaphore)


async def main():
    start = time.time()
    print("Loading P&C Insurance Sample v3.json...")
    problems = load_v3_questions()
    print(f"  {len(problems)} questions loaded")
    print(f"  Models: {[config.MODELS[m].get('display', m) for m in MODELS]}")

    shards = split_into_shards(problems, 3)
    for i, s in enumerate(shards):
        print(f"  Shard {i+1}: {len(s)} questions ({[p.id for p in s]})")

    judge_semaphore = asyncio.Semaphore(config.MAX_JUDGE_CONCURRENCY)

    print("\nRunning 3 shards in parallel...")
    await asyncio.gather(
        run_shard(0, shards[0], judge_semaphore),
        run_shard(1, shards[1], judge_semaphore),
        run_shard(2, shards[2], judge_semaphore),
    )

    elapsed = time.time() - start
    print(f"\nAll shards complete in {elapsed:.1f}s. Building output files...")

    # ── Collect results and build JSONs ───────────────────────────────────────
    model_summaries = {}
    question_attempts = {}

    for model in MODELS:
        display = config.MODELS[model].get("display", model)
        gen_path = _gen_path(model, DATASET_KEY)
        judge_path = _judge_path(model, DATASET_KEY)

        gen_records = load_all_deduped(gen_path)
        judge_records = load_all_deduped(judge_path)

        gen_lookup = {}
        for r in gen_records:
            gen_lookup[(r["question_id"], r["sample_idx"])] = r["response"]

        judge_lookup = {}
        for r in judge_records:
            judge_lookup[(r["question_id"], r["sample_idx"])] = {
                "passed": bool(r["passed"]),
                "pass_score": r.get("pass_score", 0),
                "reasoning": r.get("reasoning", ""),
                "criteria_scores": r.get("criteria_scores", []),
            }

        model_tasks = []
        total = 0
        passed_count = 0

        for problem in problems:
            qid = problem.id
            attempts = []
            any_pass = False

            for i in range(K):
                response = gen_lookup.get((qid, i), "")
                judgment = judge_lookup.get((qid, i), {})
                attempt_passed = judgment.get("passed", False)
                if attempt_passed:
                    any_pass = True

                attempts.append({
                    "attempt": i + 1,
                    "response": response,
                    "passed": attempt_passed,
                    "pass_score": judgment.get("pass_score", 0),
                    "reasoning": judgment.get("reasoning", ""),
                    "criteria_scores": judgment.get("criteria_scores", []),
                })

            total += 1
            if any_pass:
                passed_count += 1

            model_tasks.append({
                "question_id": qid,
                "category": problem.category,
                "task": problem.question,
                "correct_answer": problem.correct_answer,
                "pass_at_3": 1 if any_pass else 0,
                "attempts": attempts,
            })

        pct = (passed_count / total * 100) if total > 0 else 0
        model_summaries[display] = {
            "model": display,
            "total_questions": total,
            "questions_passed": passed_count,
            "pass_at_3_pct": round(pct, 2),
        }
        question_attempts[display] = model_tasks

    # ── Write output files ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Summary: pass@3 % per model
    summary_path = OUTPUT_DIR / "pass3_v3_model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation": "pass@3",
            "dataset": "P&C Insurance Sample v3.json",
            "total_questions": len(problems),
            "models": model_summaries,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  1) Model summary:      {summary_path}")

    # 2) Question attempts: per-model per-question with all 3 attempts and pass/fail
    attempts_path = OUTPUT_DIR / "pass3_v3_question_attempts.json"
    with open(attempts_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation": "pass@3",
            "dataset": "P&C Insurance Sample v3.json",
            "total_questions": len(problems),
            "k": K,
            "models_evaluated": [config.MODELS[m].get("display", m) for m in MODELS],
            "judge_model": config.JUDGE_MODEL,
            "temperature": config.GENERATION_TEMPERATURE,
            "pass_threshold": config.PASS_THRESHOLD,
            "by_model": question_attempts,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, ensure_ascii=False)
    print(f"  2) Question attempts:  {attempts_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PASS@3 RESULTS - P&C Insurance Sample v3 ({len(problems)} questions)")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'Passed':>8}  {'pass@3 %':>10}")
    print(f"  {'-'*48}")
    for display, stats in model_summaries.items():
        print(f"  {display:<28} {stats['questions_passed']:>4}/{stats['total_questions']:<4} {stats['pass_at_3_pct']:>9.2f}%")
    print(f"{'='*60}")
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
