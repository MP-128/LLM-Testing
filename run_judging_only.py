"""Run judging only for models that have generation data but missing judgments."""
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval_harness import config
from eval_harness.data_loader import Problem
from eval_harness.generator import _gen_path
from eval_harness.judge import judge_dataset, _judge_path
from eval_harness.storage import load_all

QUESTIONS_JSON = Path(__file__).parent.parent / "questions.json"
DATASET_KEY = "questions_json"

def load_questions() -> list[Problem]:
    with open(QUESTIONS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    problems = []
    for item in data:
        problems.append(Problem(
            id=item["question_id"],
            dataset=DATASET_KEY,
            type="rubric",
            scenario=item.get("scenario", ""),
            question=item.get("tasks", ""),
            correct_answer=item.get("correct_answer", ""),
            rubric=item.get("rubric", ""),
            category=item.get("category", ""),
            reference="",
        ))
    return problems


def dedup_keys(records):
    seen = {}
    for r in records:
        key = (r["question_id"], r["sample_idx"])
        seen[key] = r
    return seen


async def main():
    problems = load_questions()
    problem_map = {p.id: p for p in problems}
    semaphore = asyncio.Semaphore(config.MAX_JUDGE_CONCURRENCY)

    for model in config.MODELS:
        display = config.MODELS[model].get("display", model)
        gen_path = _gen_path(model, DATASET_KEY)
        judge_path = _judge_path(model, DATASET_KEY)

        gen_records = load_all(gen_path)
        gen_dedup = dedup_keys(gen_records)

        judge_records = load_all(judge_path)
        judge_dedup = dedup_keys(judge_records)

        missing = {k: v for k, v in gen_dedup.items() if k not in judge_dedup}

        if not missing:
            print(f"  {display:<28} all {len(gen_dedup)} judged - SKIP")
            continue

        print(f"  {display:<28} {len(missing)} judgments needed (of {len(gen_dedup)} gen)...")

        # Get relevant problems and records
        missing_qids = {k[0] for k in missing}
        relevant_problems = [p for p in problems if p.id in missing_qids]
        missing_records = list(missing.values())

        try:
            await judge_dataset(
                model, relevant_problems, missing_records,
                semaphore=semaphore,
            )
            # Re-check
            new_judge = dedup_keys(load_all(judge_path))
            print(f"  {display:<28} now {len(new_judge)} judgments")
        except Exception as e:
            print(f"  {display:<28} ERROR: {e}")

    print("\nDone! Run generate_pass3_outputs.py to create output files.")


if __name__ == "__main__":
    asyncio.run(main())
