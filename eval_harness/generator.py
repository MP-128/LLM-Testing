"""Async LLM generation with retry, rate limiting, and resume support."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from .config import (
    API_KEYS, GENERATION_TEMPERATURE, GENERATIONS_DIR, K,
    MAX_RETRIES, MAX_TOKENS, MODELS, OPENROUTER_BASE_URL, RETRY_BASE_DELAY,
)
from .data_loader import Problem
from .storage import already_done_ids, append


def _gen_path(model: str, dataset: str) -> Path:
    safe = model.replace("/", "_").replace(":", "_")
    return GENERATIONS_DIR / f"{safe}__{dataset}.jsonl"


async def _call_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt + 1, MAX_RETRIES, fn.__name__, e,
            )
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)


# ── Provider implementations ──────────────────────────────────────────────────

async def _generate_openrouter(model: str, prompt: str) -> str:
    """Route all generation through OpenRouter (OpenAI-compatible API)."""
    from openai import AsyncOpenAI
    api_model = MODELS[model].get("api_model", model)
    client = AsyncOpenAI(
        api_key=API_KEYS["openrouter"],
        base_url=OPENROUTER_BASE_URL,
    )
    kwargs: dict = dict(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=MAX_TOKENS,
    )
    # Some models don't support custom temperature
    if not model.startswith("o") and "gpt-5.3" not in model:
        kwargs["temperature"] = GENERATION_TEMPERATURE
    resp = await client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    if not content:
        raise ValueError(f"OpenRouter returned empty/null content for {api_model}")
    return content


_PROVIDERS = {
    "openrouter": _generate_openrouter,
}


def _build_prompt(problem: Problem) -> str:
    return (
        f"You are a domain expert. Answer the following question carefully and thoroughly.\n\n"
        f"## Scenario\n{problem.scenario}\n\n"
        f"## Question\n{problem.question}\n\n"
        "Provide your answer below:"
    )


async def generate_for_problem(
    model: str,
    problem: Problem,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    provider = MODELS[model]["provider"]
    fn = _PROVIDERS[provider]
    prompt = _build_prompt(problem)

    results = []
    for sample_idx in range(K):
        async with semaphore:
            text = await _call_with_retry(fn, model, prompt)
        if len(text.strip()) < 50:
            logger.warning(
                "Short response (%d chars) from %s for question %s sample %d",
                len(text.strip()), model, problem.id, sample_idx,
            )
        record = {
            "question_id": problem.id,
            "dataset": problem.dataset,
            "model": model,
            "sample_idx": sample_idx,
            "response": text,
            "timestamp": time.time(),
        }
        results.append(record)
    return results


async def generate_dataset(
    model: str,
    problems: list[Problem],
    concurrency: int = 5,
) -> Path:
    dataset_key = problems[0].dataset if problems else "unknown"
    out_path = _gen_path(model, dataset_key)
    done = already_done_ids(out_path)

    remaining = [p for p in problems if p.id not in done]
    if not remaining:
        return out_path

    semaphore = asyncio.Semaphore(concurrency)

    async def _process(problem: Problem):
        records = await generate_for_problem(model, problem, semaphore)
        for r in records:
            append(out_path, r)
        return records

    await asyncio.gather(*[_process(p) for p in remaining])
    return out_path
