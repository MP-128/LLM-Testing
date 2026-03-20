"""LLM-as-judge: score each response against the rubric, return structured JSON."""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import (
    API_KEYS, JUDGE_MODEL, JUDGMENTS_DIR, MAX_JUDGE_CONCURRENCY,
    MAX_RETRIES, MODELS, OPENROUTER_BASE_URL, PASS_THRESHOLD, RETRY_BASE_DELAY,
)
from .data_loader import Problem, ProblemType, extract_numeric
from .storage import append, load_all

# Pre-compiled regex patterns
_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
_LATEX_THOUSAND_RE = re.compile(r'\{,\}')
_LATEX_ESCAPE_RE = re.compile(r'\\(?=\$)')
_LATEX_LBRACE_RE = re.compile(r'\{(\d)')
_LATEX_RBRACE_RE = re.compile(r'(\d)\}')
_CODE_FENCE_RE = re.compile(r"```(?:json)?")
_JSON_GREEDY_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_FLAT_RE = re.compile(r"\{[^{}]*\}")
_CRITERIA_RE = re.compile(r"^\s*\d+\.", re.MULTILINE)


def _normalize_latex_dollars(text: str) -> str:
    """Normalize LaTeX-formatted dollar amounts for regex extraction."""
    text = _LATEX_THOUSAND_RE.sub(',', text)    # {,} -> ,  (LaTeX thousand sep)
    text = _LATEX_ESCAPE_RE.sub('', text)       # \$ -> $   (LaTeX dollar escape)
    text = _LATEX_LBRACE_RE.sub(r'\1', text)    # {2 -> 2   (leftover braces)
    text = _LATEX_RBRACE_RE.sub(r'\1', text)    # 2} -> 2   (leftover braces)
    return text


def _extract_dollar(text: str) -> float | None:
    """Extract the first dollar amount from text (e.g. '$52,750' -> 52750.0)."""
    text = _normalize_latex_dollars(text)
    match = _DOLLAR_RE.search(text)
    if match:
        return float(match.group().replace("$", "").replace(",", ""))
    return None


_HEADLINE_KEYWORDS = re.compile(
    r"(?:net payment|total payment|net covered payment|amount payable|"
    r"total insurer exposure|maximum insurer exposure|"
    r"net to insured|net to.*?husband|net to.*?wife|"
    r"final answer|total payout|should pay|policy.*?pay)[^$\n]{0,40}"
    r"(\$[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _extract_headline_dollar(text: str) -> float | None:
    """Extract the headline/final-answer dollar amount from a model response.

    Strategy:
    1. Look for dollar amounts within 15 chars of summary keywords
       (net payment, total payment, amount payable, etc.)
       and take the LARGEST such match (headline amounts are the biggest
       figure near conclusion keywords, not incidental small amounts).
    2. Fall back to the LAST dollar amount in the text.
    """
    text = _normalize_latex_dollars(text)
    # Strategy 1: keyword-adjacent amounts — take the LAST match
    # (conclusions/final answers appear after calculations in responses)
    keyword_matches = _HEADLINE_KEYWORDS.findall(text)
    if keyword_matches:
        amounts = []
        for raw in keyword_matches:
            try:
                amounts.append(float(raw.replace("$", "").replace(",", "")))
            except ValueError:
                continue
        if amounts:
            return amounts[-1]

    # Strategy 2: last dollar amount in the text
    all_matches = _DOLLAR_RE.findall(text)
    if all_matches:
        return float(all_matches[-1].replace("$", "").replace(",", ""))

    return None


def _count_rubric_criteria(rubric: str) -> int | None:
    """Parse the rubric to count top-level numbered criteria (e.g., '1. Name (X pts):').

    Returns the count, or None if the rubric format is not parseable.
    """
    matches = _CRITERIA_RE.findall(rubric)
    return len(matches) if matches else None


def _coverage_direction_matches(correct: str, response: str) -> bool:
    """Check if the response's coverage direction matches the correct answer."""
    correct_l = correct.lower()
    response_l = response.lower()

    denied_words = ["excluded", "denied", "not covered", "deny"]
    approved_words = ["covered", "approved", "approve"]

    correct_denied = any(w in correct_l for w in denied_words)
    correct_approved = any(w in correct_l for w in approved_words)

    # Mixed coverage (e.g. "fire: COVERED, vandalism: EXCLUDED") — skip gate
    if correct_denied and correct_approved:
        return True

    # Check if response contradicts correct direction
    if correct_denied:
        resp_approved = any(w in response_l for w in approved_words)
        resp_denied = any(w in response_l for w in denied_words)
        if resp_approved and not resp_denied:
            return False
    if correct_approved:
        resp_denied = any(w in response_l for w in denied_words)
        resp_approved = any(w in response_l for w in approved_words)
        if resp_denied and not resp_approved:
            return False

    return True


@dataclass
class JudgeResult:
    question_id: str
    dataset: str
    model: str           # model being judged (not the judge itself)
    sample_idx: int
    criteria_scores: list[int]   # 0/1/2 per criterion
    pass_score: float            # sum / (2 * n_criteria)
    passed: bool
    reasoning: str
    raw_response: str = ""
    judgment_type: str = "rubric"  # "rubric" | "binary" | "numeric"
    timestamp: float = 0.0


def _judge_path(model: str, dataset: str) -> Path:
    safe = model.replace("/", "_").replace(":", "_")
    return JUDGMENTS_DIR / f"{safe}__{dataset}.jsonl"


def _already_judged(path: Path) -> set[tuple[str, int]]:
    records = load_all(path)
    return {(r["question_id"], r["sample_idx"]) for r in records}


# ── Rubric prompt ─────────────────────────────────────────────────────────────

def _rubric_prompt(problem: Problem, response: str, n_criteria: int | None = None) -> str:
    if n_criteria is not None:
        criteria_instruction = (
            f"The rubric contains exactly {n_criteria} top-level criteria. "
            f"You MUST return exactly {n_criteria} scores in the criteria_scores array — "
            f"one score (0, 1, or 2) per numbered criterion. Do NOT split sub-bullets into separate scores."
        )
    else:
        criteria_instruction = (
            "Count the number of top-level numbered criteria in the rubric (e.g., items starting with '1.', '2.', etc.). "
            "Return exactly that many scores. Do NOT split sub-bullets into separate scores."
        )

    return f"""You are a strict evaluator for business reasoning tasks.

## Scenario
{problem.scenario}

## Question
{problem.question}

## Reference Answer
{problem.correct_answer}

## Evaluation Rubric
{problem.rubric}

## Candidate Response
{response}

---
Score each top-level criterion from the rubric on a scale of 0-2:
  0 = not addressed, incorrect, or wrong number/amount
  1 = correct reasoning approach but minor error in application
  2 = fully and correctly addressed with accurate numbers

CRITICAL SCORING RULES:
- For any criterion involving a specific dollar amount or numerical calculation,
  score 0 if the candidate's final number differs from the reference by more
  than 10%. Do NOT award partial credit for correct methodology with wrong results.
- For coverage determination criteria, score 0 if the candidate reaches the
  opposite conclusion from the reference (e.g., says "covered" when reference
  says "excluded", or vice versa).
- A score of 1 is reserved for cases where the reasoning approach is correct
  but a minor detail (not the primary amount or coverage direction) is off.

{criteria_instruction}
Return ONLY valid JSON in this exact format (no markdown, no prose):
{{"criteria_scores": [<score1>, <score2>, ...], "reasoning": "<brief justification>"}}"""


def _parse_judge_json(text: str) -> dict:
    # Strip markdown code fences if present
    text = _CODE_FENCE_RE.sub("", text).strip().rstrip("`").strip()

    # Strategy 1: greedy match from first { to last }
    match = _JSON_GREEDY_RE.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 2: try each flat {…} block (no nested braces)
    for m in _JSON_FLAT_RE.finditer(text):
        try:
            parsed = json.loads(m.group())
            if "criteria_scores" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No JSON found in judge response: {text[:200]}")


async def _call_judge(prompt: str, judge_model: str) -> str:
    provider = MODELS[judge_model]["provider"]
    for attempt in range(MAX_RETRIES):
        try:
            api_model = MODELS[judge_model].get("api_model", judge_model)
            if provider == "openrouter" and "anthropic" in api_model:
                # Judge uses Anthropic SDK directly for reliability
                import anthropic
                anthropic_model = api_model.replace("anthropic/", "").replace(".", "-")
                client = anthropic.AsyncAnthropic(api_key=API_KEYS["anthropic"])
                msg = await client.messages.create(
                    model=anthropic_model, max_tokens=1024, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            else:
                # Fallback: route through OpenRouter
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    api_key=API_KEYS["openrouter"],
                    base_url=OPENROUTER_BASE_URL,
                )
                resp = await client.chat.completions.create(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    return ""


async def judge_response(
    problem: Problem,
    response: str,
    model: str,
    sample_idx: int,
    semaphore: asyncio.Semaphore,
    judge_model: str,
) -> JudgeResult:
    async with semaphore:
        # ── Binary (Insurance: approve/deny) ──────────────────────────────────
        if problem.type == "binary":
            resp_lower = response.lower()
            correct_lower = problem.correct_answer.lower()
            correct_is_approve = "approve" in correct_lower

            # Fast path: explicit keyword match with no ambiguity
            has_approve = "approve" in resp_lower
            has_deny = "deny" in resp_lower or "denied" in resp_lower or "not covered" in resp_lower
            if has_approve and not has_deny:
                passed = correct_is_approve
                reasoning = "keyword match: approve"
            elif has_deny and not has_approve:
                passed = not correct_is_approve
                reasoning = "keyword match: deny"
            else:
                # Ambiguous or neither keyword — ask the judge to extract the verdict
                extract_prompt = (
                    f"A model was asked to adjudicate an insurance claim.\n\n"
                    f"Scenario: {problem.scenario}\n"
                    f"Question: {problem.question}\n"
                    f"Model response:\n{response}\n\n"
                    "Based solely on the model's response, does it recommend approving or denying the claim? "
                    'Return ONLY valid JSON: {"verdict": "approve"} or {"verdict": "deny"}'
                )
                raw = await _call_judge(extract_prompt, judge_model)
                try:
                    parsed = _parse_judge_json(raw)
                    verdict = parsed.get("verdict", "").lower()
                    if "approve" in verdict:
                        passed = correct_is_approve
                        reasoning = "llm-extracted verdict: approve"
                    elif "deny" in verdict:
                        passed = not correct_is_approve
                        reasoning = "llm-extracted verdict: deny"
                    else:
                        passed = False
                        reasoning = f"could not extract verdict: {raw[:100]}"
                except Exception:
                    passed = False
                    reasoning = f"parse error on verdict extraction"

            return JudgeResult(
                question_id=problem.id,
                dataset=problem.dataset,
                model=model,
                sample_idx=sample_idx,
                criteria_scores=[2 if passed else 0],
                pass_score=1.0 if passed else 0.0,
                passed=passed,
                reasoning=reasoning,
                judgment_type="binary",
                timestamp=time.time(),
            )

        # ── Numeric (Multi-Step: extract and compare number) ──────────────────
        if problem.type == "numeric":
            expected = extract_numeric(problem.correct_answer)
            actual = extract_numeric(response)
            if expected is not None and actual is not None:
                if expected == 0:
                    passed = actual == 0
                else:
                    tolerance = abs(expected) * 0.01  # 1%
                    passed = abs(actual - expected) <= tolerance
            else:
                passed = False
            # Also do rubric scoring for step-by-step quality
            n_criteria = _count_rubric_criteria(problem.rubric)
            prompt = _rubric_prompt(problem, response, n_criteria)
            raw = await _call_judge(prompt, judge_model)
            try:
                parsed = _parse_judge_json(raw)
                scores = [int(s) for s in parsed.get("criteria_scores", [])]
                reasoning = parsed.get("reasoning", "")
            except Exception:
                scores = []
                reasoning = raw[:200]

            if scores:
                rubric_score = sum(scores) / (2 * len(scores))
                # Combine: numeric correctness (50%) + rubric quality (50%)
                pass_score = 0.5 * (1.0 if passed else 0.0) + 0.5 * rubric_score
            else:
                pass_score = 1.0 if passed else 0.0

            return JudgeResult(
                question_id=problem.id,
                dataset=problem.dataset,
                model=model,
                sample_idx=sample_idx,
                criteria_scores=scores,
                pass_score=pass_score,
                passed=pass_score >= PASS_THRESHOLD,
                reasoning=reasoning,
                raw_response=raw,
                judgment_type="numeric",
                timestamp=time.time(),
            )

        # ── Rubric (all other datasets) ────────────────────────────────────────
        n_criteria = _count_rubric_criteria(problem.rubric)
        prompt = _rubric_prompt(problem, response, n_criteria)
        raw = await _call_judge(prompt, judge_model)
        try:
            parsed = _parse_judge_json(raw)
            scores = [int(s) for s in parsed.get("criteria_scores", [])]
            reasoning = parsed.get("reasoning", "")
        except Exception:
            # Retry once with stricter instruction
            retry_prompt = (
                "You MUST return ONLY valid JSON with no other text. "
                "Do not include any reasoning outside the JSON.\n\n" + prompt
            )
            try:
                raw = await _call_judge(retry_prompt, judge_model)
                parsed = _parse_judge_json(raw)
                scores = [int(s) for s in parsed.get("criteria_scores", [])]
                reasoning = parsed.get("reasoning", "")
            except Exception:
                scores = [0]
                reasoning = f"parse error: {raw[:200]}"

        # Validate criteria count against rubric structure
        expected_n = n_criteria or 0
        if expected_n > 0 and scores and len(scores) != expected_n:
            original_len = len(scores)
            if original_len > expected_n:
                scores = list(scores[:expected_n])
            else:
                scores = scores + [0] * (expected_n - original_len)
            reasoning += f" [CRITERIA NORMALIZED: {original_len} -> {expected_n}]"

        pass_score = sum(scores) / (2 * len(scores)) if scores else 0.0

        # ── Hard gates: override rubric score if fundamentals are wrong ───
        # Numerical gate: if correct answer has a dollar amount and response
        # differs by >10%, cap score to force FAIL
        if "$" in problem.correct_answer:
            correct_amt = _extract_dollar(problem.correct_answer)
            response_amt = _extract_headline_dollar(response)
            if correct_amt is not None and response_amt is not None and correct_amt > 0:
                # Before triggering the gate, check if the correct amount
                # appears anywhere in the response (handles multi-payee answers
                # where headline extraction picks a different valid amount).
                correct_amt_str = f"${correct_amt:,.0f}"
                response_normalized = _normalize_latex_dollars(response)
                correct_present_in_response = correct_amt_str in response_normalized

                ratio = response_amt / correct_amt if correct_amt else 0
                # Guard: if extracted amount is >1000x or <0.001x the expected,
                # the regex likely grabbed an incidental figure — skip the gate.
                extraction_plausible = 0.001 <= ratio <= 1000 if ratio > 0 else response_amt == 0
                if extraction_plausible and abs(response_amt - correct_amt) / correct_amt > 0.10:
                    if correct_present_in_response:
                        reasoning += f" [NUM GATE SKIPPED: ${correct_amt:,.0f} found in response despite headline mismatch]"
                    else:
                        pass_score = min(pass_score, 0.5)
                        reasoning += f" [NUM GATE: expected ${correct_amt:,.0f}, got ${response_amt:,.0f}]"
                elif not extraction_plausible:
                    reasoning += f" [NUM GATE SKIPPED: extraction unreliable — expected ${correct_amt:,.0f}, extracted ${response_amt:,.0f}]"

        # Coverage direction gate: if correct answer says excluded/denied
        # but response says covered/approved (or vice versa), cap score
        if not _coverage_direction_matches(problem.correct_answer, response):
            pass_score = min(pass_score, 0.5)
            reasoning += " [DIRECTION GATE: coverage direction mismatch]"

        return JudgeResult(
            question_id=problem.id,
            dataset=problem.dataset,
            model=model,
            sample_idx=sample_idx,
            criteria_scores=scores,
            pass_score=pass_score,
            passed=pass_score >= PASS_THRESHOLD,
            reasoning=reasoning,
            raw_response=raw,
            judgment_type="rubric",
            timestamp=time.time(),
        )


async def judge_dataset(
    model: str,
    problems: list[Problem],
    generation_records: list[dict],
    concurrency: int = 5,
    judge_model: str = JUDGE_MODEL,
    semaphore: asyncio.Semaphore | None = None,
) -> Path:
    dataset_key = problems[0].dataset if problems else "unknown"
    out_path = _judge_path(model, dataset_key)
    done = _already_judged(out_path)

    problem_map = {p.id: p for p in problems}
    effective_semaphore = semaphore if semaphore is not None else asyncio.Semaphore(concurrency)

    async def _process(rec: dict):
        qid = rec["question_id"]
        idx = rec["sample_idx"]
        if (qid, idx) in done:
            return
        problem = problem_map.get(qid)
        if not problem:
            return
        result = await judge_response(problem, rec["response"], model, idx, effective_semaphore, judge_model)
        append(out_path, {
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

    await asyncio.gather(*[_process(r) for r in generation_records])
    return out_path
