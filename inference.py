#!/usr/bin/env python
"""inference.py

Competition inference entrypoint.

Contract (per prompt):
- Must be named `inference.py` at repo root.
- Must use OpenAI client for all LLM calls.
- Must read credentials/config from environment variables:
    HF_TOKEN (or API_KEY)  -> API key
    API_BASE_URL           -> OpenAI-compatible endpoint (default: HF router)
    MODEL_NAME             -> model identifier
    LOCAL_IMAGE_NAME       -> declared for compatibility with submission template

STDOUT contract:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Notes:
- This script runs one episode for TASK_NAME.
- The environment here is the in-process SQL analyst env (not docker).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import OpenAI

from env.environment import SQLAnalystEnv
from env.models import Action


BENCHMARK = os.getenv("BENCHMARK", "openenv-sql-analyst")
TASK_NAME = os.getenv("TASK_NAME", "sales_summary")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.10"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Contract: all fields must be on a single line (no embedded newlines).
    action = re.sub(r"\s+", " ", action).strip()
    error_val = re.sub(r"\s+", " ", error).strip() if error else "null"
    done_val = str(done).lower()
    # reward must be formatted to 2 decimals
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Example in prompt uses 2 decimals for score; keep 2 decimals for strictness.
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = (
    "You are a data analyst agent. You can query a SQLite database via the environment. "
    "You must respond with exactly one of these blocks:\n\n"
    "ACTION: query\n"
    "SQL: <single SELECT query>\n\n"
    "OR\n\n"
    "ACTION: answer\n"
    "ANSWER: <final answer text>\n"
)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:sql|SQL|json|JSON|text|TEXT)?\n", "", text)
    return text.replace("```", "")


def _extract_sql(text: str) -> Optional[str]:
    m = re.search(r"SQL:\s*(.+?)(?:\n\s*(?:ACTION:|ANSWER:)\b|$)", text, flags=re.I | re.S)
    if not m:
        return None
    sql = m.group(1).strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip() + ";"
    return sql or None


def _extract_answer(text: str) -> Optional[str]:
    m = re.search(r"ANSWER:\s*(.+?)(?:\n\s*ACTION:\b|$)", text, flags=re.I | re.S)
    if not m:
        return None
    ans = m.group(1).strip()
    return ans or None


def parse_action(response: str) -> Tuple[Optional[str], Optional[str]]:
    if not response:
        return None, None

    text = _strip_code_fences(response).strip()

    action_match = re.search(r"ACTION:\s*(query|answer)\b", text, flags=re.I)
    if action_match:
        if action_match.group(1).lower() == "query":
            return _extract_sql(text), None
        return None, _extract_answer(text)

    # Fallbacks: try to extract either block
    sql = _extract_sql(text)
    if sql:
        return sql, None
    ans = _extract_answer(text)
    if ans:
        return None, ans

    # Last resort: if it looks like SQL
    if re.search(r"\bselect\b", text, flags=re.I):
        candidate = text.split(";", 1)[0].strip()
        if candidate:
            return candidate + ";", None

    return None, None


@dataclass
class StepResult:
    action_str: str
    reward: float
    done: bool
    error: Optional[str]


def llm_next(client: OpenAI, model: str, user_message: str) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=400,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "Missing API key. Set HF_TOKEN (preferred), API_KEY, or OPENAI_API_KEY in the environment."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # LOCAL_IMAGE_NAME is intentionally declared for template compatibility.
    _ = LOCAL_IMAGE_NAME
    env = SQLAnalystEnv(db_path="data/analyst.db", max_steps=MAX_STEPS)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # Always emit [END], even on exception.
    try:
        obs = env.reset(TASK_NAME)

        user_message = (
            "Schema:\n"
            f"{obs.schema_description}\n\n"
            f"Question:\n{obs.question}\n\n"
            "Remember: respond with exactly one ACTION block."
        )

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            try:
                response = llm_next(client, MODEL_NAME, user_message)
                sql, answer = parse_action(response)

                if sql:
                    action = Action(action_type="query", sql=sql)
                    obs, reward, done, info = env.step(action)
                    steps_taken = step
                    rewards.append(float(reward.score))
                    # Action string should be single-line.
                    action_str = re.sub(r"\s+", " ", f"query:{sql}").strip()
                    log_step(step=step, action=action_str, reward=float(reward.score), done=done, error=None)

                    # Short result summary for next prompt.
                    if obs.last_result and obs.last_result.success:
                        preview = obs.last_result.rows[:2] if obs.last_result.rows else []
                        result_summary = f"rows={obs.last_result.row_count} preview={preview}"
                    elif obs.last_result:
                        result_summary = f"query_error={obs.last_result.error}"
                    else:
                        result_summary = "no_result"

                    user_message = (
                        f"Question:\n{obs.question}\n\n"
                        f"Last result: {result_summary}\n\n"
                        "Next action?"
                    )
                    continue

                if answer:
                    action = Action(action_type="answer", final_answer=answer)
                    obs, reward, done, info = env.step(action)
                    steps_taken = step
                    rewards.append(float(reward.score))
                    action_str = re.sub(r"\s+", " ", f"answer:{answer}").strip()
                    log_step(step=step, action=action_str, reward=float(reward.score), done=done, error=None)
                    break

                # If parse failed, count as a step with 0 reward but keep going.
                steps_taken = step
                rewards.append(0.0)
                log_step(step=step, action="parse_error", reward=0.0, done=False, error=None)
                user_message = (
                    f"Question:\n{obs.question}\n\n"
                    "Your last response did not match the required format. "
                    "Respond with exactly one ACTION block."
                )

            except Exception as exc:
                # Log as a step error and continue (unless max steps).
                steps_taken = step
                rewards.append(0.0)
                log_step(step=step, action="exception", reward=0.0, done=False, error=str(exc).replace("\n", " "))

        score = float(rewards[-1]) if rewards else 0.0
        # Clamp to [0,1] per requirement.
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
