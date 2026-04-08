"""
env/reward.py — Reward function 

Computes per-step and final rewards according to the scoring rubric.

Scoring summary
---------------
Query action:
  +0.10  if SQL is valid (runs without error)
  +0.05  if result has ≥1 row
  −0.05  if the query is a duplicate of a previous query
  −0.10  if steps_taken > 70 % of max_steps (approaching-limit penalty)

Answer action:
  Grader score (0.0–1.0) from the task's grade() method
  −0.05  per query beyond the first 5 (efficiency penalty)
  +partial_query_credit accumulated during the episode (capped at 0.20)

Final score = clamp(grader_score − efficiency_penalty + partial_credit, 0, 1)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from env.models import Action, QueryResult, Reward

if TYPE_CHECKING:
    # Avoid circular / runtime import — BaseTask lives in tasks/
    from tasks.grader import BaseTask


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))


def compute_step_reward(
    action: Action,
    query_result: Optional[QueryResult],
    task: "BaseTask",
    steps_taken: int,
    max_steps: int,
    query_history: list[QueryResult] | None = None,
    cumulative_partial: float = 0.0,
) -> Reward:
    """Compute the reward for a single environment step.

    Parameters
    ----------
    action : Action
        The action the agent just took.
    query_result : QueryResult | None
        Result of the SQL execution (None for answer actions).
    task : BaseTask
        The active task, needed for grading final answers.
    steps_taken : int
        Number of steps taken *including* the current one.
    max_steps : int
        Hard step limit for the episode.
    query_history : list[QueryResult] | None
        All previous query results (used for duplicate detection).
    cumulative_partial : float
        Partial credit accumulated so far from previous query steps.

    Returns
    -------
    Reward
        A fully populated Reward object.  This function **never raises**.
    """
    try:
        if action.action_type == "query":
            return _score_query(
                action, query_result, steps_taken, max_steps, query_history or []
            )
        else:
            return _score_answer(
                action, task, query_history or [], cumulative_partial
            )
    except Exception as exc:  # noqa: BLE001
        # The reward function must *never* crash — return a zero-score
        # reward with a diagnostic message if anything unexpected happens.
        return Reward(
            score=0.0,
            sql_valid=False,
            result_shape_correct=False,
            answer_correct=False,
            partial_credit=0.0,
            penalty=0.0,
            reason=f"Reward computation error: {exc}",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_query(
    action: Action,
    query_result: Optional[QueryResult],
    steps_taken: int,
    max_steps: int,
    history: list[QueryResult],
) -> Reward:
    """Score a ``query`` action."""
    partial = 0.0
    penalty = 0.0
    reasons: list[str] = []

    sql_valid = False
    shape_ok = False

    if query_result is not None and query_result.success:
        # +0.10 for a valid SQL query
        sql_valid = True
        partial += 0.10
        reasons.append("+0.10 valid SQL")

        # +0.05 if at least one row returned
        if query_result.row_count > 0:
            shape_ok = True
            partial += 0.05
            reasons.append("+0.05 non-empty result")

    # −0.05 if this exact SQL was already executed
    if query_result is not None:
        prev_sqls = {qr.sql.strip().lower() for qr in history}
        if query_result.sql.strip().lower() in prev_sqls:
            penalty += 0.05
            reasons.append("-0.05 duplicate query")

    # −0.10 if approaching the step limit (>70 %)
    if steps_taken > max_steps * 0.7:
        penalty += 0.10
        reasons.append("-0.10 approaching step limit")

    score = _clamp(partial - penalty)

    return Reward(
        score=score,
        sql_valid=sql_valid,
        result_shape_correct=shape_ok,
        answer_correct=False,  # not applicable to query actions
        partial_credit=partial,
        penalty=penalty,
        reason="; ".join(reasons) if reasons else "no credit",
    )


def _score_answer(
    action: Action,
    task: "BaseTask",
    history: list[QueryResult],
    cumulative_partial: float,
) -> Reward:
    """Score an ``answer`` action."""
    reasons: list[str] = []

    # Delegate to the task grader
    grader_score = task.grade(action.final_answer or "")
    reasons.append(f"grader={grader_score:.2f}")

    answer_correct = grader_score >= 0.99  # consider 1.0 as fully correct

    # Efficiency penalty: −0.05 per query beyond 5
    n_queries = len(history)
    eff_penalty = max(0.0, (n_queries - 5) * 0.05)
    if eff_penalty > 0:
        reasons.append(f"-{eff_penalty:.2f} efficiency ({n_queries} queries)")

    # Cap the accumulated partial credit at 0.20
    partial = min(cumulative_partial, 0.20)
    if partial > 0:
        reasons.append(f"+{partial:.2f} partial credit (capped)")

    score = _clamp(grader_score - eff_penalty + partial)
    reasons.append(f"final={score:.2f}")

    return Reward(
        score=score,
        sql_valid=True,  # N/A for answer, default True
        result_shape_correct=True,  # N/A
        answer_correct=answer_correct,
        partial_credit=partial,
        penalty=eff_penalty,
        reason="; ".join(reasons),
    )
