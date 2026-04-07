"""
env/environment.py — Core RL-style environment (Person A)

Implements the standard reset / step / state interface for the OpenEnv
SQL Analyst environment.  The environment is **synchronous** (no async)
and NOT thread-safe (acceptable for hackathon scope).
"""

from __future__ import annotations

import importlib
from typing import Any

from env.database import DatabaseRunner
from env.models import Action, EpisodeState, Observation, QueryResult, Reward
from env.reward import compute_step_reward
from tasks.grader import BaseTask


# ---------------------------------------------------------------------------
# Task registry — maps task_id → module path + class name
# New tasks can be registered here by Person B.
# ---------------------------------------------------------------------------
_TASK_REGISTRY: dict[str, tuple[str, str]] = {
    "sales_summary": ("tasks.task_easy", "SalesSummaryTask"),
    "churn_analysis": ("tasks.task_medium", "ChurnAnalysisTask"),
    "root_cause":     ("tasks.task_hard", "RootCauseTask"),
}


def _load_task(task_id: str) -> BaseTask:
    """Dynamically import and instantiate a task by its *task_id*.

    Raises
    ------
    ValueError
        If *task_id* is not found in the registry.
    """
    if task_id not in _TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(_TASK_REGISTRY.keys())}"
        )
    module_path, class_name = _TASK_REGISTRY[task_id]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


class SQLAnalystEnv:
    """OpenEnv-compliant environment for the SQL Analyst benchmark.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (``data/analyst.db``).
    max_steps : int
        Maximum number of actions the agent can take per episode.
    """

    def __init__(self, db_path: str, max_steps: int = 10) -> None:
        self._db = DatabaseRunner(db_path)
        self._max_steps = max_steps

        # Episode state (initialised on reset)
        self._task: BaseTask | None = None
        self._task_id: str = ""
        self._steps_taken: int = 0
        self._cumulative_score: float = 0.0
        self._cumulative_partial: float = 0.0
        self._query_history: list[QueryResult] = []
        self._done: bool = True  # True until reset() is called

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """Start a new episode for the given *task_id*.

        1. Load the task definition from the registry.
        2. Clear all episode state.
        3. Return the initial Observation.
        """
        self._task = _load_task(task_id)
        self._task_id = task_id
        self._steps_taken = 0
        self._cumulative_score = 0.0
        self._cumulative_partial = 0.0
        self._query_history = []
        self._done = False

        return Observation(
            schema_description=self._db.get_schema_description(),
            question=self._task.get_question(),
            query_history=[],
            last_result=None,
            steps_taken=0,
            max_steps=self._max_steps,
            task_id=task_id,
            done=False,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Execute one agent action and return (observation, reward, done, info).

        Raises
        ------
        RuntimeError
            If step() is called before reset() or after the episode has ended.
        """
        if self._task is None or self._done:
            raise RuntimeError(
                "Cannot call step() before reset() or after the episode has ended."
            )

        info: dict[str, Any] = {}

        # --- Check step limit ------------------------------------------------
        if self._steps_taken >= self._max_steps:
            self._done = True
            reward = Reward(
                score=0.0,
                sql_valid=False,
                result_shape_correct=False,
                answer_correct=False,
                partial_credit=0.0,
                penalty=0.0,
                reason="Step limit reached — episode terminated.",
            )
            obs = self._build_observation()
            return obs, reward, True, info

        self._steps_taken += 1

        # --- Handle "query" action -------------------------------------------
        if action.action_type == "query":
            assert action.sql is not None  # guaranteed by Action validator

            # Safety check
            if not self._db.is_safe_query(action.sql):
                query_result = QueryResult(
                    sql=action.sql,
                    success=False,
                    rows=[],
                    error="Blocked: destructive SQL detected.",
                    row_count=0,
                )
                reward = Reward(
                    score=0.0,
                    sql_valid=False,
                    result_shape_correct=False,
                    answer_correct=False,
                    partial_credit=0.0,
                    penalty=0.10,
                    reason="Destructive SQL blocked; -0.10 penalty.",
                )
                self._query_history.append(query_result)
                obs = self._build_observation()
                info["blocked"] = True
                return obs, reward, False, info

            # Execute the query
            query_result = self._db.run_query(action.sql)
            self._query_history.append(query_result)

            # Compute step reward
            reward = compute_step_reward(
                action=action,
                query_result=query_result,
                task=self._task,
                steps_taken=self._steps_taken,
                max_steps=self._max_steps,
                query_history=self._query_history[:-1],  # exclude current
                cumulative_partial=self._cumulative_partial,
            )
            self._cumulative_partial += reward.partial_credit
            self._cumulative_score += reward.score

            obs = self._build_observation()
            return obs, reward, False, info

        # --- Handle "answer" action ------------------------------------------
        assert action.action_type == "answer"

        reward = compute_step_reward(
            action=action,
            query_result=None,
            task=self._task,
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            query_history=self._query_history,
            cumulative_partial=self._cumulative_partial,
        )
        self._cumulative_score += reward.score
        self._done = True

        obs = self._build_observation()
        info["final_score"] = reward.score
        return obs, reward, True, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> EpisodeState:
        """Return a read-only snapshot of the current episode.  No side effects."""
        return EpisodeState(
            task_id=self._task_id,
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            cumulative_score=self._cumulative_score,
            done=self._done,
            query_history=list(self._query_history),
            current_question=self._task.get_question() if self._task else "",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct the current Observation from internal state."""
        assert self._task is not None
        return Observation(
            schema_description=self._db.get_schema_description(),
            question=self._task.get_question(),
            query_history=list(self._query_history),
            last_result=self._query_history[-1] if self._query_history else None,
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            task_id=self._task_id,
            done=self._done,
        )
