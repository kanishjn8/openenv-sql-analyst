"""
env/models.py — Pydantic v2 data models (Person A)

Defines the contracts between the environment and any agent or evaluation
framework.  Every piece of data that crosses the environment boundary is
one of the models declared here.
"""

from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# QueryResult — output of a single SQL query execution
# ---------------------------------------------------------------------------
class QueryResult(BaseModel):
    """Represents the outcome of executing a single SQL query."""

    sql: str                    # The SQL string that was executed
    success: bool               # True if the query ran without error
    rows: list[dict]            # Rows returned (empty list on error / DML)
    error: Optional[str] = None # Error message if success is False
    row_count: int              # Number of rows returned


# ---------------------------------------------------------------------------
# Observation — what the agent sees at every time-step
# ---------------------------------------------------------------------------
class Observation(BaseModel):
    """Full observation presented to the agent after each action."""

    schema_description: str              # Formatted schema string
    question: str                        # The business question to answer
    query_history: list[QueryResult]     # All queries executed so far
    last_result: Optional[QueryResult] = None  # Most recent query output
    steps_taken: int                     # Number of actions taken so far
    max_steps: int                       # Hard action limit (default 10)
    task_id: str                         # Identifier for the current task
    done: bool                           # Whether the episode has ended


# ---------------------------------------------------------------------------
# Action — what the agent sends to the environment
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """An action the agent submits to the environment.

    Exactly one of `sql` or `final_answer` must be provided depending on
    the `action_type`.
    """

    action_type: Literal["query", "answer"]
    sql: Optional[str] = None            # Required when action_type == "query"
    final_answer: Optional[str] = None   # Required when action_type == "answer"

    # --- validation --------------------------------------------------------
    @model_validator(mode="after")
    def _check_fields(self) -> "Action":
        """Ensure the correct field is populated for each action type."""
        if self.action_type == "query" and not self.sql:
            raise ValueError("'sql' must be provided when action_type is 'query'")
        if self.action_type == "answer" and not self.final_answer:
            raise ValueError(
                "'final_answer' must be provided when action_type is 'answer'"
            )
        return self


# ---------------------------------------------------------------------------
# Reward — feedback returned for every step
# ---------------------------------------------------------------------------
class Reward(BaseModel):
    """Feedback signal returned after each environment step."""

    score: float                   # Final score, clamped to [0.0, 1.0]
    sql_valid: bool                # Did the SQL execute without error?
    result_shape_correct: bool     # Did the result have expected structure?
    answer_correct: bool           # Did the final answer match ground truth?
    partial_credit: float          # Incremental credit earned this step
    penalty: float                 # Any penalty applied this step
    reason: str                    # Human-readable explanation


# ---------------------------------------------------------------------------
# EpisodeState — snapshot of current episode (no side effects)
# ---------------------------------------------------------------------------
class EpisodeState(BaseModel):
    """Serialisable snapshot of the running episode."""

    task_id: str
    steps_taken: int
    max_steps: int
    cumulative_score: float
    done: bool
    query_history: list[QueryResult]
    current_question: str
