"""
tasks/grader.py — Base task / grader interface (Person B owns task subclasses)

Person A provides this abstract base class so that the environment and
reward modules can depend on a stable contract without importing concrete
task implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTask(ABC):
    """Abstract base class that every task must subclass.

    Attributes
    ----------
    task_id : str
        Unique string identifier (e.g. ``"sales_summary"``).
    difficulty : str
        One of ``"easy"``, ``"medium"``, ``"hard"``.
    question : str
        The business question presented to the agent.
    ground_truth : dict
        Pre-computed expected values used for grading.
    """

    task_id: str
    difficulty: str
    question: str
    ground_truth: dict

    @abstractmethod
    def grade(self, final_answer: str) -> float:
        """Grade the agent's final answer.

        Parameters
        ----------
        final_answer : str
            Free-form text submitted by the agent.

        Returns
        -------
        float
            A score in ``[0.0, 1.0]``.
        """
        ...

    def get_question(self) -> str:
        """Return the business question for this task."""
        return self.question


# ---------------------------------------------------------------------------
# Task registry and router
# ---------------------------------------------------------------------------

def get_task_by_id(task_id: str) -> BaseTask:
    """Load and return the correct task instance by task_id.
    
    Parameters
    ----------
    task_id : str
        One of: "sales_summary", "churn_analysis", "root_cause"
    
    Returns
    -------
    BaseTask
        An instance of the corresponding task class.
    
    Raises
    ------
    ValueError
        If task_id is not recognised.
    """
    if task_id == "sales_summary":
        from tasks.task_easy import SalesSummaryTask
        return SalesSummaryTask()
    elif task_id == "churn_analysis":
        from tasks.task_medium import ChurnAnalysisTask
        return ChurnAnalysisTask()
    elif task_id == "root_cause":
        from tasks.task_hard import RootCauseTask
        return RootCauseTask()
    else:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available: sales_summary, churn_analysis, root_cause"
        )
