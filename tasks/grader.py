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
