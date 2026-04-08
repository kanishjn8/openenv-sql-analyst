"""
tasks/task_medium.py — Medium task: Churn analysis 

Ground truth extracted from the actual database (analyst.db).
"""

from __future__ import annotations

import re
from tasks.grader import BaseTask


class ChurnAnalysisTask(BaseTask):
    """
    Medium task: Identify customers who have placed at least 3 orders
    but have not made any purchase in the last 90 days (relative to 2024-03-31).
    List the top 10 by total lifetime spend.
    
    Ground truth from Phase 3 verification:
      - churned_count: 87 (total churned customers)
      - top_10_ids: [203, 78, 89, 12, 167, 45, 301, 56, 324, 99]
    """

    task_id = "churn_analysis"
    difficulty = "medium"
    question = (
        "Which customers have placed at least 3 orders but have not made "
        "any purchase in the last 90 days (relative to 2024-03-31)? "
        "List the top 10 by total lifetime spend."
    )
    ground_truth = {
        "churned_count": 87,
        "top_10_ids": [203, 78, 89, 12, 167, 45, 301, 56, 324, 99],
    }

    def grade(self, final_answer: str) -> float:
        """Grade the agent's answer.
        
        Scoring rubric:
          +0.40 if at least 7 of the top 10 IDs are present
          +0.30 (instead of 0.40) if all 10 are present AND in the correct order
          +0.20 if at least 5 IDs match (partial credit)
          +0.30 if the agent states the correct total churned count
          
        Cap at 1.0, never crash.
        """
        if not final_answer or not isinstance(final_answer, str):
            return 0.0
        
        score = 0.0
        
        # Extract all integers from the answer
        integers = re.findall(r'\b\d+\b', final_answer)
        
        try:
            integers = [int(x) for x in integers]
        except ValueError:
            integers = []
        
        if not integers:
            return 0.0
        
        expected_ids = self.ground_truth["top_10_ids"]
        expected_count = self.ground_truth["churned_count"]
        
        # Count how many of the top 10 IDs appear in the answer
        found_ids = [uid for uid in expected_ids if uid in integers]
        n_found = len(found_ids)
        
        # Check if all 10 are present AND in the correct order
        if n_found == 10:
            # Check if they appear in the correct order in the answer
            idx_list = [integers.index(uid) for uid in expected_ids]
            if idx_list == sorted(idx_list):
                # All 10 present and in correct order
                score += 0.30
            else:
                # All 10 present but not in order
                score += 0.40
        elif n_found >= 7:
            # At least 7 of the top 10
            score += 0.40
        elif n_found >= 5:
            # At least 5
            score += 0.20
        
        # Check if the total churned count is mentioned
        if expected_count in integers:
            score += 0.30
        
        return min(score, 1.0)


def get_task() -> ChurnAnalysisTask:
    """Helper function for environment.py to load this task."""
    return ChurnAnalysisTask()
