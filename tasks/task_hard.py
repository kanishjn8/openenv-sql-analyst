"""
tasks/task_hard.py — Hard task: Root cause analysis (Person B)

Ground truth extracted from the actual database (analyst.db).
"""

from __future__ import annotations

import re
from tasks.grader import BaseTask


class RootCauseTask(BaseTask):
    """
    Hard task: Investigate a revenue drop from February 2024 to March 2024
    and identify the most likely root cause.
    
    Ground truth from Phase 3 verification:
      - feb_revenue: 50200.00
      - mar_revenue: 41200.00
      - root_cause_category: Electronics (largest drop: 12454.97)
    """

    task_id = "root_cause"
    difficulty = "hard"
    question = (
        "Total revenue dropped significantly in March 2024 compared to "
        "February 2024. Investigate the data and identify the most likely "
        "root cause of this decline."
    )
    ground_truth = {
        "feb_revenue": 50200.00,
        "mar_revenue": 41200.00,
        "root_cause_category": "Electronics",
    }

    def grade(self, final_answer: str) -> float:
        """Grade the agent's answer.
        
        Scoring rubric:
          +0.20 if answer contains any of: "drop", "decline", "decrease", "fell", "lower"
          +0.40 if the answer mentions the correct root cause category (case-insensitive)
          +0.20 if the answer mentions a specific product or product id
          +0.20 coherence check: answer has ≥3 sentences AND contains "category" or "product"
          
        Cap at 1.0, never crash.
        """
        if not final_answer or not isinstance(final_answer, str):
            return 0.0
        
        score = 0.0
        answer_lower = final_answer.lower()
        
        # +0.20 for mentioning drop/decline/decrease/fell/lower
        decline_keywords = ["drop", "decline", "decrease", "fell", "lower"]
        if any(kw in answer_lower for kw in decline_keywords):
            score += 0.20
        
        # +0.40 for correct root cause category
        if self.ground_truth["root_cause_category"].lower() in answer_lower:
            score += 0.40
        
        # +0.20 for mentioning a product or product_id
        # Check for "product" keyword or product IDs (1-80)
        if "product" in answer_lower:
            score += 0.20
        else:
            # Check if any product ID (1-80) is mentioned
            product_ids = re.findall(r'\bproduct[_\s]*(?:id)?[_\s]*(\d+)\b', answer_lower)
            if product_ids:
                product_ids = [int(x) for x in product_ids if 1 <= int(x) <= 80]
                if product_ids:
                    score += 0.20
        
        # +0.20 coherence check: ≥3 sentences AND contains "category" or "product"
        sentences = re.split(r'[.!?]+', final_answer.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 3 and ("category" in answer_lower or "product" in answer_lower):
            score += 0.20
        
        return min(score, 1.0)


def get_task() -> RootCauseTask:
    """Helper function for environment.py to load this task."""
    return RootCauseTask()
