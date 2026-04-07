"""
tasks/task_easy.py — Easy task: Q4 2023 sales summary (Person B)

Ground truth extracted from the actual database (analyst.db).
"""

from __future__ import annotations

import re
from tasks.grader import BaseTask


class SalesSummaryTask(BaseTask):
    """
    Easy task: Compute Q4 2023 completed-order revenue by region
    and identify the region with the highest sales.
    
    Ground truth from Phase 3 verification:
      - total_revenue: 1534367.72
      - top_region: East
    """

    task_id = "sales_summary"
    difficulty = "easy"
    question = (
        "What was the total revenue from completed orders in Q4 2023 "
        "(October through December), and which region generated the "
        "highest sales during that period?"
    )
    ground_truth = {
        "total_revenue": 1534367.72,
        "top_region": "East",
    }

    def grade(self, final_answer: str) -> float:
        """Grade the agent's answer.
        
        Scoring rubric:
          +0.50 if the answer contains the correct top region (case-insensitive)
          +0.50 if a number in the answer is within 1% of total_revenue
          +0.25 (instead of 0.50) if the number is within 10% but not 1%
          
        Total: up to 1.0, never crashes on malformed input.
        """
        if not final_answer or not isinstance(final_answer, str):
            return 0.0
        
        score = 0.0
        
        # Check for correct region (case-insensitive)
        if self.ground_truth["top_region"].lower() in final_answer.lower():
            score += 0.50
        
        # Extract all numbers from the answer
        numbers = re.findall(r'\d+(?:\.\d+)?', final_answer)
        
        if numbers:
            # Convert to floats
            try:
                numbers_float = [float(n) for n in numbers]
            except ValueError:
                numbers_float = []
            
            # Find best match for total_revenue
            target = self.ground_truth["total_revenue"]
            tolerance_1pct = target * 0.01  # 1% margin
            tolerance_10pct = target * 0.10  # 10% margin
            
            # Check if any number is within 1%
            for num in numbers_float:
                if abs(num - target) <= tolerance_1pct:
                    score += 0.50
                    break
            else:
                # No exact match; check if any is within 10%
                for num in numbers_float:
                    if abs(num - target) <= tolerance_10pct:
                        score += 0.25
                        break
        
        return min(score, 1.0)


def get_task() -> SalesSummaryTask:
    """Helper function for environment.py to load this task."""
    return SalesSummaryTask()
