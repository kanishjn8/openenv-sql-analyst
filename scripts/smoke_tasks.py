"""scripts/smoke_tasks.py

Quick in-process smoke tests for the environment.

Runs short episodes for multiple tasks to validate:
- reset() works
- step(query) returns results and a reward
- step(answer) terminates the episode and returns a final reward

This avoids needing to run the HTTP server.
"""

from __future__ import annotations

from env.environment import SQLAnalystEnv
from env.models import Action


def run_task(env: SQLAnalystEnv, task_id: str, query_sql: str, final_answer: str) -> None:
    print(f"\n=== TASK: {task_id} ===")
    obs = env.reset(task_id)
    print("Q:", obs.question)

    obs, r1, done, _info = env.step(Action(action_type="query", sql=query_sql))
    print("query reward:", r1.score, "done:", done)
    print("last_result.success:", obs.last_result.success if obs.last_result else None)

    obs, r2, done, _info = env.step(Action(action_type="answer", final_answer=final_answer))
    print("answer reward:", r2.score, "answer_correct:", r2.answer_correct, "done:", done)
    print("reason:", r2.reason)


def main() -> None:
    env = SQLAnalystEnv(db_path="data/analyst.db", max_steps=10)

    run_task(
        env,
        task_id="churn_analysis",
        query_sql=(
            "SELECT customer_id, COUNT(*) AS n_orders "
            "FROM orders GROUP BY customer_id ORDER BY n_orders DESC LIMIT 5;"
        ),
        final_answer=(
            "Top churned IDs: 203, 78, 89, 12, 167, 45, 301, 56, 324, 99. "
            "Total churned count: 87."
        ),
    )

    run_task(
        env,
        task_id="root_cause",
        query_sql=(
            "SELECT strftime('%Y-%m', order_date) AS month, "
            "SUM(total_amount) AS revenue "
            "FROM orders WHERE status='completed' "
            "GROUP BY 1 ORDER BY 1 DESC LIMIT 3;"
        ),
        final_answer=(
            "Revenue fell from February to March. The biggest decline appears in the "
            "Electronics category, likely related to a product issue (out of stock). "
            "This category drop explains most of the decrease."
        ),
    )


if __name__ == "__main__":
    main()
