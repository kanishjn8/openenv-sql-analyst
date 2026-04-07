"""
baseline/run_baseline.py — LLM agent baseline (Person B)

Runs an agent loop that solves all three tasks by:
1. Querying the database with SQL
2. Submitting final answers when ready

The LLM is selected based on available API keys:
  - ANTHROPIC_API_KEY → claude-3-5-haiku-20241022
  - OPENAI_API_KEY → gpt-4o-mini
"""

from __future__ import annotations

import os
import random
import re
import sys
from typing import Optional

from env.environment import SQLAnalystEnv
from env.models import Action

# Set seed for reproducibility
random.seed(42)

# ---------------------------------------------------------------------------
# API Detection and Client Setup
# ---------------------------------------------------------------------------

def get_api_client() -> tuple[str, object]:
    """
    Detect and return the LLM client and model name.
    
    Returns
    -------
    tuple[str, object]
        (model_name, client_instance)
    
    Raises
    ------
    RuntimeError
        If neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set.
    """
    if "ANTHROPIC_API_KEY" in os.environ:
        try:
            import anthropic
            key = os.environ["ANTHROPIC_API_KEY"]
            client = anthropic.Anthropic(api_key=key)
            return "claude-3-5-haiku-20241022", client
        except ImportError:
            print("❌ ANTHROPIC_API_KEY found but anthropic package not installed.")
            print("   Run: pip install anthropic")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialise Anthropic client: {e}")
            sys.exit(1)
    
    elif "OPENAI_API_KEY" in os.environ:
        try:
            import openai
            key = os.environ["OPENAI_API_KEY"]
            client = openai.OpenAI(api_key=key)
            return "gpt-4o-mini", client
        except ImportError:
            print("❌ OPENAI_API_KEY found but openai package not installed.")
            print("   Run: pip install openai")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to initialise OpenAI client: {e}")
            sys.exit(1)
    
    else:
        print("❌ No API key found.")
        print("")
        print("To run the baseline, set one of:")
        print("  export ANTHROPIC_API_KEY='sk-...'")
        print("  export OPENAI_API_KEY='sk-...'")
        print("")
        sys.exit(1)


# ---------------------------------------------------------------------------
# LLM Call Wrapper
# ---------------------------------------------------------------------------

def call_llm(
    client: object,
    model: str,
    system_prompt: str,
    user_message: str,
) -> Optional[str]:
    """
    Call the LLM with the given prompts.
    
    Returns
    -------
    Optional[str]
        The LLM's response text, or None if the call fails.
    """
    try:
        if "claude" in model:
            # Anthropic API
            response = client.messages.create(
                model=model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        else:
            # OpenAI API
            response = client.chat.completions.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️  LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# SQL Parsing
# ---------------------------------------------------------------------------

def extract_sql_from_response(response: str) -> Optional[str]:
    """Extract SQL from ACTION: query blocks."""
    # Match: SQL: ... followed by newline or end
    match = re.search(r"SQL:\s*(.+?)(\n|$)", response, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        # Remove trailing markers or partial text
        sql = re.split(r'(?:ACTION:|ANSWER:)', sql, maxsplit=1)[0].strip()
        return sql if sql else None
    return None


def extract_answer_from_response(response: str) -> Optional[str]:
    """Extract final answer from ACTION: answer blocks."""
    # Match: ANSWER: ... followed by newline or end
    match = re.search(r"ANSWER:\s*(.+?)(\n|$)", response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Remove trailing markers
        answer = re.split(r'ACTION:', answer, maxsplit=1)[0].strip()
        return answer if answer else None
    return None


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    env: SQLAnalystEnv,
    client: object,
    model: str,
    system_prompt: str,
) -> float:
    """
    Run a single task to completion.
    
    Returns
    -------
    float
        The final reward score for this task.
    """
    # Reset environment for this task
    obs = env.reset(task_id)
    
    # Build initial user message
    user_message = f"{obs.schema_description}\n\n{obs.question}"
    
    conversation_history = []
    final_score = 0.0
    
    while not obs.done:
        # Call LLM
        response = call_llm(client, model, system_prompt, user_message)
        
        if response is None:
            print(f"  ⚠️  LLM call failed; trying raw answer")
            response = "I don't have an answer."
        
        conversation_history.append(response)
        
        # Try to extract SQL query
        sql = extract_sql_from_response(response)
        if sql:
            # Execute query
            action = Action(action_type="query", sql=sql)
            try:
                obs, reward, done, info = env.step(action)
                if reward.sql_valid:
                    print(f"    ✓ Query executed (rows: {reward.result_shape_correct})")
                else:
                    print(f"    ✗ Query failed: {reward.reason}")
                
                # Update user message with result
                if obs.last_result:
                    result_summary = f"Query returned {obs.last_result.row_count} rows."
                    if obs.last_result.rows:
                        result_summary += f"\nFirst few rows: {obs.last_result.rows[:2]}"
                    user_message = result_summary
                
                final_score = reward.score
                continue
            except Exception as e:
                print(f"    ✗ Step failed: {e}")
                break
        
        # Try to extract final answer
        answer = extract_answer_from_response(response)
        if answer:
            # Submit answer
            action = Action(action_type="answer", final_answer=answer)
            try:
                obs, reward, done, info = env.step(action)
                print(f"    ✓ Answer submitted: score={reward.score:.2f}")
                final_score = reward.score
                break
            except Exception as e:
                print(f"    ✗ Step failed: {e}")
                break
        
        # If neither query nor answer detected, prompt again
        if obs.steps_taken < obs.max_steps:
            print(f"    ⚠️  No SQL or ANSWER block detected; retrying...")
            user_message = (
                f"Please provide either a SQL query or a final answer. "
                f"Current step: {obs.steps_taken}/{obs.max_steps}.\n"
                f"Question: {obs.question}"
            )
        else:
            print(f"    ✗ Max steps reached without answer")
            break
    
    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the baseline agent on all three tasks."""
    print("\n" + "=" * 70)
    print("OpenEnv SQL Analyst Baseline — Agent with LLM Loop")
    print("=" * 70 + "\n")
    
    # Detect API
    try:
        model, client = get_api_client()
        print(f"✅ Using model: {model}\n")
    except SystemExit:
        raise
    
    # Environment setup
    try:
        env = SQLAnalystEnv("data/analyst.db", max_steps=10)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run: python data/seed.py")
        sys.exit(1)
    
    # System prompt
    system_prompt = """You are a data analyst. You have access to a SQLite database.
You will be given a business question. Your job is to write SQL queries
to explore the data and then provide a final answer.

To run a query, respond with:
ACTION: query
SQL: <your sql here>

To submit your final answer, respond with:
ACTION: answer
ANSWER: <your answer here>

Be efficient. You have at most 10 actions. Think before querying."""
    
    # Task list
    tasks = [
        ("sales_summary", "easy"),
        ("churn_analysis", "medium"),
        ("root_cause", "hard"),
    ]
    
    scores = []
    
    for task_id, difficulty in tasks:
        print(f"Running Task ({difficulty}): {task_id}")
        print("-" * 70)
        
        try:
            score = run_task(task_id, env, client, model, system_prompt)
            scores.append(score)
            print()
        except Exception as e:
            print(f"❌ Task failed with error: {e}\n")
            scores.append(0.0)
    
    # Summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for (task_id, difficulty), score in zip(tasks, scores):
        print(f"Task {tasks.index((task_id, difficulty)) + 1} ({difficulty:6s}) - {task_id:17s}: {score:.2f}")
    
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nAverage: {avg:.3f}")
    else:
        print("\nNo scores recorded.")
    
    print()


if __name__ == "__main__":
    main()
