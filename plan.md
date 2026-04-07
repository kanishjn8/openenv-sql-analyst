# OpenEnv SQL Analyst — Full Build Plan

## Project Summary

Build a complete OpenEnv-compliant environment where an AI agent acts as a data analyst.
The agent receives a SQLite database and a business question, writes SQL queries to explore
the data, and returns a final answer. The environment scores the agent on correctness,
query efficiency, and reasoning quality.

This is a real-world task simulation suitable for RL agent evaluation and benchmarking.

---

## Ownership Split

### Person A — Environment Core
Owns: data layer, Pydantic models, environment logic, reward  function, Docker, deployment

### Person B — Tasks, Graders, Baseline, Docs
Owns: task definitions, graders, baseline inference script, openenv.yaml, README

### Critical Coordination Point
Person B cannot finalize graders until Person A has committed the database seed.
Person A must share analyst.db and the exact ground truth query results early.
Person B precomputes expected answers from that database and hardcodes them into graders.

---

## Repository Structure

```
openenv-sql-analyst/
|
|-- env/
|   |-- __init__.py
|   |-- environment.py        # Person A — core environment class
|   |-- models.py             # Person A — Pydantic models
|   |-- database.py           # Person A — SQLite query runner
|   |-- reward.py             # Person A — reward function
|
|-- tasks/
|   |-- __init__.py
|   |-- task_easy.py          # Person B — Task 1: Sales Summary
|   |-- task_medium.py        # Person B — Task 2: Churn Analysis
|   |-- task_hard.py          # Person B — Task 3: Root Cause Investigation
|   |-- grader.py             # Person B — shared grader utilities
|
|-- data/
|   |-- schema.sql            # Person A — table definitions
|   |-- seed.py               # Person A — synthetic data generator
|   |-- analyst.db            # Person A — generated SQLite file (gitignored, built at runtime)
|
|-- baseline/
|   |-- run_baseline.py       # Person B — LLM agent inference script
|
|-- server.py                 # Person A — HTTP wrapper for HF Spaces
|-- openenv.yaml              # Person B — OpenEnv metadata spec
|-- Dockerfile                # Person A — container definition
|-- requirements.txt          # Person A — Python dependencies
|-- README.md                 # Person B — setup and usage docs
```

---

## Data Models (env/models.py) — Person A

All models are Pydantic BaseModel subclasses. These are the contracts between the
environment and any agent or evaluation framework.

### QueryResult

```python
class QueryResult(BaseModel):
    sql: str
    success: bool
    rows: list[dict]
    error: Optional[str]
    row_count: int
```

Represents the output of a single SQL query execution.

### Observation

```python
class Observation(BaseModel):
    schema_description: str    # Full schema as formatted text
    question: str              # The business question to answer
    query_history: list[QueryResult]   # All queries run so far
    last_result: Optional[QueryResult] # Most recent query output
    steps_taken: int           # Number of actions taken
    max_steps: int             # Hard limit (default: 10)
    task_id: str               # Which task is running
    done: bool                 # Whether the episode has ended
```

### Action

```python
class Action(BaseModel):
    action_type: Literal["query", "answer"]
    sql: Optional[str]          # Required if action_type == "query"
    final_answer: Optional[str] # Required if action_type == "answer"
```

Validation rule: if action_type is "query", sql must not be None.
Validation rule: if action_type is "answer", final_answer must not be None.

### Reward

```python
class Reward(BaseModel):
    score: float               # Final score, clamped to [0.0, 1.0]
    sql_valid: bool            # Did the SQL run without error
    result_shape_correct: bool # Did result have expected columns/structure
    answer_correct: bool       # Did final answer match ground truth
    partial_credit: float      # Incremental credit earned this step
    penalty: float             # Any penalty applied this step
    reason: str                # Human-readable explanation
```

### EpisodeState

```python
class EpisodeState(BaseModel):
    task_id: str
    steps_taken: int
    max_steps: int
    cumulative_score: float
    done: bool
    query_history: list[QueryResult]
    current_question: str
```

---

## Database Layer (data/schema.sql + data/seed.py + env/database.py) — Person A

### Schema — data/schema.sql

Four tables simulating a small e-commerce company.

```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    signup_date DATE NOT NULL,
    region TEXT NOT NULL  -- values: North, South, East, West
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,  -- values: Electronics, Apparel, Home, Sports, Beauty
    price REAL NOT NULL
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL  -- values: completed, refunded, pending
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
```

### Seed Data Requirements — data/seed.py

The seed script must produce deterministic data using a fixed random seed (42).
Target row counts:
- customers: 500 rows
- products: 80 rows
- orders: 3000 rows spanning Jan 2023 to Mar 2024
- order_items: ~8000 rows

Planted anomalies for graders (must be exact and documented):

1. Q4 2023 revenue = 187432.50, top region = North (used by Task 1 grader)
2. Churned customers (3+ orders, no order in last 90 days from 2024-03-31):
   exactly 47 customers, top 10 by lifetime value are seeded with fixed IDs
3. March 2024 revenue = 41200.00, February 2024 revenue = 50200.00 (drop = 18.2%)
   Root cause: Electronics category dropped from 22100 to 9800 due to a product
   being marked out of stock (simulate by giving 0 orders to product_id 7 in March)

After generating, save to data/analyst.db using sqlite3.

### Query Runner — env/database.py

```python
class DatabaseRunner:
    def __init__(self, db_path: str)
    def get_schema_description(self) -> str
    def run_query(self, sql: str) -> QueryResult
    def is_safe_query(self, sql: str) -> bool
```

is_safe_query blocks: DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE.
Returns False if any of these keywords appear (case-insensitive) in the SQL.

run_query catches all sqlite3 exceptions and returns them as QueryResult with
success=False and the error message populated.

get_schema_description returns a formatted string listing all tables, columns,
and types. This is what the agent sees in every Observation.

---

## Core Environment (env/environment.py) — Person A

```python
class SQLAnalystEnv:
    def __init__(self, db_path: str, max_steps: int = 10)
    def reset(self, task_id: str) -> Observation
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]
    def state(self) -> EpisodeState
```

### reset(task_id)

1. Load the task definition from the tasks/ directory by task_id
2. Reset query history to empty list
3. Reset steps_taken to 0
4. Reset cumulative_score to 0.0
5. Return an Observation with schema, question, empty history, done=False

### step(action)

1. Validate the action (check action_type and required fields)
2. If steps_taken >= max_steps, return done=True with zero reward
3. If action_type == "query":
   a. Check is_safe_query — if False, return penalty reward and continue
   b. Run the query via DatabaseRunner
   c. Append result to query_history
   d. Compute partial reward from reward.py
   e. Increment steps_taken
   f. Return new Observation, Reward, done=False, info dict
4. If action_type == "answer":
   a. Call the task grader with the final_answer
   b. Compute final reward
   c. Return Observation with done=True, final Reward, done=True, info dict

### state()

Return current EpisodeState snapshot. No side effects.

---

## Reward Function (env/reward.py) — Person A

```python
def compute_step_reward(
    action: Action,
    query_result: Optional[QueryResult],
    task: BaseTask,
    steps_taken: int,
    max_steps: int
) -> Reward
```

### Scoring Logic

For a "query" action:
- +0.10 if SQL is valid and runs without error (sql_valid = True)
- +0.05 if result has at least one row (non-empty result)
- -0.05 if query is a duplicate of a previous query (penalize looping)
- -0.10 if steps_taken > max_steps * 0.7 (approaching limit penalty)

For an "answer" action:
- Delegate to the task grader which returns a score from 0.0 to 1.0
- The grader score becomes the dominant component of the final reward
- -0.05 per query beyond 5 (efficiency penalty, applied to final score)

Final score formula:
```
score = clamp(grader_score - efficiency_penalty + partial_query_credit, 0.0, 1.0)
```

partial_query_credit accumulates across the episode from query steps.
It is capped at 0.20 so it never dominates over answer correctness.

---

## Task Definitions — Person B

### Base Task Class (tasks/grader.py)

```python
from abc import ABC, abstractmethod

class BaseTask(ABC):
    task_id: str
    difficulty: str  # "easy", "medium", "hard"
    question: str
    ground_truth: dict  # precomputed expected values

    @abstractmethod
    def grade(self, final_answer: str) -> float:
        pass

    def get_question(self) -> str:
        return self.question
```

### Task 1 — Easy: Sales Summary (tasks/task_easy.py)

task_id: "sales_summary"
difficulty: "easy"

Question:
"What was the total revenue from completed orders in Q4 2023 (October through December),
and which region generated the highest sales during that period?"

Ground truth (hardcoded from seeded database):
```python
ground_truth = {
    "total_revenue": 187432.50,
    "top_region": "North"
}
```

Grader logic:
1. Parse the agent's final_answer string
2. Extract a number (revenue) and a region name
3. Score:
   - +0.50 if top_region matches exactly (case-insensitive)
   - +0.50 if total_revenue is within 1% of ground truth
   - Partial: +0.25 if revenue is within 10%

The agent can state the answer in natural language. Use regex to extract numbers
and region names. Do not require a specific format.

### Task 2 — Medium: Churn Analysis (tasks/task_medium.py)

task_id: "churn_analysis"
difficulty: "medium"

Question:
"Which customers have placed at least 3 orders but have not made any purchase in the
last 90 days (relative to 2024-03-31)? List the top 10 by total lifetime spend."

Ground truth:
```python
ground_truth = {
    "churned_count": 47,
    "top_10_customer_ids": [12, 45, 78, 203, 167, 89, 301, 56, 144, 290]  # from seed
}
```

Grader logic:
1. Extract customer IDs mentioned in the final_answer
2. Score:
   - +0.40 if at least 7 of the top 10 IDs are present
   - +0.30 if all 10 IDs are present and in correct order
   - +0.30 if agent also states the total count (47) correctly
   - Partial: +0.20 if at least 5 of the top 10 are present

### Task 3 — Hard: Root Cause Investigation (tasks/task_hard.py)

task_id: "root_cause"
difficulty: "hard"

Question:
"Total revenue dropped significantly in March 2024 compared to February 2024.
Investigate the data and identify the most likely root cause of this decline."

Ground truth:
```python
ground_truth = {
    "feb_revenue": 50200.00,
    "mar_revenue": 41200.00,
    "root_cause_category": "Electronics",
    "root_cause_product_id": 7
}
```

Grader logic:
1. Check if the agent confirms a revenue drop exists — +0.20
2. Check if the agent identifies Electronics as the affected category — +0.40
3. Check if the agent mentions a specific product or product_id 7 — +0.20
4. Check if the agent provides a coherent causal explanation — +0.20
   (heuristic: answer is at least 3 sentences and contains "category" or "product")

Score is the sum of earned components. Maximum is 1.0.

---

## Baseline Inference Script (baseline/run_baseline.py) — Person B

This script runs an LLM agent against all three tasks and prints reproducible scores.

### Requirements
- Reads OPENAI_API_KEY from environment variables
- Uses openai Python SDK
- Model: gpt-4o
- Temperature: 0 (for reproducibility)

### Agent Loop

```
for each task in [sales_summary, churn_analysis, root_cause]:
    obs = env.reset(task_id)
    build system prompt with schema and instructions
    while not done:
        call OpenAI API with conversation history
        parse response for action_type, sql, or final_answer
        construct Action object
        obs, reward, done, info = env.step(action)
        append query result to conversation history
    record final reward.score
print scores and average
```

### System Prompt

```
You are a data analyst. You have access to a SQLite database.
You will be given a business question. Your job is to write SQL queries
to explore the data and then provide a final answer.

To run a query, respond with:
ACTION: query
SQL: <your sql here>

To submit your final answer, respond with:
ACTION: answer
ANSWER: <your answer here>

Be efficient. You have at most 10 actions. Think before querying.
```

### Expected Output Format

```
Task 1 (easy)   - sales_summary:   0.85
Task 2 (medium) - churn_analysis:  0.60
Task 3 (hard)   - root_cause:      0.40
Average: 0.617
```

---

## HTTP Server (server.py) — Person A

Wraps the environment in a simple FastAPI server for Hugging Face Spaces deployment.

```python
POST /reset   body: { "task_id": "sales_summary" }   returns: Observation
POST /step    body: Action                             returns: { observation, reward, done, info }
GET  /state                                            returns: EpisodeState
GET  /health                                           returns: { "status": "ok" }
```

The server instantiates a single SQLAnalystEnv instance on startup.
It is not thread-safe for concurrent users (acceptable for hackathon scope).

---

## Dockerfile — Person A

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python data/seed.py

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

Port 7860 is the default for Hugging Face Spaces.

---

## requirements.txt — Person A

```
pydantic>=2.0
fastapi>=0.110
uvicorn>=0.29
openai>=1.0
sqlite3  # stdlib, no install needed
```

---

## openenv.yaml — Person B

```yaml
name: sql-analyst
version: 1.0.0
author: team
description: >
  An OpenEnv environment where an AI agent answers business questions
  by writing SQL queries against a simulated e-commerce database.
  Tasks range from simple aggregations to multi-step root cause analysis.

tasks:
  - id: sales_summary
    difficulty: easy
    description: Compute Q4 revenue and identify top region
  - id: churn_analysis
    difficulty: medium
    description: Identify churned high-value customers
  - id: root_cause
    difficulty: hard
    description: Investigate a revenue drop and explain the cause

observation_space:
  schema_description: str
  question: str
  query_history: list
  last_result: object
  steps_taken: int
  max_steps: int

action_space:
  action_type: query | answer
  sql: str (optional)
  final_answer: str (optional)

reward_range: [0.0, 1.0]
episode_termination: final answer submitted or max_steps reached
```

---

## README.md — Person B

Sections to include:

1. Overview — what the environment simulates and why it is useful
2. Installation — git clone, pip install -r requirements.txt, python data/seed.py
3. Quick Start — how to run reset/step/state with a code example
4. Task Descriptions — one paragraph per task, difficulty, and what correct looks like
5. Observation and Action Space — table of all fields with types and descriptions
6. Reward Function — explain partial credit and penalties
7. Running the Baseline — how to set OPENAI_API_KEY and run run_baseline.py
8. Docker — docker build and docker run commands
9. Hugging Face Space — link and how to call the API endpoints

---

## Build Order and Timeline

### Session 1 — Foundation (both work in parallel after first 30 min)

Person A:
1. Create repository structure and empty files
2. Write schema.sql
3. Write seed.py with fixed seed and planted anomalies
4. Generate analyst.db and share the file with Person B
5. Write database.py

Person B (after receiving analyst.db):
1. Run the three ground-truth queries manually against analyst.db
2. Record exact values for all three ground_truth dicts
3. Write task_easy.py, task_medium.py, task_hard.py with hardcoded ground truth
4. Start writing grader.py base class

### Session 2 — Core Logic

Person A:
1. Write models.py (all five Pydantic models)
2. Write environment.py (reset, step, state)
3. Write reward.py

Person B:
1. Finish grader.py with all three graders
2. Write run_baseline.py agent loop (can test with mock env)
3. Test graders against known answers to verify determinism

### Session 3 — Integration and Packaging

Person A:
1. Write server.py
2. Write Dockerfile and requirements.txt
3. Test docker build and docker run locally
4. Deploy to Hugging Face Spaces

Person B:
1. Run run_baseline.py against live environment
2. Record and verify baseline scores
3. Write openenv.yaml
4. Write README.md

### Session 4 — Validation and Polish

Both together:
1. Run openenv validate and fix any spec compliance issues
2. Verify all three tasks produce scores between 0.0 and 1.0
3. Verify graders are deterministic (run twice, confirm same score)
4. Verify Docker container starts and responds correctly
5. Final review of README

---

## Validation Checklist

Person A owns:
- [ ] analyst.db is generated by seed.py with seed=42
- [ ] database.py blocks all destructive SQL
- [ ] reset() returns clean state with no history
- [ ] step() with query returns QueryResult in Observation
- [ ] step() with answer returns done=True
- [ ] state() returns EpisodeState with no side effects
- [ ] reward scores are always between 0.0 and 1.0
- [ ] docker build completes without errors
- [ ] docker run starts and /health returns 200
- [ ] HF Space is live and /reset responds

Person B owns:
- [ ] Task 1 grader returns 1.0 for correct answer, 0.0 for wrong answer
- [ ] Task 2 grader returns partial credit for 5 of 10 correct IDs
- [ ] Task 3 grader returns 0.0 if agent says wrong category
- [ ] All three graders are deterministic (no randomness)
- [ ] run_baseline.py runs end to end with a valid OPENAI_API_KEY
- [ ] Baseline scores match documented expected ranges
- [ ] openenv validate passes on openenv.yaml
- [ ] README has working setup instructions

---

## Notes for Agentic AI Building This

1. Always use pydantic v2 syntax (model_validator, field_validator, not @validator)
2. All file paths are relative to the repository root
3. analyst.db is never committed to git — it is always generated at build time by seed.py
4. Ground truth values in graders are hardcoded constants, not computed at runtime
5. The graders must be pure functions — same input always produces same output
6. The reward function must never raise an exception — catch all errors and return score=0.0
7. The server must handle malformed JSON gracefully and return HTTP 422 with a clear message
8. Use sqlite3 from the Python standard library — do not use SQLAlchemy or any ORM
9. Do not use async in environment.py — keep it synchronous for simplicity
10. The baseline script must print scores to stdout in a machine-parseable format