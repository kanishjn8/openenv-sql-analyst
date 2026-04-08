# OpenEnv SQL Analyst — Agent Evaluation Benchmark

An OpenEnv environment where an AI agent answers business questions by writing SQL queries against a simulated e-commerce database. The environment provides structured tasks ranging from simple aggregations to multi-step root cause analysis, enabling systematic evaluation of agent reasoning and SQL proficiency.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Tasks](#tasks)
5. [Observation and Action Space](#observation-and-action-space)
6. [Reward Function](#reward-function)
7. [Running the Baseline Agent](#running-the-baseline-agent)
8. [Docker](#docker)
9. [API Endpoints](#api-endpoints)

---

## Overview

**SQL Analyst** is a deterministic benchmarking environment for evaluating AI agents' ability to:
- Write correct SQL queries to explore data
- Aggregate and analyse structured information
- Draw logical conclusions from query results
- Reason through root cause analysis

The environment contains:
- **4 database tables**: `customers`, `products`, `orders`, `order_items` (~500 customers, ~80 products, ~3000 orders)
- **3 tasks** of increasing difficulty (easy, medium, hard)
- **Deterministic data generation** (seed=42) for reproducible evaluation
- **Per-step rewards** for query validity and efficiency
- **Task-specific graders** that reward partial correctness

### Use Cases

- **Agent evaluation**: Benchmark LLM agents on structured data analysis
- **Iterative improvement**: Measure progress on reasoning tasks
- **Instruction tuning**: Generate synthetic data for training models on SQL reasoning
- **Curriculum learning**: Progress from simple queries (easy) to root cause analysis (hard)

---

## Installation

### Prerequisites

- Python 3.9+
- pip or uv package manager

### Environment variables (.env)

This repo includes a `.env.example` file with the required variables for the competition `inference.py` contract:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME`

Copy it locally (do not commit secrets):

```bash
cp .env.example .env
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/openenv-sql-analyst.git
cd openenv-sql-analyst
```

### Step 2: Install Dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Using `uv`:

```bash
uv sync
```

### Step 3: Generate the Database

The database is **not committed to git** — generate it deterministically:

```bash
python data/seed.py
```

Expected output:

```
✅ Database built at C:\...\data\analyst.db
   customers: 500 rows
   products: 80 rows
   orders: 3000 rows
   order_items: 8601 rows
```

To verify:

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/analyst.db')
for table in ['customers', 'products', 'orders', 'order_items']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count}')
"
```

---

## Quick Start

### Example: Run a Manual Episode

```python
from env.environment import SQLAnalystEnv
from env.models import Action

# Initialise environment
env = SQLAnalystEnv('data/analyst.db')

# Start a new episode
obs = env.reset('sales_summary')
print(f"Question: {obs.question}")
print(f"Schema:\n{obs.schema_description[:200]}")

# Submit a query
obs, reward, done, info = env.step(
    Action(action_type='query', sql='SELECT COUNT(*) FROM orders')
)
print(f"Query valid: {reward.sql_valid}, Score: {reward.score}")

# Submit a final answer
obs, reward, done, info = env.step(
    Action(
        action_type='answer',
        final_answer='The total revenue was high and East was the top region.'
    )
)
print(f"Final score: {reward.score}, Done: {done}")
```

### Example: Run All Three Tasks

```python
from env.environment import SQLAnalystEnv

env = SQLAnalystEnv('data/analyst.db')

for task_id in ['sales_summary', 'churn_analysis', 'root_cause']:
    obs = env.reset(task_id)
    print(f"\n{task_id}: {obs.question}")
    state = env.state()
    print(f"State: {state.task_id}, Steps: {state.steps_taken}/{state.max_steps}")
```

---

## Tasks

### Task 1: Sales Summary (Easy)

**ID**: `sales_summary` | **Difficulty**: Easy

Compute the total revenue from **completed orders in Q4 2023** (October–December) and identify **which region generated the highest sales** during that period.

**What a correct answer looks like:**
> "Total revenue from Q4 2023 completed orders was $1,534,367.72. The East region generated the highest sales with $420,069.27."

**Evaluation**: Agent earns +0.50 for identifying the correct top region (case-insensitive), +0.50 if any number in the response is within 1% of the true total revenue, or +0.25 if within 10%.

---

### Task 2: Churn Analysis (Medium)

**ID**: `churn_analysis` | **Difficulty**: Medium

Identify **customers with ≥3 orders who have not made a purchase in the last 90 days** (relative to 2024-03-31). Rank the **top 10 by lifetime value**.

**What a correct answer looks like:**
> "87 customers have churned. The top 10 by lifetime spend are: 203, 78, 89, 12, 167, 45, 301, 56, 324, 99. Customer 203 has the highest lifetime value at $39,789.35."

**Evaluation**: 
- +0.40 if ≥7 of the top 10 customer IDs appear in the answer
- +0.30 if all 10 IDs appear in the correct order (replaces the 0.40)
- +0.20 if ≥5 IDs appear (partial credit)
- +0.30 if the correct total churned count (87) is stated
- Capped at 1.0

---

### Task 3: Root Cause Analysis (Hard)

**ID**: `root_cause` | **Difficulty**: Hard

Analyse the revenue drop from February 2024 to March 2024 and identify the **most likely root cause**.

**What a correct answer looks like:**
> "Revenue fell from $50,200 in February to $41,200 in March, a 17.9% decline. The root cause is the Electronics category, which dropped from $22,138 to $9,683 due to product availability issues."

**Evaluation**:
- +0.20 if answer mentions decline/drop/decrease/fell/lower
- +0.40 if answer mentions the correct root cause category (case-insensitive: "Electronics")
- +0.20 if answer mentions a specific product or product ID
- +0.20 coherence bonus if answer has ≥3 sentences AND contains "category" or "product"
- Capped at 1.0

---

## Observation and Action Space

### Observation (`Observation` model)

Returned after each `step()` call. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema_description` | `str` | Human-readable schema of all 4 tables and columns |
| `question` | `str` | The business question to answer |
| `query_history` | `list[QueryResult]` | All queries executed so far (SQL, result rows, errors) |
| `last_result` | `QueryResult \| None` | Most recent query result (None initially) |
| `steps_taken` | `int` | Number of actions taken so far (0–10) |
| `max_steps` | `int` | Hard limit on actions per episode (default: 10) |
| `task_id` | `str` | Current task ID |
| `done` | `bool` | Whether the episode has ended |

### Action (`Action` model)

Submitted by the agent via `step()`. Exactly one of `sql` or `final_answer` must be set:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"query" \| "answer"` | Type of action |
| `sql` | `str \| None` | SQL query (required if `action_type == "query"`) |
| `final_answer` | `str \| None` | Final answer text (required if `action_type == "answer"`) |

### Reward (`Reward` model)

Returned after each step. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Final score for this step, clamped to [0.0, 1.0] |
| `sql_valid` | `bool` | Whether the SQL executed without error |
| `result_shape_correct` | `bool` | Whether the result had ≥1 row (for queries) |
| `answer_correct` | `bool` | Whether the final answer matched ground truth (≥0.99 score) |
| `partial_credit` | `float` | Incremental credit earned (for query actions) |
| `penalty` | `float` | Any penalties applied (efficiency, duplicates) |
| `reason` | `str` | Human-readable explanation of reward calculation |

---

## Reward Function

### Query Actions

When the agent submits a SQL query:

- **+0.10** if the SQL is valid (executes without error)
- **+0.05** if the result has ≥1 row
- **−0.05** if the query duplicates a previous query
- **−0.10** if steps taken > 70% of max_steps (efficiency penalty)

### Answer Actions

When the agent submits a final answer:

- **Grader score** (0.0–1.0) from the task's `grade()` method
- **−0.05 per query** beyond the first 5 (efficiency penalty)
- **+accumulated partial credit** from earlier query steps (capped at 0.20)

### Final Score Calculation

```
final_score = clamp(grader_score − efficiency_penalty + partial_credit, 0.0, 1.0)
```

### Efficiency Incentives

The environment rewards concise problem-solving:
- Queries beyond 5 incur −0.05 each
- Approaching the 10-step limit (>70%) incurs −0.10
- Partial credit from intermediate queries is capped at +0.20

---

## Running the Baseline Agent

The `baseline/run_baseline.py` script implements an LLM agent that solves all three tasks by:
1. Accepting SQL queries to explore the database
2. Parsing LLM responses for `ACTION: query` and `ACTION: answer` blocks
3. Managing conversation history
4. Submitting final answers and reporting scores

### Prerequisites

Install one of:

```bash
# Anthropic (Claude)
pip install anthropic

# OpenAI
pip install openai
```

### Step 1: Set Your API Key

```bash
# Anthropic
export ANTHROPIC_API_KEY='sk-ant-...'

# OpenAI
export OPENAI_API_KEY='sk-...'
```

On Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY='sk-ant-...'
$env:OPENAI_API_KEY='sk-...'
```

### Step 2: Run the Baseline

```bash
python baseline/run_baseline.py
```

Expected output:

```
======================================================================
OpenEnv SQL Analyst Baseline — Agent with LLM Loop
======================================================================

✅ Using model: claude-3-5-haiku-20241022

Running Task (easy): sales_summary
----------------------------------------------------------------------
  ✓ Query executed (rows: True)
  ✓ Query executed (rows: True)
  ✓ Answer submitted: score=0.75
  
Running Task (medium): churn_analysis
----------------------------------------------------------------------
  ✓ Query executed (rows: True)
  ✓ Query executed (rows: True)
  ✓ Answer submitted: score=0.60
  
Running Task (hard): root_cause
----------------------------------------------------------------------
  ✓ Query executed (rows: True)
  ✓ Answer submitted: score=0.50

======================================================================
RESULTS
======================================================================
Task 1 (easy  ) - sales_summary:   0.75
Task 2 (medium) - churn_analysis:  0.60
Task 3 (hard  ) - root_cause:      0.50

Average: 0.617

```

### Agent Prompt Template

The baseline uses this system prompt:

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

---

## Docker

Build a self-contained container image:

```bash
docker build -t sql-analyst .
```

Run the container:

```bash
docker run -p 7860:7860 sql-analyst
```

The container automatically:
- Installs dependencies
- Generates the database (seed.py)
- Starts the FastAPI server on port 7860

### Verify in the Container

```bash
curl http://localhost:7860/health
# {"status": "ok"}
```

---

## Submission validation

This repo includes `scripts/validate-submission.sh`, which runs the common submission checks:

- Ping your deployed HF Space (`/health` then `/reset`)
- Build the Docker image locally
- Run `openenv validate` (requires `openenv-core`)

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-space.hf.space
```

---

## API Endpoints

The `server.py` FastAPI application provides a synchronous HTTP API.

### Health Check

```
GET /health
```

**Response:**
```json
{"status": "ok"}
```

---

### Reset Episode

```
POST /reset
Content-Type: application/json

{
  "task_id": "sales_summary"
}
```

**Response:**
```json
{
  "schema_description": "=== Database Schema ===\n...",
  "question": "What was the total revenue...",
  "query_history": [],
  "last_result": null,
  "steps_taken": 0,
  "max_steps": 10,
  "task_id": "sales_summary",
  "done": false
}
```

**Available task IDs:**
- `sales_summary`
- `churn_analysis`
- `root_cause`

---

### Submit Action (Query or Answer)

```
POST /step
Content-Type: application/json

{
  "action_type": "query",
  "sql": "SELECT COUNT(*) FROM orders"
}
```

**Query Response:**
```json
{
  "observation": {
    "schema_description": "...",
    "question": "...",
    "query_history": [
      {
        "sql": "SELECT COUNT(*) FROM orders",
        "success": true,
        "rows": [{"COUNT(*)": 3000}],
        "error": null,
        "row_count": 1
      }
    ],
    "last_result": {...},
    "steps_taken": 1,
    "max_steps": 10,
    "task_id": "sales_summary",
    "done": false
  },
  "reward": {
    "score": 0.15,
    "sql_valid": true,
    "result_shape_correct": true,
    "answer_correct": false,
    "partial_credit": 0.15,
    "penalty": 0.0,
    "reason": "+0.10 valid SQL; +0.05 non-empty result"
  },
  "done": false,
  "info": {}
}
```

**Answer Request:**
```json
{
  "action_type": "answer",
  "final_answer": "The total Q4 revenue was $1,534,367.72 and East was the top region."
}
```

**Answer Response:**
```json
{
  "observation": {...},
  "reward": {
    "score": 0.75,
    "sql_valid": true,
    "result_shape_correct": true,
    "answer_correct": true,
    "partial_credit": 0.15,
    "penalty": 0.0,
    "reason": "grader=0.75; +0.15 partial credit (capped)"
  },
  "done": true,
  "info": {"final_score": 0.75}
}
```

---

### Get Episode State

```
GET /state
```

**Response:**
```json
{
  "task_id": "sales_summary",
  "steps_taken": 1,
  "max_steps": 10,
  "cumulative_score": 0.15,
  "done": false,
  "query_history": [...],
  "current_question": "What was the total revenue..."
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: analyst.db` | Run `python data/seed.py` |
| `ModuleNotFoundError: env` | Check Python path; run from project root |
| API key errors | Verify env vars: `echo $ANTHROPIC_API_KEY` or `$env:ANTHROPIC_API_KEY` |
| Port 7860 in use | `docker run -p 8000:7860 ...` to map to 8000 |
| Slow imports | Pre-compile: `python -m py_compile baseline/run_baseline.py` |

---

## Project Structure

```
openenv-sql-analyst/
├── data/
│   ├── schema.sql           # Table DDL (4 tables)
│   ├── seed.py              # Deterministic data generator (seed=42)
│   └── analyst.db           # Generated SQLite database (not in git)
├── env/
│   ├── __init__.py
│   ├── models.py            # Pydantic data models
│   ├── database.py          # Safe SQL runner
│   ├── environment.py       # Core reset/step/state class
│   └── reward.py            # Reward computation
├── tasks/
│   ├── __init__.py
│   ├── grader.py            # BaseTask abstract class & router
│   ├── task_easy.py         # SalesSummaryTask
│   ├── task_medium.py       # ChurnAnalysisTask
│   └── task_hard.py         # RootCauseTask
├── baseline/
│   ├── __init__.py
│   └── run_baseline.py      # LLM agent baseline
├── server.py                # FastAPI HTTP wrapper
├── Dockerfile               # Container definition
├── openenv.yaml             # OpenEnv metadata
├── requirements.txt         # pip dependencies
├── pyproject.toml           # uv / PEP 621 project metadata
└── README.md                # This file
```

---

## Citation

If you use OpenEnv SQL Analyst in your research, please cite:

```
@software{openenv_sql_analyst,
  title={OpenEnv SQL Analyst: A Deterministic Benchmark for Agent Evaluation},
  author={Team},
  year={2024},
  url={https://github.com/yourusername/openenv-sql-analyst}
}
```

---

## License

MIT
