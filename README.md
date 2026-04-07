# 🚀 Startup Guide — OpenEnv SQL Analyst

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1. Clone & Enter the Project

```bash
cd openenv-sql-analyst
```

---

## 2. Create a Virtual Environment & Install Dependencies

```bash
# Create a venv and install all core dependencies
uv venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

uv sync
```

To also install the **baseline** dependencies (OpenAI SDK):

```bash
uv sync --extra baseline
```

To install **dev** dependencies (pytest, httpx):

```bash
uv sync --extra dev
```

---

## 3. Generate the Database

The `analyst.db` file is **not committed to git** — it must be generated
from the deterministic seed script:

```bash
python data/seed.py
```

Expected output:

```
✅ Database built at data/analyst.db
   customers: 500 rows
   products: 80 rows
   orders: ~3000 rows
   order_items: ~8000 rows
```

You can also specify a custom output path:

```bash
python data/seed.py --out /tmp/analyst.db
```

---

## 4. Start the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

The API is now live at **http://localhost:7860**.

### Verify it's running

```bash
curl http://localhost:7860/health
# → {"status": "ok"}
```

---

## 5. API Usage

### Reset an episode

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "sales_summary"}'
```

Available task IDs: `sales_summary`, `churn_analysis`, `root_cause`

### Submit a query action

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "query",
    "sql": "SELECT COUNT(*) FROM orders"
  }'
```

### Submit a final answer

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "answer",
    "final_answer": "The total Q4 2023 revenue was $187,432.50 and North was the top region."
  }'
```

### Get current episode state

```bash
curl http://localhost:7860/state
```

---

## 6. Docker (Optional)

Build and run via Docker:

```bash
docker build -t sql-analyst .
docker run -p 7860:7860 sql-analyst
```

The database is seeded during the Docker build step automatically.

---

## 7. Project Structure (Person A Files)

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `env/__init__.py`     | Package init — exports all public symbols        |
| `env/models.py`       | Pydantic v2 data models (5 models)               |
| `env/database.py`     | SQLite query runner with safety checks           |
| `env/environment.py`  | Core reset/step/state environment                |
| `env/reward.py`       | Per-step and final reward computation            |
| `data/schema.sql`     | Table DDL (4 tables)                             |
| `data/seed.py`        | Deterministic data generator (seed=42)           |
| `server.py`           | FastAPI HTTP wrapper                             |
| `Dockerfile`          | Container definition for HF Spaces              |
| `requirements.txt`    | pip dependencies (for Docker)                    |
| `pyproject.toml`      | uv / PEP 621 project metadata                   |
| `tasks/grader.py`     | Abstract BaseTask class (contract for Person B)  |

---

## 8. For Person B

Person B needs `analyst.db` before writing graders. After step 3 above,
share the file or have Person B run `python data/seed.py` themselves.

Person B's files to create:

- `tasks/task_easy.py` — SalesSummaryTask
- `tasks/task_medium.py` — ChurnAnalysisTask
- `tasks/task_hard.py` — RootCauseTask
- `baseline/run_baseline.py` — LLM agent inference script
- `openenv.yaml` — OpenEnv metadata spec
- `README.md` — Full project documentation

The environment dynamically loads tasks via the registry in
`env/environment.py`. Person B's task classes must subclass
`tasks.grader.BaseTask` and implement the `grade()` method.

---

## Troubleshooting

| Problem                          | Fix                                              |
|----------------------------------|--------------------------------------------------|
| `FileNotFoundError: analyst.db`  | Run `python data/seed.py` first                  |
| `ModuleNotFoundError`            | Activate venv: `source .venv/bin/activate`       |
| Port 7860 in use                 | Use `--port 8000` or kill the existing process   |
| `uv` not found                   | Install it: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
