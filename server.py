"""
server.py — FastAPI HTTP wrapper for the SQL Analyst environment 
Exposes four endpoints:
    POST /reset   — start a new episode           → Observation
    POST /step    — submit an action               → {observation, reward, done, info}
    GET  /state   — get current episode snapshot   → EpisodeState
    GET  /health  — liveness check                 → {"status": "ok"}

The server is NOT thread-safe — it hosts a single shared environment
instance, which is acceptable for hackathon / demo scope.

Port 7860 is the default for Hugging Face Spaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import SQLAnalystEnv
from env.models import Action, EpisodeState, Observation, Reward


# ---------------------------------------------------------------------------
# App and environment setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv SQL Analyst",
    description="An OpenEnv environment where an AI agent answers business "
                "questions by writing SQL queries.",
    version="1.0.0",
)

# Resolve the database path relative to this file
_DB_PATH = Path(__file__).resolve().parent / "data" / "analyst.db"
_env = SQLAnalystEnv(db_path=str(_DB_PATH), max_steps=10)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    # Some evaluators call POST /reset with an empty body.
    task_id: str = "sales_summary"


class StepResponse(BaseModel):
    """Response body for POST /step."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — always returns 200 OK."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = Body(default_factory=ResetRequest)) -> Observation:
    """Start (or restart) an episode for the given task.

    Returns the initial Observation with schema, question, and empty history.
    """
    try:
        obs = _env.reset(body.task_id)
        return obs
    except ValueError as exc:
        # Unknown task_id — return 422 with a clear message
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Submit an agent action and receive the next observation + reward.

    The request body must be a valid Action (see env/models.py).
    """
    try:
        obs, reward, done, info = _env.step(action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as exc:
        # e.g. calling step() before reset()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=EpisodeState)
def state() -> EpisodeState:
    """Return the current episode state snapshot.  No side effects."""
    return _env.state()


# ---------------------------------------------------------------------------
# Malformed JSON handler — return 422 with a clear message
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — ensures a JSON response."""
    return JSONResponse(
        status_code=422,
        content={"detail": f"Unhandled error: {exc}"},
    )
