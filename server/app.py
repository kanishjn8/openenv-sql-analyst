"""ASGI app entrypoint for OpenEnv validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from env.environment import SQLAnalystEnv
from env.models import Action, EpisodeState, Observation, Reward


app = FastAPI(
    title="OpenEnv SQL Analyst",
    description="An OpenEnv environment where an AI agent answers business questions by writing SQL queries.",
    version="1.0.0",
)

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "analyst.db"
_env = SQLAnalystEnv(db_path=str(_DB_PATH), max_steps=10)


class ResetRequest(BaseModel):
    # Some evaluators call POST /reset with an empty body.
    # Defaulting ensures reset still works while preserving compatibility with
    # explicit task selection.
    task_id: str = "sales_summary"


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = Body(default_factory=ResetRequest)) -> Observation:
    try:
        return _env.reset(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    try:
        obs, reward, done, info = _env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state", response_model=EpisodeState)
def state() -> EpisodeState:
    return _env.state()


@app.exception_handler(Exception)
async def _global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": f"Unhandled error: {exc}"})


def main() -> None:
    """CLI server entrypoint required by OpenEnv validators."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

