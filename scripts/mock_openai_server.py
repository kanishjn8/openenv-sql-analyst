"""Minimal OpenAI-compatible mock server for local inference testing.

Implements:
  POST /v1/chat/completions

This is intentionally tiny and deterministic so `inference.py` can be smoke-tested
without real API keys.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Mock OpenAI Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest) -> Dict[str, Any]:
    # Very small heuristic: first respond with a SQL query, then with an answer.
    prompt = "\n".join(m.content for m in body.messages)

    if "Last result:" not in prompt:
        content = "ACTION: query\nSQL: SELECT 1 AS one;\n"
    else:
        content = "ACTION: answer\nANSWER: (mock) done\n"

    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
