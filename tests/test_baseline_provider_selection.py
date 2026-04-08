from __future__ import annotations

from baseline.run_baseline import extract_answer_from_response, extract_sql_from_response, get_api_client


def test_extract_sql_and_answer() -> None:
    resp = """Some text\nACTION: query\nSQL: SELECT 1;\n"""
    assert extract_sql_from_response(resp) == "SELECT 1;"

    resp2 = """ACTION: answer\nANSWER: hello world\n"""
    assert extract_answer_from_response(resp2) == "hello world"


def test_get_api_client_prefers_anthropic_over_openai(monkeypatch) -> None:
    # Avoid requiring real API keys/working upstream credentials for this test.
    # We only validate provider-priority selection when both env vars are set.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    try:
        model, _client = get_api_client()
    except SystemExit:
        # anthropic package may be unavailable in a minimal environment.
        return

    assert "claude" in model
