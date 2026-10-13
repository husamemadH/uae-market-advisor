import json
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.run import RunStatus
from openai import OpenAI
from rich.console import Console
from sqlalchemy.exc import OperationalError

import app
from advisor import answer_question, create_agent, session_usage
from observability import (
    CURRENT_TURN,
    RequestUsage,
    RunLog,
    RunRecord,
    TurnObservation,
    capture_usage,
    render_usage,
    summarize_usage,
)
from sessions import SessionStore


def final_message(content=None):
    return {
        "role": "assistant",
        "content": content
        or json.dumps(
            {
                "kind": "insufficient_evidence",
                "cities": ["dubai"],
                "headline": "Synthetic answer.",
                "sections": [],
                "assumptions": [],
                "next_steps": [],
                "questions": [],
            }
        ),
    }


def reply(message=None, usage=None):
    return httpx.Response(
        200,
        json={
            "id": "generation-fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [
                {"index": 0, "message": message or final_message(), "finish_reason": "stop"}
            ],
            "usage": usage,
        },
    )


def model_client(agent, transport):
    client = httpx.Client(transport=httpx.MockTransport(transport))
    agent.model.client = OpenAI(api_key="fixture-key", http_client=client, max_retries=0)
    return client


@pytest.mark.parametrize("cost", [None, "NaN", float("inf"), -1, True, "invalid"])
def test_missing_or_invalid_cost_never_becomes_zero(cost):
    request = RequestUsage()
    capture_usage(request, SimpleNamespace(usage=SimpleNamespace(cost=cost)))
    summary = summarize_usage([request])
    assert summary.reported_cost_usd is None
    assert summary.input_tokens is None
    assert not summary.usage_complete


def test_explicit_zero_is_preserved_and_missing_one_request_makes_total_unknown():
    known = RequestUsage(input_tokens=10, output_tokens=3, reported_cost_usd=Decimal("0"))
    complete = summarize_usage([known])
    assert complete.reported_cost_usd == 0
    assert complete.usage_complete
    partial = summarize_usage([known, RequestUsage()])
    assert partial.input_tokens is None
    assert partial.output_tokens is None
    assert partial.reported_cost_usd is None
    assert partial.known_reported_cost_usd == 0
    assert not partial.usage_complete


@pytest.mark.integration
@pytest.mark.parametrize(
    "failure,category",
    [
        (401, "authentication"),
        (403, "authentication"),
        (402, "credits"),
        (429, "rate_limit"),
        (400, "model_configuration"),
        (404, "model_configuration"),
        (500, "provider"),
        ("timeout", "timeout"),
        ("connection", "connection"),
        ("schema", "validation"),
        ("cancel", "cancelled"),
    ],
)
def test_real_adapter_errors_are_categorized_logged_once_and_recover(tmp_path, failure, category):
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        if len(bodies) > 1:
            return reply()
        if failure == "timeout":
            raise httpx.ReadTimeout("fixture timeout", request=request)
        if failure == "connection":
            raise httpx.ConnectError("fixture connection", request=request)
        if failure == "cancel":
            raise KeyboardInterrupt()
        if failure == "schema":
            return reply(final_message("TRUNCATED {"))
        return httpx.Response(failure, json={"error": {"message": "PRIVATE_PROVIDER_DIAGNOSTIC"}})

    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    session_id = SessionStore(agent.db).new()
    with model_client(agent, respond):
        failed = answer_question("Fail this turn", session_id, agent)
        assert len(bodies) == 1  # No SDK, application, or framework retry.
        assert failed.error_category == category
        assert failed.content is None and not failed.markdown
        assert failed.status == (
            RunStatus.cancelled if category == "cancelled" else RunStatus.error
        )
        assert failed.run_id
        assert failed.usage.reported_cost_usd is None
        recovered = answer_question("Recover this turn", session_id, agent)
        assert recovered.status == RunStatus.completed
        assert "Fail this turn" not in str(bodies[-1]["messages"])
    records, unreadable = RunLog(tmp_path / "runs.jsonl").records(session_id)
    assert unreadable == 0 and len(records) == 2
    assert records[0].run_id == failed.run_id
    assert records[0].error_category == category
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in (tmp_path / "runs.jsonl").read_text()
    assert CURRENT_TURN.get() is None
    agent.db.close()


@pytest.mark.integration
def test_failed_continuation_retains_known_cost_and_tool_timings(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return reply(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "lookup-fixture",
                            "type": "function",
                            "function": {
                                "name": "lookup_market_evidence",
                                "arguments": json.dumps(
                                    {
                                        "cities": ["dubai"],
                                        "topics": ["pricing"],
                                        "query": "",
                                        "limit": 1,
                                    }
                                ),
                            },
                        }
                    ],
                },
                {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16, "cost": 0.004},
            )
        raise httpx.ReadTimeout("fixture timeout", request=request)

    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    session_id = SessionStore(agent.db).new()
    with model_client(agent, respond):
        result = answer_question("Synthetic lookup then failure", session_id, agent)
    assert len(calls) == 2
    assert result.error_category == "timeout"
    assert result.usage.reported_cost_usd is None
    assert result.usage.known_reported_cost_usd == Decimal("0.004")
    assert result.record.requests[0].input_tokens == 12
    assert result.record.tool_events[0].result_count == 1
    assert result.record.tool_events[0].duration_ms >= 0
    assert result.record.tool_output_characters > 0
    output = session_usage(session_id, agent)
    assert "Reported cost (USD): Unknown" in output
    assert "subtotal (USD): 0.004; incomplete" in output
    agent.db.close()


@pytest.mark.integration
@pytest.mark.parametrize("operation", ["get_session", "upsert_session", "upsert_run", "silent_run"])
def test_sqlite_failures_never_claim_a_completed_saved_answer(tmp_path, monkeypatch, operation):
    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    session_id = SessionStore(agent.db).new()
    method = "upsert_run" if operation == "silent_run" else operation
    original = getattr(SqliteDb, method)
    failure = (
        Mock(return_value=None)
        if operation == "silent_run"
        else Mock(side_effect=OperationalError("fixture", {}, Exception("disk failure")))
    )
    monkeypatch.setattr(SqliteDb, method, failure)
    requests = []

    def respond(request):
        requests.append(request)
        return reply()

    with model_client(agent, respond):
        result = answer_question("Storage failure fixture", session_id, agent)
    assert result.status == RunStatus.error
    assert result.error_category == "storage"
    assert result.content is None and result.markdown == ""
    if operation == "get_session":
        assert requests == []
    monkeypatch.setattr(SqliteDb, method, original)
    saved = agent.db.get_run(result.run_id)
    assert saved is None or saved.status != RunStatus.completed
    assert RunLog(tmp_path / "runs.jsonl").records(session_id)[0][0].error_category == "storage"
    agent.db.close()


@pytest.mark.integration
def test_log_write_failure_keeps_valid_answer_and_shows_warning(tmp_path):
    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    (tmp_path / "runs.jsonl").mkdir()
    with model_client(agent, lambda request: reply()):
        result = answer_question("Question", str(uuid4()), agent)
    assert result.status == RunStatus.completed
    assert any("usage history is incomplete" in warning for warning in result.warnings)
    agent.db.close()


@pytest.mark.integration
def test_usage_command_is_local_resumable_and_isolated(tmp_path, monkeypatch):
    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    first, second = SessionStore(agent.db).new(), SessionStore(agent.db).new()
    with model_client(
        agent,
        lambda request: reply(
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
                "cost": 0.001,
            }
        ),
    ):
        first_result = answer_question("First session question", first, agent)
    agent.db.close()
    reader = create_agent("fixture-key", tmp_path / "sessions.db")
    assert first_result.run_id in session_usage(first, reader)
    assert first_result.run_id not in session_usage(second, reader)
    console = Console(file=StringIO(), color_system=None, width=120)
    monkeypatch.setattr(console, "input", Mock(side_effect=["/usage", "/new", "/usage", "/quit"]))
    model = Mock(side_effect=AssertionError("usage must not call the model"))
    monkeypatch.setattr(OpenRouter, "response", model)
    assert app.chat_loop(reader, console, session_id=first) == 0
    model.assert_not_called()
    output = console.file.getvalue()
    assert "Reported cost (USD): 0.001" in output
    assert "No recorded usage for this session" in output
    reader.db.close()


def test_log_read_skips_corrupt_line_and_reports_incomplete_history(tmp_path):
    record = RunRecord(
        run_id=str(uuid4()),
        session_id=str(uuid4()),
        timestamp_utc=datetime.now(timezone.utc),
        model_requested="fixture",
        prompt_version="test",
        evidence_dataset_version="fixture",
        question="Question",
        outcome="completed",
        latency_ms=1,
    )
    log = RunLog(tmp_path / "runs.jsonl")
    log.append(record)
    with log.path.open("a") as output:
        output.write('{"truncated":\n')
    records, unreadable = log.records(record.session_id)
    assert records == [record] and unreadable == 1
    assert "Unknown" in render_usage(records, unreadable)
    assert "unreadable log records" in render_usage(records, unreadable)


def test_actual_configuration_disables_retries_and_sets_request_timeout(tmp_path):
    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    client = agent.model.get_client()
    assert client.max_retries == 0
    assert client.timeout == 45
    assert agent.retries == 0
    assert agent.model.max_tokens == 4096
    assert agent.tool_call_limit == 6
    client.close()
    agent.db.close()


def test_limit_results_are_timed_and_tool_output_remains_bounded():
    from agno.run import RunContext

    from evidence import EvidenceStore
    from turn_tools import MAX_TOOL_OUTPUT_CHARACTERS, REGISTRY_KEY, TurnRegistry, build_tools

    registry = TurnRegistry(output_characters=MAX_TOOL_OUTPUT_CHARACTERS - 1600)
    trace = TurnObservation()
    token = CURRENT_TURN.set(trace)
    context = RunContext(
        run_id=str(uuid4()), session_id=str(uuid4()), dependencies={REGISTRY_KEY: registry}
    )
    _, calculate, _ = build_tools(EvidenceStore(records=()))
    try:
        output = calculate(context, "Fixture", 40, 12, 2, 1, 0.5, 0.25, 30000, 30)
    finally:
        CURRENT_TURN.reset(token)
    assert json.loads(output)["status"] == "limit_reached"
    assert not registry.calculations
    assert registry.output_characters <= MAX_TOOL_OUTPUT_CHARACTERS
    assert trace.tool_events[0].status == "limit_reached"
    assert trace.tool_events[0].result_count == 0


@pytest.mark.integration
def test_actual_tool_loop_stops_executing_at_six_calls(tmp_path):
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return reply(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": "lookup_market_evidence",
                                "arguments": json.dumps(
                                    {
                                        "cities": ["dubai"],
                                        "topics": ["pricing"],
                                        "query": "",
                                        "limit": 1,
                                    }
                                ),
                            },
                        }
                        for index in range(8)
                    ],
                }
            )
        return reply()

    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    with model_client(agent, respond):
        result = answer_question("Synthetic tool limit check", str(uuid4()), agent)
    assert result.status == RunStatus.completed
    assert len(result.record.tool_events) == 6
    assert len(result.tools) == 6
    assert result.record.tool_output_characters <= 24000
    assert len(bodies) == 2
    assert any(
        "limit" in message.get("content", "").lower()
        for message in bodies[-1]["messages"]
        if message["role"] == "tool"
    )
    agent.db.close()
