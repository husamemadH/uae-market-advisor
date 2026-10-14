"""Exercise citation correction through real Agno, HTTP parsing, tools and SQLite."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from agno.run import RunStatus
from openai import OpenAI

from advisor import answer_question, create_agent, last_answer
from evidence import EvidenceStore
from observability import RunLog
from sessions import SessionStore
from turn_tools import MAX_TOOL_CALLS


def answer(reference):
    return {
        "kind": "advice",
        "cities": ["dubai"],
        "headline": "Compare the supplied research.",
        "sections": [
            {
                "title": "Observation",
                "items": [
                    {
                        "text": "A synthetic menu observation.",
                        "basis": "fact",
                        "evidence_ids": [reference],
                        "calculation_ids": [],
                    }
                ],
            }
        ],
        "assumptions": [],
        "next_steps": ["Inspect the original menu."],
        "questions": [],
    }


def lookup(call_id, query="pizza"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "lookup_market_evidence",
            "arguments": json.dumps(
                {"cities": ["dubai"], "topics": ["pricing"], "query": query, "limit": 1}
            ),
        },
    }


def reply(message, number):
    return httpx.Response(
        200,
        json={
            "id": f"recovery-fixture-{number}",
            "object": "chat.completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
        },
    )


@pytest.mark.integration
@pytest.mark.parametrize("outcome", ["corrected", "invalid_again", "provider_failure"])
def test_citation_correction_is_bounded_accounted_and_saved(tmp_path, outcome):
    requests = []
    store = EvidenceStore.load(
        Path(__file__).parent / "fixtures/synthetic_evidence.json", on_date=date(2026, 10, 12)
    )

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        number = len(requests)
        if number == 2:
            assert "single correction attempt" in str(body["messages"])
            if outcome == "provider_failure":
                return httpx.Response(429, json={"error": {"message": "fixture rate limit"}})
            if outcome == "corrected":
                return reply(
                    {"role": "assistant", "content": None, "tool_calls": [lookup("fresh")]},
                    number,
                )
        if number == 3:
            tool_message = next(m for m in reversed(body["messages"]) if m["role"] == "tool")
            reference = json.loads(tool_message["content"])["data"]["records"][0]["id"]
        else:
            reference = "INVENTED"
        return reply({"role": "assistant", "content": json.dumps(answer(reference))}, number)

    agent = create_agent("fixture-key", tmp_path / "sessions.db", evidence_store=store)
    sessions = SessionStore(agent.db)
    session_id = sessions.new()
    sessions.update_profile(session_id, "branch_format", "takeaway")
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        agent.model.client = OpenAI(api_key="fixture-key", http_client=http_client, max_retries=0)
        result = answer_question("Compare Dubai pizza competitors.", session_id, agent)
    assert result.record.validation_errors == ["unretrieved_evidence"]
    assert len(result.record.attempt_run_ids) == 2
    assert agent.db.get_run(result.record.attempt_run_ids[0]).status == RunStatus.error
    assert result.run_id == result.record.attempt_run_ids[1]
    assert sessions.profile(session_id).branch_format == "takeaway"
    assert agent.tool_call_limit == MAX_TOOL_CALLS
    records, unreadable = RunLog(tmp_path / "runs.jsonl").records(session_id)
    assert unreadable == 0 and len(records) == 1
    assert records[0].model_requests == len(requests)
    if outcome == "corrected":
        assert len(requests) == 3
        assert result.status == RunStatus.completed
        assert result.usage.reported_cost_usd == Decimal("0.003")
        assert result.usage.input_tokens == 30
        assert result.record.tool_events[0].name == "lookup_market_evidence"
        assert last_answer(session_id, agent).markdown == result.markdown
        assert not any(
            "single correction attempt" in str(m.content)
            for m in agent.db.get_run(result.run_id).messages
        )
    elif outcome == "invalid_again":
        assert len(requests) == 2
        assert result.error_category == "validation"
        assert result.content is None and not result.markdown
        assert result.usage.reported_cost_usd == Decimal("0.002")
    else:
        assert len(requests) == 2
        assert result.error_category == "rate_limit"
        assert result.usage.reported_cost_usd is None
        assert result.usage.known_reported_cost_usd == Decimal("0.001")
    agent.db.close()


@pytest.mark.integration
def test_correction_shares_remaining_tool_budget_with_first_attempt(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        number = len(requests)
        if number in {1, 3}:
            return reply(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        lookup(f"empty-{number}-{index}", query="no_matching_keyword")
                        for index in range(1 if number == 1 else MAX_TOOL_CALLS)
                    ],
                },
                number,
            )
        return reply({"role": "assistant", "content": json.dumps(answer("INVENTED"))}, number)

    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    session_id = SessionStore(agent.db).new()
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        agent.model.client = OpenAI(api_key="fixture-key", http_client=http_client, max_retries=0)
        result = answer_question("Look up competitors.", session_id, agent)
    assert result.error_category == "validation"
    assert len(requests) == 4
    assert len(result.record.attempt_run_ids) == 2
    assert len(result.record.tool_events) == MAX_TOOL_CALLS
    assert agent.tool_call_limit == MAX_TOOL_CALLS
    agent.db.close()


@pytest.mark.integration
def test_exhausted_tool_budget_prevents_correction(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return reply(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        lookup(f"empty-{number}", query="no_matching_keyword")
                        for number in range(MAX_TOOL_CALLS)
                    ],
                },
                1,
            )
        return reply({"role": "assistant", "content": json.dumps(answer("INVENTED"))}, 2)

    agent = create_agent("fixture-key", tmp_path / "sessions.db")
    session_id = SessionStore(agent.db).new()
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        agent.model.client = OpenAI(api_key="fixture-key", http_client=http_client, max_retries=0)
        result = answer_question("Look up competitors.", session_id, agent)
    assert result.error_category == "validation"
    assert len(requests) == 2
    assert len(result.record.attempt_run_ids) == 1
    assert len(result.record.tool_events) == MAX_TOOL_CALLS
    agent.db.close()
