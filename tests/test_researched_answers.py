import json
from collections import Counter
from datetime import date
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from agno.run import RunContext, RunStatus
from openai import OpenAI
from pydantic import ValidationError

from advisor import answer_question, create_agent
from answers import AdvisorAnswer, AnswerValidationError, render_answer, validate_answer
from evidence import MAX_PAYLOAD_CHARACTERS, EvidenceStore
from turn_tools import MAX_TOOL_CALLS, REGISTRY_KEY, TurnRegistry, build_tools


@pytest.fixture
def store():
    return EvidenceStore.load(
        Path(__file__).parent / "fixtures" / "synthetic_evidence.json", on_date=date(2026, 10, 12)
    )


def context(registry):
    return RunContext(
        run_id=str(uuid4()), session_id=str(uuid4()), dependencies={REGISTRY_KEY: registry}
    )


def answer(*, basis="fact", evidence_ids=None, calculation_ids=None, **changes):
    return AdvisorAnswer.model_validate(
        {
            "kind": "advice",
            "cities": ["dubai"],
            "headline": "Compare the supplied observations.",
            "sections": [
                {
                    "title": "Dubai",
                    "items": [
                        {
                            "text": "A synthetic menu observation for this test.",
                            "basis": basis,
                            "evidence_ids": evidence_ids or [],
                            "calculation_ids": calculation_ids or [],
                        }
                    ],
                }
            ],
            "assumptions": [],
            "next_steps": ["Check the branch before making a decision."],
            "questions": [],
            **changes,
        }
    )


def test_lookup_registers_only_returned_records_and_preserves_json_limit(store):
    registry = TurnRegistry()
    lookup, _ = build_tools(store)
    payload = lookup(context(registry), ["dubai"], ["pricing"], limit=1)
    result = json.loads(payload)
    returned = {r["id"] for r in result["data"]["records"]}
    assert set(registry.evidence) == returned
    assert result["data"]["omitted_count"] > 0
    assert len(payload) <= MAX_PAYLOAD_CHARACTERS
    omitted = next(r.id for r in store.records if r.id not in returned)
    with pytest.raises(AnswerValidationError, match="could not be validated"):
        validate_answer(answer(evidence_ids=[*returned, omitted]), registry)


def test_lookup_invalid_and_empty_results_cannot_register_evidence(store):
    registry = TurnRegistry()
    lookup, _ = build_tools(store)
    assert (
        json.loads(lookup(context(registry), ["dubai"], ["pricing"], "no_matching_keyword"))[
            "status"
        ]
        == "no_results"
    )
    assert (
        json.loads(lookup(context(registry), ["amman"], ["pricing"]))["status"] == "invalid_input"
    )
    assert not registry.evidence
    assert registry.tool_calls == 2


@pytest.mark.parametrize(
    "kwargs,category",
    [
        ({}, "fact_without_evidence"),
        ({"evidence_ids": ["invented"]}, "unretrieved_evidence"),
        ({"evidence_ids": ["SEARCH-1"]}, "unretrieved_evidence"),
        ({"basis": "calculation"}, "calculation_without_result"),
        (
            {"basis": "calculation", "calculation_ids": ["CALC-old-turn"]},
            "unregistered_calculation",
        ),
        ({"basis": "proposal", "evidence_ids": ["invented"]}, "unretrieved_evidence"),
        ({"basis": "research_lead"}, "search_not_available"),
        (
            {"basis": "proposal", "headline": "See https://made-up.invalid/source"},
            "model_supplied_url",
        ),
    ],
)
def test_invalid_references_fail_the_whole_answer(kwargs, category):
    with pytest.raises(AnswerValidationError) as error:
        render_answer(answer(**kwargs), TurnRegistry())
    assert error.value.category == category


def test_evidence_must_apply_to_an_answer_city(store):
    record = next(r for r in store.records if r.cities == ["abu_dhabi"])
    with pytest.raises(AnswerValidationError) as error:
        validate_answer(
            answer(evidence_ids=[record.id]), TurnRegistry(evidence={record.id: record})
        )
    assert error.value.category == "evidence_city_mismatch"


@pytest.mark.parametrize(
    "changes",
    [
        {"sections": []},
        {"next_steps": []},
        {"cities": []},
        {"cities": ["dubai", "dubai"]},
        {"kind": "clarification", "questions": []},
        {"source_url": "https://invented.invalid"},
        {"questions": ["One?", "Two?", "Three?"]},
        {"sections": [{"title": "Empty", "items": []}]},
    ],
)
def test_answer_contract_requires_substance_and_bounds(changes):
    with pytest.raises(ValidationError):
        answer(basis="proposal", **changes)


def test_renderer_deduplicates_pages_without_losing_dates_qualifiers_or_flags(store):
    first = store.records[0]
    second = first.model_copy(
        update={
            "id": "SYN-SECOND",
            "entity": "Another observation",
            "needs_review": True,
            "expired": True,
            "valid_until": date(2026, 10, 1),
            "qualifiers": ["Second observation qualifier."],
        }
    )
    registry = TurnRegistry(evidence={r.id: r for r in (first, second)})
    output = render_answer(answer(evidence_ids=[first.id, second.id]), registry)
    assert "[1](<https://example.invalid/fixtures/dubai-menu>)" in output
    assert "2. [" not in output
    assert "checked 2026-10-12 UTC" in output
    assert "published 2026-10-01" in output
    assert "valid until 2026-10-01" in output
    assert "NEEDS REVIEW" in output and "EXPIRED" in output
    assert "Second observation qualifier." in output
    assert first.qualifiers[0] in output
    assert first.source_title in output


def test_model_markdown_is_plain_text_not_an_authoritative_link():
    output = render_answer(
        answer(basis="proposal", headline="[Fake source](javascript:alert)"), TurnRegistry()
    )
    assert "[Fake source](javascript:" not in output
    assert r"\[Fake source\]" in output


def test_successful_calculations_get_ids_and_invalid_calculations_do_not(store):
    registry = TurnRegistry()
    _, calculate = build_tools(store)
    bad = json.loads(calculate(context(registry), "Invalid", 0, 12, 2, 1))
    assert bad["status"] == "invalid_input"
    assert not registry.calculations
    good = json.loads(calculate(context(registry), "Supplied example", 40, 12, 2, 1))
    ref = good["data"]["calculation_id"]
    assert registry.calculations[ref]["contribution_per_order_aed"] == "25"
    output = render_answer(answer(basis="calculation", calculation_ids=[ref]), registry)
    assert "| Revenue per order | 40.00 |" in output
    assert "| Food cost | 12.00 |" in output
    assert "| Contribution per order | 25.00 |" in output
    with pytest.raises(AnswerValidationError):
        validate_answer(answer(basis="calculation", calculation_ids=[ref]), TurnRegistry())


def test_shared_tool_budget_limits_lookup_and_calculator(store):
    registry = TurnRegistry()
    ctx = context(registry)
    lookup, calculate = build_tools(store)
    for _ in range(MAX_TOOL_CALLS):
        lookup(ctx, ["dubai"], ["pricing"], "no_matches")
    assert json.loads(calculate(ctx, "Example", 40, 12, 2, 1))["status"] == "limit_reached"
    assert json.loads(lookup(ctx, ["dubai"], ["pricing"]))["status"] == "limit_reached"
    assert not registry.calculations and not registry.evidence
    assert len(registry.warnings) == 1


@pytest.mark.integration
def test_real_agno_lookup_and_followup_registry_isolation_with_failed_run_persistence(
    tmp_path, store
):
    requests = []
    calls = Counter()
    ref = store.records[0].id

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        question = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
        calls[question] += 1
        if question != "Reuse the previous citation" and calls[question] == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{len(requests)}",
                        "type": "function",
                        "function": {
                            "name": "lookup_market_evidence",
                            "arguments": json.dumps(
                                {
                                    "cities": ["dubai"],
                                    "topics": ["pricing"],
                                    "query": "pizza",
                                    "limit": 1,
                                }
                            ),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": answer(evidence_ids=[ref]).model_dump_json()}
            reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"fixture-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        agent = create_agent("fixture-key", tmp_path / "sessions.db", evidence_store=store)
        agent.model.client = OpenAI(
            api_key="fixture-key", base_url="https://openrouter.ai/api/v1", http_client=http_client
        )
        session_id = str(uuid4())
        first = answer_question("Find Dubai pizza evidence", session_id, agent)
        assert first.status == RunStatus.completed
        assert list(first.sources) == [ref]
        assert "**Sources**" in first.markdown
        for tool in requests[0]["tools"]:
            assert "run_context" not in tool["function"]["parameters"]["properties"]
        assert "advisor_turn" not in requests[0]["messages"][0]["content"]
        with pytest.raises(AnswerValidationError) as error:
            answer_question("Reuse the previous citation", session_id, agent)
        assert error.value.category == "unretrieved_evidence"
        saved = agent.get_session(session_id=session_id)
        assert [r.status for r in saved.runs] == [RunStatus.completed, RunStatus.error]
        third = answer_question("Retrieve again for this question", session_id, agent)
        assert third.status == RunStatus.completed
        users = [m["content"] for m in requests[-1]["messages"] if m["role"] == "user"]
        assert "Find Dubai pizza evidence" in users
        assert "Reuse the previous citation" not in users
        assert set(third.sources) == {ref}
        agent.db.close()
