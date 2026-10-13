import json
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from agno.run import RunStatus
from agno.run.agent import RunOutput
from openai import OpenAI
from pydantic import ValidationError

from advisor import answer_question, create_agent, export_answer, last_answer
from answers import AdvisorAnswer, AnswerSection, Statement, render_answer
from observability import RunLog
from sessions import SessionStore
from turn_tools import TurnRegistry


@pytest.mark.integration
@pytest.mark.parametrize(
    "food,contribution,monthly,daily", [(12, "20", 1500, 50), (40, "-8", None, None)]
)
def test_actual_agno_tool_loop_with_native_schema_and_reasoning_metadata(
    tmp_path, food, contribution, monthly, daily
):
    # Mock HTTP, retaining the real SDK, adapter, tool execution, and schema parsing.
    requests = []
    arguments = {
        "scenario_label": "Synthetic fixture",
        "average_order_revenue_aed": 40,
        "food_cost_per_order_aed": food,
        "packaging_per_order_aed": 2,
        "other_variable_cost_per_order_aed": 1,
        "delivery_share": 0.5,
        "delivery_commission_rate": 0.25,
        "monthly_fixed_costs_aed": 30000,
        "operating_days_per_month": 30,
    }
    reasoning = [{"type": "reasoning.text", "text": "Fixture reasoning", "id": "reasoning-1"}]

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_details": reasoning,
                "tool_calls": [
                    {
                        "id": "call-calculator",
                        "type": "function",
                        "function": {
                            "name": "calculate_unit_economics",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "kind": "advice",
                        "cities": ["dubai"],
                        "headline": "Contribution calculation.",
                        "sections": [
                            {
                                "title": "Result",
                                "items": [
                                    {
                                        "text": f"Contribution is {contribution} AED per order.",
                                        "basis": "calculation",
                                        "evidence_ids": [],
                                        "calculation_ids": [
                                            json.loads(
                                                next(
                                                    m["content"]
                                                    for m in reversed(body["messages"])
                                                    if m["role"] == "tool"
                                                )
                                            )["data"]["calculation_id"]
                                        ],
                                    }
                                ],
                            }
                        ],
                        "assumptions": ["Synthetic inputs; not researched market figures."],
                        "next_steps": ["Check the supplied costs."],
                        "questions": [],
                    }
                ),
            }
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"fixture-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek/deepseek-v4.1-flash",
                "provider": "fixture-provider",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": 11 if len(requests) == 1 else 17,
                    "completion_tokens": 3 if len(requests) == 1 else 5,
                    "total_tokens": 14 if len(requests) == 1 else 22,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "cost": 0.001 if len(requests) == 1 else 0.002,
                },
            },
        )

    db_file = tmp_path / "sessions.db"
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        agent = create_agent("fixture-key", db_file)
        sessions = SessionStore(agent.db)
        session_id = sessions.new()
        sessions.update_profile(session_id, "monthly_rent_budget_aed", "12000 AED")
        agent.model.client = OpenAI(
            api_key="fixture-key", base_url="https://openrouter.ai/api/v1", http_client=http_client
        )
        result = answer_question("Calculate from the synthetic inputs.", session_id, agent)
        assert sessions.profile(session_id).monthly_rent_budget_aed == Decimal("12000")
        agent.db.close()

    assert result.status == RunStatus.completed
    assert isinstance(result.content, AdvisorAnswer)
    assert f"{contribution} AED" in result.content.sections[0].items[0].text
    assert f"| Contribution per order | {contribution}.00 | AED/order |" in result.markdown
    assert len(requests) == 2
    assert result.usage.model_requests == 2
    assert result.usage.input_tokens == 28
    assert result.usage.output_tokens == 8
    assert result.usage.reasoning_tokens == 2
    assert result.usage.cached_tokens == 4
    assert result.usage.reported_cost_usd == Decimal("0.003")
    assert result.usage.usage_complete
    assert result.record.provider_if_available == ["fixture-provider"]
    assert result.record.tool_events[0].name == "calculate_unit_economics"
    assert result.record.tool_events[0].result_count == 1
    assert result.record.tool_events[0].duration_ms >= 0
    records, unreadable = RunLog(tmp_path / "runs.jsonl").records(session_id)
    assert unreadable == 0
    assert records == [result.record]
    assert "Fixture reasoning" not in (tmp_path / "runs.jsonl").read_text()
    for request in requests:
        assert request["provider"]["require_parameters"] is True
        assert request["provider"]["order"] == ["Morph"]
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["strict"] is True
        assert request["model"] == "deepseek/deepseek-v4.1-flash"
    tool_message = next(message for message in requests[1]["messages"] if message["role"] == "tool")
    # Wrappers return JSON text, including the application-generated calculation ID.
    payload = json.loads(tool_message["content"])
    assert Decimal(payload["data"]["contribution_per_order_aed"]) == Decimal(contribution)
    assert payload["data"]["break_even_orders_month"] == monthly
    assert payload["data"]["break_even_orders_day"] == daily
    ref = payload["data"]["calculation_id"]
    assert result.calculations == {ref: payload}
    if monthly is None:
        assert payload["data"]["viability"] == "not_viable_under_inputs"
        assert "**Not viable under these inputs:**" in result.markdown
    assistant = next(message for message in requests[1]["messages"] if message.get("tool_calls"))
    assert assistant["reasoning_details"] == reasoning
    assert result.tools[0].tool_name == "calculate_unit_economics"
    assert result.tools[0].tool_args == arguments
    schema = next(
        t["function"]["parameters"]
        for t in requests[0]["tools"]
        if t["function"]["name"] == "calculate_unit_economics"
    )
    assert set(schema["required"]) == set(arguments)
    assert schema["properties"]["operating_days_per_month"]["type"] == "integer"

    # Reopen the actual SQLite database and check the complete tool result,
    # not just the model's narrative. This also covers nonviable calculations.
    reader = create_agent("fixture-key", db_file)
    try:
        saved = reader.get_session(session_id=session_id)
        assert saved.runs[0].status == RunStatus.completed
        assert json.loads(saved.runs[0].tools[0].result) == payload
        stored_answer = AdvisorAnswer.model_validate(saved.runs[0].content)
        assert stored_answer.sections[0].items[0].calculation_ids == [ref]
        restored = last_answer(session_id, reader)
        assert restored.markdown == result.markdown
        assert restored.calculations == result.calculations
        path = export_answer(restored, tmp_path / "exports")
        exported = path.read_text()
        assert "Synthetic inputs; not researched market figures." in exported
        assert "| Monthly fixed costs | 30000.00 | AED/month |" in exported
        assert f"| Contribution per order | {contribution}.00 | AED/order |" in exported
        if monthly is None:
            assert "Not viable under these inputs" in exported
        restored.calculations[ref]["data"]["contribution_per_order_aed"] = "999999"
        assert (
            last_answer(session_id, reader).calculations[ref]["data"]["contribution_per_order_aed"]
            == payload["data"]["contribution_per_order_aed"]
        )
    finally:
        reader.db.close()


@pytest.mark.parametrize("content", ["raw answer", None, {"answer": "unparsed dictionary"}])
def test_service_rejects_completed_untyped_content(content):
    agent = Mock()
    agent.run.return_value = RunOutput(status=RunStatus.completed, content=content)
    result = answer_question("Question", str(uuid4()), agent)
    assert result.error_category == "validation"
    assert result.content is None
    assert result.markdown == ""


def test_answer_schema_rejects_empty_headline_and_extra_fields():
    base = dict(
        kind="insufficient_evidence",
        cities=[],
        sections=[],
        assumptions=[],
        next_steps=[],
        questions=[],
    )
    for fields in [{"headline": "  "}, {"headline": "Text", "unexpected": True}]:
        with pytest.raises(ValidationError):
            AdvisorAnswer.model_validate({**base, **fields})


def test_renderer_uses_answer_fields():
    answer = AdvisorAnswer(
        kind="advice",
        cities=["dubai"],
        headline="Review the scenario.",
        sections=[
            AnswerSection(
                title="Scenario",
                items=[
                    Statement(
                        text="Use the supplied costs as assumptions.",
                        basis="assumption",
                        evidence_ids=[],
                        calculation_ids=[],
                    )
                ],
            )
        ],
        assumptions=["Hypothetical."],
        next_steps=["Check inputs."],
        questions=[],
    )
    output = render_answer(answer, TurnRegistry())
    assert "Review the scenario." in output
    assert "Hypothetical." in output
    assert "Check inputs." in output
    assert '"headline"' not in output
