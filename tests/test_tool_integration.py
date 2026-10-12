import ast
import json
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from agno.run import RunStatus
from agno.run.agent import RunOutput
from openai import OpenAI
from pydantic import ValidationError

from advisor import answer_question, create_agent
from answers import AdvisorAnswer, render_answer


def test_actual_agno_tool_loop_with_native_schema_and_reasoning_metadata(tmp_path):
    # Mock HTTP, retaining the real SDK, adapter, tool execution, and schema parsing.
    requests = []
    arguments = {
        "scenario_label": "Synthetic fixture",
        "average_order_revenue_aed": 40,
        "food_cost_per_order_aed": 12,
        "packaging_per_order_aed": 2,
        "other_variable_cost_per_order_aed": 1,
    }
    reasoning = [{"type": "reasoning.text", "text": "Fixture reasoning", "id": "reasoning-1"}]

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant", "content": None,
                "reasoning_details": reasoning,
                "tool_calls": [{"id": "call-calculator", "type": "function", "function": {
                    "name": "calculate_unit_economics", "arguments": json.dumps(arguments),
                }}],
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": json.dumps({
                "answer": "Contribution is 25 AED per order.",
                "assumptions": ["Synthetic inputs; excludes fixed costs."],
                "next_steps": [], "questions": [],
            })}
            finish_reason = "stop"
        return httpx.Response(200, json={
            "id": f"fixture-{len(requests)}", "object": "chat.completion", "created": 0,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        agent = create_agent("fixture-key", tmp_path / "sessions.db")
        agent.model.client = OpenAI(api_key="fixture-key", base_url="https://openrouter.ai/api/v1", http_client=http_client)
        result = answer_question("Calculate from the synthetic inputs.", str(uuid4()), agent)

    assert result.status == RunStatus.completed
    assert isinstance(result.content, AdvisorAnswer)
    assert "25 AED" in result.content.answer
    assert len(requests) == 2
    for request in requests:
        assert request["provider"]["require_parameters"] is True
        assert request["provider"]["order"] == ["Morph"]
        assert request["response_format"]["type"] == "json_schema"
        assert request["response_format"]["json_schema"]["strict"] is True
        assert request["model"] == "deepseek/deepseek-v4.1-flash"
    tool_message = next(message for message in requests[1]["messages"] if message["role"] == "tool")
    # Agno serializes a dictionary tool return with str(), rather than JSON.
    assert ast.literal_eval(tool_message["content"])["data"]["contribution_per_order_aed"] == "25"
    assistant = next(message for message in requests[1]["messages"] if message.get("tool_calls"))
    assert assistant["reasoning_details"] == reasoning
    assert result.tools[0].tool_name == "calculate_unit_economics"
    assert result.tools[0].tool_args == arguments


@pytest.mark.parametrize("content", ["raw answer", None, {"answer": "unparsed dictionary"}])
def test_service_rejects_completed_untyped_content(content):
    agent = Mock()
    agent.run.return_value = RunOutput(status=RunStatus.completed, content=content)
    with pytest.raises(ValueError, match="could not be validated"):
        answer_question("Question", str(uuid4()), agent)


def test_answer_schema_rejects_empty_answer_and_extra_fields():
    for answer in [{"answer": "  "}, {"answer": "Text", "unexpected": True}]:
        with pytest.raises(ValidationError):
            AdvisorAnswer.model_validate({"assumptions": [], "next_steps": [], "questions": [], **answer})


def test_renderer_uses_answer_fields():
    answer = AdvisorAnswer(answer="25 AED per order.", assumptions=["Hypothetical."], next_steps=["Check inputs."], questions=[])
    output = render_answer(answer)
    assert "25 AED per order." in output
    assert "Hypothetical." in output
    assert "Check inputs." in output
    assert '"answer"' not in output
