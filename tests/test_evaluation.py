import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

import evaluate
from evaluation_fixtures import FixtureTransport, fixture_store


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("evaluation") / "fixture"
    transport = FixtureTransport()
    with (
        patch("evaluation_fixtures.FixtureTransport", return_value=transport),
        patch(
            "socket.create_connection", side_effect=AssertionError("Fixture mode must stay offline")
        ),
    ):
        evaluate.run_suite(output, mode="fixture")
    return output, transport


@pytest.mark.integration
def test_fixture_suite_uses_real_service_isolates_sessions_and_preserves_followup(fixture_run):
    output, transport = fixture_run
    artifacts = {case.id: read(output / f"{case.id}.json") for case in evaluate.load_cases()}
    assert all(item["status"] == "completed" for item in artifacts.values())
    assert len({item["session_id"] for key, item in artifacts.items() if key != "E11"}) == 11
    assert artifacts["E01"]["session_id"] == artifacts["E11"]["session_id"]
    assert artifacts["E11"]["profile_after"]["branch_format"] == "takeaway"
    assert artifacts["E10"]["checks"]["calculator_exact"] is True
    assert all(item["checks"]["reference_integrity"] for item in artifacts.values())
    for request in transport.requests:
        questions = [m["content"] for m in request["messages"] if m["role"] == "user"]
        if questions[-1] == artifacts["E11"]["case"]["question"]:
            assert questions == [artifacts["E01"]["case"]["question"], questions[-1]]
        else:
            assert len(questions) == 1
    summary = read(output / "summary.json")
    assert summary["executed"] == 12
    assert summary["session_sequence_valid"]
    assert summary["qualifying_reviews"] == 0
    assert not summary["executive_release_gate_passed"]
    assert summary["usage"]["reported_cost_usd"] is None
    assert all(row["score"] is None for row in summary["cases"])
    assert len(summary["coverage_by_requested_topic_and_answer_city"]["dubai"]) == 8
    assert len(summary["coverage_by_requested_topic_and_answer_city"]["abu_dhabi"]) == 8
    with pytest.raises(ValueError, match="cannot be scored"):
        evaluate.report(output, output / "absent-reviews.json")


@pytest.mark.integration
def test_followup_selection_includes_parent_and_existing_artifacts_cannot_be_overwritten(tmp_path):
    output = tmp_path / "selected"
    manifest = evaluate.run_suite(output, mode="fixture", selected=["E11"])
    assert manifest["execution_order"] == ["E01", "E11"]
    assert read(output / "E02.json")["reason"] == "not selected"
    with pytest.raises(FileExistsError):
        evaluate.run_suite(output, mode="fixture", selected=["E11"])


@pytest.mark.integration
def test_failed_prerequisite_skips_followup_but_keeps_other_cases(tmp_path):
    transport = FixtureTransport()
    original = transport.respond
    transport.respond = lambda req: (
        httpx.Response(401, json={"error": {"message": "fixture"}})
        if transport.case.id == "E01"
        else original(req)
    )
    with patch("evaluation_fixtures.FixtureTransport", return_value=transport):
        evaluate.run_suite(tmp_path / "failed", mode="fixture", selected=["E11", "E02"])
    assert read(tmp_path / "failed/E01.json")["error_category"] == "authentication"
    assert read(tmp_path / "failed/E11.json")["reason"] == "prerequisite did not complete"
    assert read(tmp_path / "failed/E02.json")["status"] == "completed"


@pytest.mark.parametrize(
    "cost", [Decimal(0), Decimal(-1), Decimal("NaN"), Decimal("Infinity"), None]
)
def test_live_mode_rejects_absent_or_invalid_budget_before_agent_creation(tmp_path, cost):
    with patch("evaluate.create_agent") as create:
        with pytest.raises(ValueError, match="positive"):
            evaluate.run_suite(
                tmp_path / "invalid", mode="live", api_key="fixture", max_cost_usd=cost
            )
        create.assert_not_called()
    assert not (tmp_path / "invalid").exists()


@pytest.mark.integration
@pytest.mark.parametrize("cost,reason", [("0.02", "threshold reached"), (None, "cost unknown")])
def test_live_budget_stops_before_next_turn_and_preserves_partial_results(
    tmp_path, monkeypatch, cost, reason
):
    original = evaluate.create_agent

    def create(*args, **kwargs):
        agent = original(*args, **kwargs)
        message = {
            "kind": "insufficient_evidence",
            "cities": ["dubai"],
            "headline": "Fixture",
            "sections": [],
            "assumptions": [],
            "next_steps": [],
            "questions": [],
        }
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "id": "fixture",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "fixture",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": json.dumps(message)},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 5,
                            "cost": float(cost) if cost else None,
                        },
                    },
                )
            )
        )
        agent.model.client = OpenAI(api_key="fixture", http_client=client, max_retries=0)
        return agent

    monkeypatch.setattr(evaluate, "create_agent", create)
    manifest = evaluate.run_suite(
        tmp_path / "budget", mode="live", api_key="fixture", max_cost_usd=Decimal("0.02")
    )
    assert manifest["execution_order"] == ["E01"]
    assert reason in manifest["stop_reason"]
    assert read(tmp_path / "budget/E01.json")["status"] == "completed"
    assert read(tmp_path / "budget/E11.json")["status"] == "not_run"
    assert reason in read(tmp_path / "budget/E11.json")["reason"]
    assert read(tmp_path / "budget/summary.json")["executed"] == 1


def test_reviews_require_matching_runs_and_human_review_and_cannot_override_hard_failures(
    fixture_run, tmp_path
):
    source, _ = fixture_run
    # Synthetic report inputs only; nothing in this test represents a real evaluation.
    for path in source.glob("*.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    manifest = read(tmp_path / "manifest.json")
    manifest["mode"] = "live"
    evaluate.write_json(tmp_path / "manifest.json", manifest)
    reviews = [
        {
            "case_id": case.id,
            "run_id": read(tmp_path / f"{case.id}.json")["run_id"],
            "reviewer": "synthetic test",
            "reviewer_type": "assistant_provisional",
            "scores": {name: 2 for name in evaluate.Scores.model_fields},
            "hard_failures": [],
            "claim_checks": ["Synthetic report test only"],
            "notes": "fixture",
            "improvement": "fixture",
        }
        for case in evaluate.load_cases()
    ]
    review_path = tmp_path / "reviews.json"
    evaluate.write_json(review_path, reviews)
    summary = evaluate.report(tmp_path, review_path)
    assert summary["qualifying_reviews"] == 12
    assert not summary["executive_release_gate_passed"]
    for review in reviews:
        review["reviewer_type"] = "human"
    evaluate.write_json(review_path, reviews)
    assert evaluate.report(tmp_path, review_path)["executive_release_gate_passed"]
    reviews[0]["hard_failures"] = ["unsupported factual claim"]
    evaluate.write_json(review_path, reviews)
    assert not evaluate.report(tmp_path, review_path)["executive_release_gate_passed"]
    reviews[0]["hard_failures"] = []
    evaluate.write_json(review_path, reviews)
    leaked = read(tmp_path / "E02.json")
    leaked["session_id"] = read(tmp_path / "E01.json")["session_id"]
    evaluate.write_json(tmp_path / "E02.json", leaked)
    summary = evaluate.report(tmp_path, review_path)
    assert not summary["session_sequence_valid"]
    assert not summary["executive_release_gate_passed"]
    reviews[0]["run_id"] = "unrelated-run"
    evaluate.write_json(review_path, reviews)
    with pytest.raises(ValueError, match="run ID"):
        evaluate.report(tmp_path, review_path)


@pytest.mark.parametrize("change", ["wrong_result", "missing_result", "wrong_input"])
def test_calculator_check_detects_wrong_or_missing_values(fixture_run, change):
    from types import SimpleNamespace

    from agno.run import RunStatus

    from answers import AdvisorAnswer

    output, _ = fixture_run
    artifact = read(output / "E10.json")
    calculation = next(iter(artifact["calculations"].values()))["data"]
    if change == "wrong_result":
        calculation["break_even_orders_day"] = 49
    elif change == "missing_result":
        calculation.pop("break_even_orders_day")
    else:
        calculation["inputs"]["food_cost_per_order_aed"] = "13"
    result = SimpleNamespace(
        status=RunStatus.completed,
        content=AdvisorAnswer.model_validate(artifact["answer"]),
        calculations=artifact["calculations"],
        sources={},
    )
    checks = evaluate.automatic_checks(evaluate.load_cases()[9], result, {}, {})
    assert checks["reference_integrity"]
    assert checks["calculator_exact"] is False


def test_old_contradictory_prices_keep_individual_context_and_need_review(tmp_path):
    from evidence import EvidenceStore

    original = (
        fixture_store().records[0].model_dump(mode="json", exclude={"needs_review", "expired"})
    )
    records = [
        {
            **original,
            "id": "OLD",
            "observation": "Small pizza pickup AED 20.",
            "checked_on": "2020-01-01",
            "valid_until": "2020-02-01",
        },
        {**original, "id": "NEW", "observation": "Large pizza delivery AED 40."},
    ]
    path = tmp_path / "conflicting.json"
    evaluate.write_json(path, records)
    result = EvidenceStore.load(path).lookup_market_evidence(["dubai"], ["pricing"], "pizza")
    assert len(result.data.records) == 2
    old = next(item for item in result.data.records if item.id == "OLD")
    assert old.expired and old.needs_review
    assert "Small" in old.observation
    assert result.warnings
