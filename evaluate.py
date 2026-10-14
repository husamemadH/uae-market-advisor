"""Reproducible executive cases through the existing application service."""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Literal, get_args
from uuid import uuid4

from agno.run import RunStatus
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from advisor import MODEL_ID, PROJECT_DIR, answer_question, create_agent
from answers import AdvisorAnswer, AnswerValidationError, validate_answer
from evidence import City, Topic
from observability import summarize_usage
from sessions import SessionStorageError, SessionStore
from settings import Settings
from turn_tools import MAX_SEARCH_CALLS, MAX_TOOL_OUTPUT_CHARACTERS, TurnRegistry

CASE_FILE = PROJECT_DIR / "evaluation" / "cases.json"


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    cities: list[City]
    topics: list[Topic]
    profile: dict[str, str] = Field(default_factory=dict)
    followup_of: str | None = None
    expected: list[str]


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relevance: StrictInt = Field(ge=0, le=2)
    evidence_support: StrictInt = Field(ge=0, le=2)
    actionability: StrictInt = Field(ge=0, le=2)
    business_reasoning: StrictInt = Field(ge=0, le=2)
    clarity: StrictInt = Field(ge=0, le=2)


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    run_id: str
    reviewer: str
    reviewer_type: Literal["human", "assistant_provisional"]
    scores: Scores
    hard_failures: list[str]
    claim_checks: list[str] = Field(min_length=1)
    notes: str
    improvement: str


def load_cases() -> list[EvaluationCase]:
    cases = [EvaluationCase.model_validate(item) for item in json.loads(CASE_FILE.read_text())]
    ids = [case.id for case in cases]
    if ids != [f"E{i:02}" for i in range(1, 13)]:
        raise ValueError("Expected the twelve ordered executive cases E01–E12.")
    for case in cases:
        if case.followup_of and case.followup_of not in ids[: ids.index(case.id)]:
            raise ValueError("A follow-up must reference an earlier case.")
    return cases


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint() -> dict:
    files = sorted(PROJECT_DIR.glob("*.py")) + [CASE_FILE, PROJECT_DIR / "uv.lock"]
    digests = {str(p.relative_to(PROJECT_DIR)): sha256(p.read_bytes()).hexdigest() for p in files}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "git_commit": commit,
        "file_sha256": digests,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("agno", "openai", "pydantic", "ddgs")},
    }


def automatic_checks(case: EvaluationCase, result, profile_before, profile_after) -> dict:
    """Only deterministic contracts; none of these checks establishes semantic support."""
    completed = result.status == RunStatus.completed
    integrity = None
    if completed:
        try:
            validate_answer(
                result.content,
                TurnRegistry(
                    evidence=result.sources,
                    calculations=result.calculations,
                ),
            )
            integrity = True
        except AnswerValidationError:
            integrity = False
    checks = {
        "reference_integrity": integrity,
        "requested_cities_present": set(case.cities) <= set(result.content.cities)
        if completed
        else None,
        "durable_profile_unchanged": profile_before == profile_after,
        "calculator_exact": None,
        "claim_support": "requires review",
    }
    if case.id == "E10" and completed:
        expected = {
            "weighted_delivery_fee_per_order_aed": "5",
            "variable_cost_per_order_aed": "20",
            "contribution_per_order_aed": "20",
            "contribution_margin": "0.5",
            "break_even_orders_month": "1500",
            "break_even_orders_day": "50",
        }
        expected_inputs = {
            "average_order_revenue_aed": "40",
            "food_cost_per_order_aed": "12",
            "packaging_per_order_aed": "2",
            "other_variable_cost_per_order_aed": "1",
            "delivery_share": "0.5",
            "delivery_commission_rate": "0.25",
            "monthly_fixed_costs_aed": "30000",
            "operating_days_per_month": "30",
        }
        cited = {
            ref
            for section in result.content.sections
            for item in section.items
            for ref in item.calculation_ids
        }
        try:
            checks["calculator_exact"] = any(
                ref in cited
                and all(
                    Decimal(str(payload["data"].get(key))) == Decimal(value)
                    for key, value in expected.items()
                )
                and all(
                    Decimal(str(payload["data"]["inputs"].get(key))) == Decimal(value)
                    for key, value in expected_inputs.items()
                )
                for ref, payload in result.calculations.items()
            )
        except (InvalidOperation, KeyError, TypeError):
            checks["calculator_exact"] = False
    return checks


def capture_case(case, session_id, result, before, after) -> dict:
    cited = (
        sorted(
            {
                ref
                for section in result.content.sections
                for item in section.items
                for ref in item.evidence_ids
            }
        )
        if result.content
        else []
    )
    return {
        "case": case.model_dump(),
        "status": result.status.value.lower(),
        "session_id": session_id,
        "run_id": result.run_id,
        "profile_before": before,
        "profile_after": after,
        "answer": result.content.model_dump(mode="json") if result.content else None,
        "rendered_answer": result.markdown,
        "cited_evidence_ids": cited,
        "retrieved_sources": {
            key: item.model_dump(mode="json") for key, item in result.sources.items()
        },
        "calculations": result.calculations,
        "tool_trace": [
            {
                "name": tool.tool_name,
                "arguments": tool.tool_args,
                "result": tool.result,
                "error": tool.tool_call_error,
            }
            for tool in result.tools
        ],
        "record": result.record.model_dump(mode="json") if result.record else None,
        "warnings": result.warnings,
        "error_category": result.error_category,
        "error_message": result.error_message,
        "checks": automatic_checks(case, result, before, after),
    }


def run_suite(
    output: Path,
    *,
    mode: str,
    selected: list[str] | None = None,
    max_cost_usd: Decimal | None = None,
    api_key: str | None = None,
    search_enabled: bool = True,
) -> dict:
    cases = load_cases()
    wanted = set(selected) if selected else {case.id for case in cases}
    if not wanted <= {case.id for case in cases}:
        raise ValueError("Unknown case ID.")
    if mode not in {"fixture", "live"}:
        raise ValueError("Choose fixture or live mode.")
    if mode == "live" and (
        not api_key or max_cost_usd is None or not max_cost_usd.is_finite() or max_cost_usd <= 0
    ):
        raise ValueError("Live mode requires an API key and a finite positive --max-cost-usd.")
    for case in reversed(cases):
        if case.id in wanted and case.followup_of:
            wanted.add(case.followup_of)
    # Run designed follow-ups directly after their parent, including selected reruns.
    ordered = [case for case in cases if not case.followup_of]
    for case in cases:
        if case.followup_of:
            index = next(i for i, parent in enumerate(ordered) if parent.id == case.followup_of)
            ordered.insert(index + 1, case)
    output.mkdir(parents=True, exist_ok=False)
    transport, http_client, evidence_store = None, None, None
    if mode == "fixture":
        import httpx
        from openai import OpenAI

        from evaluation_fixtures import FixtureTransport, fixture_store

        transport, evidence_store = FixtureTransport(), fixture_store()
        http_client = httpx.Client(transport=httpx.MockTransport(transport.respond))
    agent = create_agent(
        api_key if mode == "live" else "fixture-key",
        output / "sessions.db",
        evidence_store=evidence_store,
        settings=Settings(search_enabled=search_enabled if mode == "live" else False),
    )
    if transport:
        agent.model.client = OpenAI(api_key="fixture-key", http_client=http_client, max_retries=0)
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_cases": sorted(wanted),
        "execution_order": [],
        "configuration": {
            "model": MODEL_ID,
            "max_tokens": agent.model.max_tokens,
            "timeout_seconds": agent.model.timeout,
            "sdk_retries": 0,
            "agent_retries": agent.retries,
            "provider_routing": agent.model.extra_body,
            "history_turns": agent.num_history_runs,
            "tool_call_limit": agent.tool_call_limit,
            "search_enabled": search_enabled if mode == "live" else False,
            "search_call_limit": MAX_SEARCH_CALLS,
            "tool_output_characters": MAX_TOOL_OUTPUT_CHARACTERS,
            "max_cost_usd": str(max_cost_usd) if max_cost_usd is not None else None,
            "budget_policy": "Stop between turns at reported cost threshold or unknown cost; not a hard billing cap.",
            **agent.metadata,
        },
        **fingerprint(),
    }
    artifacts = {
        case.id: {
            "case": case.model_dump(),
            "status": "not_run",
            "reason": "not selected" if case.id not in wanted else "pending",
        }
        for case in cases
    }
    for case_id, artifact in artifacts.items():
        write_json(output / f"{case_id}.json", artifact)
    write_json(output / "manifest.json", manifest)
    sessions, requests, cost, stopped = SessionStore(agent.db), [], Decimal(0), None
    try:
        for case in ordered:
            if case.id not in wanted:
                continue
            if stopped:
                artifacts[case.id]["reason"] = stopped
            elif case.followup_of and artifacts[case.followup_of]["status"] != "completed":
                artifacts[case.id]["reason"] = "prerequisite did not complete"
            else:
                if case.followup_of:
                    session_id = artifacts[case.followup_of]["session_id"]
                else:
                    session_id = sessions.new()
                    for key, value in case.profile.items():
                        sessions.update_profile(session_id, key, value)
                before = sessions.profile(session_id).model_dump(mode="json")
                if transport:
                    transport.case = case
                print(f"{case.id}: running ({mode})", flush=True)
                result = answer_question(case.question, session_id, agent)
                after = sessions.profile(session_id).model_dump(mode="json")
                artifacts[case.id] = capture_case(case, session_id, result, before, after)
                # Keep rejected structured content for diagnosis, never raw provider errors/reasoning.
                # It remains separate from the validated answer and is never rendered as successful.
                if result.status != RunStatus.completed:
                    try:
                        saved = agent.db.get_run(result.run_id)
                        rejected = AdvisorAnswer.model_validate(saved.content if saved else None)
                        artifacts[case.id]["rejected_answer"] = rejected.model_dump(mode="json")
                    except (ValueError, SessionStorageError):
                        pass
                manifest["execution_order"].append(case.id)
                if result.record:
                    requests.extend(result.record.requests)
                if mode == "live":
                    if result.usage.reported_cost_usd is None:
                        stopped = "cost unknown; stopped before another paid turn"
                    else:
                        cost += result.usage.reported_cost_usd
                        if cost >= max_cost_usd:
                            stopped = "reported cost threshold reached"
                if result.status == RunStatus.cancelled:
                    stopped = "cancelled"
                print(
                    f"{case.id}: {artifacts[case.id]['status']}; reported USD {result.usage.reported_cost_usd}",
                    flush=True,
                )
            write_json(output / f"{case.id}.json", artifacts[case.id])
            write_json(output / "manifest.json", manifest)
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["usage"] = summarize_usage(requests).model_dump(mode="json")
        manifest["stop_reason"] = stopped
        write_json(output / "manifest.json", manifest)
        agent.db.close()
        if agent.model.client:
            agent.model.client.close()
        if http_client:
            http_client.close()
        report(output)
    return manifest


def report(output: Path, reviews_path: Path | None = None) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    artifacts = {
        case.id: json.loads((output / f"{case.id}.json").read_text()) for case in load_cases()
    }
    reviews = {}
    if reviews_path:
        if manifest["mode"] != "live":
            raise ValueError("Fixture outputs cannot be scored as advisory quality.")
        for raw in json.loads(reviews_path.read_text()):
            review = Review.model_validate(raw)
            artifact = artifacts.get(review.case_id, {})
            if review.case_id in reviews or review.run_id != artifact.get("run_id"):
                raise ValueError("Reviews must uniquely match the saved case and run ID.")
            reviews[review.case_id] = review
    rows, qualifying, human_qualifying, hard_failures = [], 0, 0, []
    for case_id, artifact in artifacts.items():
        review = reviews.get(case_id)
        checks = artifact.get("checks", {})
        failures = [name for name, value in checks.items() if value is False]
        if review:
            failures.extend(review.hard_failures)
        total = sum(review.scores.model_dump().values()) if review else None
        passed = (
            artifact["status"] == "completed" and total is not None and total >= 8 and not failures
        )
        qualifying += passed
        human_qualifying += passed and review.reviewer_type == "human"
        hard_failures.extend(f"{case_id}: {failure}" for failure in failures)
        rows.append(
            {
                "case": case_id,
                "status": artifact["status"],
                "score": total,
                "reviewer_type": review.reviewer_type if review else "pending",
                "qualifies": passed,
                "failures": failures,
            }
        )
    executed = [item for item in artifacts.values() if item["status"] != "not_run"]
    independent = [item["session_id"] for item in executed if not item["case"]["followup_of"]]
    isolated = len(independent) == len(set(independent))
    if artifacts["E11"]["status"] != "not_run":
        isolated &= artifacts["E11"]["session_id"] == artifacts["E01"].get("session_id")
    if not isolated:
        hard_failures.append("session isolation/sequence failure")
    coverage = {
        city: sorted(
            {
                topic
                for item in executed
                if item["status"] == "completed"
                and city in item["case"]["cities"]
                and city in item["answer"]["cities"]
                for topic in item["case"]["topics"]
            }
        )
        for city in ("dubai", "abu_dhabi")
    }
    coverage_complete = all(set(topics) == set(get_args(Topic)) for topics in coverage.values())
    all_reviewed = len(reviews) == 12 and all(r.reviewer_type == "human" for r in reviews.values())
    gate = (
        manifest["mode"] == "live"
        and all_reviewed
        and human_qualifying >= 10
        and not hard_failures
        and coverage_complete
        and len(executed) == 12
    )
    summary = {
        "mode": manifest["mode"],
        "cases": rows,
        "executed": len(executed),
        "coverage_by_requested_topic_and_answer_city": coverage,
        "coverage_note": "Tags and city presence only; substantive topic coverage requires review.",
        "session_sequence_valid": isolated,
        "qualifying_reviews": qualifying,
        "qualifying_human_reviews": human_qualifying,
        "hard_failures": hard_failures,
        "executive_release_gate_passed": gate,
        "review_status": "human review complete" if all_reviewed else "human review pending",
        "usage": manifest.get("usage"),
    }
    write_json(output / "summary.json", summary)
    lines = [
        f"Executive evaluation — {manifest['mode']}",
        "Fixture runs check plumbing only. Assistant scores are provisional, not human acceptance.",
        "Case  Outcome     Score  Reviewer",
    ]
    lines.extend(
        f"{r['case']}   {r['status']:<11} {str(r['score']) if r['score'] is not None else '—':<6} {r['reviewer_type']}"
        for r in rows
    )
    lines += [
        f"Qualifying reviewed cases: {qualifying}/12; human: {human_qualifying}/12",
        f"Executive release gate passed: {gate}; {summary['review_status']}",
        "Reference integrity does not establish factual support.",
        f"Hard failures: {json.dumps(hard_failures)}",
        f"Usage: {json.dumps(manifest.get('usage'))}",
    ]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    run.add_argument(
        "--case", action="append", dest="selected", choices=[case.id for case in load_cases()]
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--max-cost-usd", type=Decimal)
    run.add_argument("--no-search", action="store_true")
    render = commands.add_parser("report")
    render.add_argument("output", type=Path)
    render.add_argument("--reviews", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            report(args.output, args.reviews)
        else:
            load_dotenv(PROJECT_DIR / ".env")
            output = args.output or PROJECT_DIR / "runtime" / "evaluations" / (
                f"{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
            )
            run_suite(
                output,
                mode=args.mode,
                selected=args.selected,
                max_cost_usd=args.max_cost_usd,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                search_enabled=Settings.from_environment().search_enabled and not args.no_search,
            )
            print(f"Saved evaluation: {output}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evaluation: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
