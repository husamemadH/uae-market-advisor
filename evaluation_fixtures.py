"""Scripted HTTP responses for the evaluation harness, never advisory-quality evidence."""

import json
from datetime import date
from typing import get_args

import httpx

from evidence import EvidenceItem, EvidenceStore, Topic

CALCULATOR_INPUTS = {
    "scenario_label": "Synthetic evaluation scenario",
    "average_order_revenue_aed": 40,
    "food_cost_per_order_aed": 12,
    "packaging_per_order_aed": 2,
    "other_variable_cost_per_order_aed": 1,
    "delivery_share": 0.5,
    "delivery_commission_rate": 0.25,
    "monthly_fixed_costs_aed": 30000,
    "operating_days_per_month": 30,
}


def fixture_store() -> EvidenceStore:
    return EvidenceStore(
        records=tuple(
            EvidenceItem(
                id=f"FIXTURE-{city}",
                cities=[city],
                topics=list(get_args(Topic)),
                entity="Synthetic evaluation fixture",
                observation=f"Synthetic {city} observation; not market research.",
                source_title="Offline harness fixture",
                source_url=f"https://example.com/{city}",
                publisher="Fixture",
                source_kind="business",
                checked_on=date.today(),
                needs_review=False,
                expired=False,
            )
            for city in ("dubai", "abu_dhabi")
        )
    )


class FixtureTransport:
    """Execute real Agno tools/SQLite, replacing only the provider's HTTP boundary."""

    def __init__(self):
        self.case = None
        self.requests = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        case = self.case
        # Prior-turn tool messages must not be mistaken for this turn's results.
        last_user = max(i for i, m in enumerate(body["messages"]) if m["role"] == "user")
        tool_messages = [m for m in body["messages"][last_user + 1 :] if m["role"] == "tool"]
        if not tool_messages:
            name = "calculate_unit_economics" if case.id == "E10" else "lookup_market_evidence"
            arguments = (
                CALCULATOR_INPUTS
                if case.id == "E10"
                else {"cities": case.cities, "topics": case.topics or ["locations"], "query": ""}
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"fixture-{case.id}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
        else:
            data = json.loads(tool_messages[-1]["content"])["data"]
            item = {
                "text": "Synthetic harness answer; do not score as live advisory quality.",
                "basis": "calculation" if case.id == "E10" else "fact",
                "evidence_ids": [r["id"] for r in data.get("records", [])],
                "calculation_ids": [data["calculation_id"]] if case.id == "E10" else [],
            }
            answer = {
                "kind": "advice",
                "cities": case.cities,
                "headline": "OFFLINE FIXTURE ONLY",
                "sections": [{"title": "Fixture", "items": [item]}],
                "assumptions": ["Synthetic fixture, not a real model response."],
                "next_steps": ["Run live evaluation and review claim support."],
                "questions": [],
            }
            message = {"role": "assistant", "content": json.dumps(answer)}
        return httpx.Response(
            200,
            json={
                "id": f"fixture-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture-script",
                "provider": "offline-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if not tool_messages else "stop",
                    }
                ],
                # Omitted usage stays unknown; fixture data must never masquerade as measured usage.
            },
        )
