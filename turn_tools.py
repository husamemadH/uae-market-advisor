"""Agno tool wrappers and the records available to one answer."""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import uuid4

from agno.run import RunContext

from economics import calculate_unit_economics as calculate
from evidence import City, Topic, EvidenceItem, EvidenceStore

MAX_TOOL_CALLS = 6
REGISTRY_KEY = "advisor_turn"


@dataclass
class TurnRegistry:
    evidence: dict[str, EvidenceItem] = field(default_factory=dict)
    calculations: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tool_calls: int = 0
    validation_error: str | None = None

    def take_call(self) -> bool:
        if self.tool_calls >= MAX_TOOL_CALLS:
            return False
        self.tool_calls += 1
        return True

    def note_warnings(self, warnings: list[str]) -> None:
        for warning in warnings:
            if warning not in self.warnings:
                self.warnings.append(warning)


def get_registry(run_context: RunContext) -> TurnRegistry:
    registry = (run_context.dependencies or {}).get(REGISTRY_KEY)
    if not isinstance(registry, TurnRegistry):
        raise ValueError("Missing turn context. Run the advisor through answer_question.")
    return registry


def _limit_result(registry: TurnRegistry) -> str:
    warnings = ["The per-question tool limit has been reached; disclose remaining evidence gaps."]
    registry.note_warnings(warnings)
    return json.dumps({"status": "limit_reached", "data": None, "warnings": warnings})


def build_tools(store: EvidenceStore) -> list:
    def lookup_market_evidence(
        run_context: RunContext,
        cities: list[City],
        topics: list[Topic],
        query: str = "",
        limit: int = 8,
    ) -> str:
        """Retrieve curated research for this question, including factual follow-ups.

        Filter city and topic, then match query words against entity/observation.
        Use a short entity or subject query; use an empty query for broad coverage.
        Maximum 12 records, with complete JSON limited to 8,000 characters.
        Cite returned IDs only. Read dates, qualifiers, and freshness flags.
        No matches means missing research, not evidence that something is absent.
        """
        registry = get_registry(run_context)
        if not registry.take_call():
            return _limit_result(registry)
        result = store.lookup_market_evidence(cities, topics, query, limit)
        # Register only records actually sent back, never omitted matches.
        for record in result.data.records:
            registry.evidence[record.id] = record
        registry.note_warnings(result.warnings)
        return result.model_dump_json()

    def calculate_unit_economics(
        run_context: RunContext,
        scenario_label: str,
        average_order_revenue_aed: Decimal,
        food_cost_per_order_aed: Decimal,
        packaging_per_order_aed: Decimal,
        other_variable_cost_per_order_aed: Decimal,
    ) -> str:
        """Calculate contribution per order from explicit AED scenario inputs.

        All four amounts are required, including explicit zero costs. Revenue
        must be positive and net of discounts/pass-through tax; costs must be
        nonnegative. Ask for missing inputs. Label hypothetical inputs clearly.
        This models per-order variable costs only, not fixed costs or break-even.
        Cite the returned calculation_id for calculations in the final answer.
        """
        registry = get_registry(run_context)
        if not registry.take_call():
            return _limit_result(registry)
        result = calculate(scenario_label, average_order_revenue_aed, food_cost_per_order_aed,
                           packaging_per_order_aed, other_variable_cost_per_order_aed)
        if result["status"] == "ok":
            calculation_id = f"CALC-{uuid4().hex}"
            result["data"]["calculation_id"] = calculation_id
            registry.calculations[calculation_id] = result["data"]
        registry.note_warnings(result["warnings"])
        return json.dumps(result)

    return [lookup_market_evidence, calculate_unit_economics]
