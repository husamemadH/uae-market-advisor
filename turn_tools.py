"""Agno tool wrappers and the records available to one answer."""

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agno.run import RunContext
from pydantic import StrictInt

from economics import calculate_unit_economics as calculate
from evidence import City, EvidenceItem, EvidenceStore, Topic

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
        delivery_share: Decimal,
        delivery_commission_rate: Decimal,
        monthly_fixed_costs_aed: Decimal,
        operating_days_per_month: StrictInt,
    ) -> str:
        """Calculate contribution, margin, and monthly/daily operating break-even.

        All eight numeric inputs are required; use explicit zero where applicable.
        Call only after all inputs are available. If any are missing, ask the user
        instead of calling. Never fill required arguments with placeholders, even
        labelled illustrative ones. Invent inputs only for a hypothetical example
        explicitly requested by the user, and identify every invented value.
        Revenue is positive AED/order, net of discounts and pass-through tax.
        Food, packaging and other variable costs are nonnegative averages per order.
        Delivery share and commission are fractions 0-1: 25 percent means 0.25.
        Commission applies to supplied revenue on delivery orders; exclude it from
        other variable costs. Monthly fixed costs must not double-count per-order
        costs. Operating days must be an integer 1-31. Label the input basis as
        user-provided, sourced, or hypothetical. Arithmetic does not verify demand.
        Cite calculation_id even for not_viable_under_inputs results: nonpositive
        contribution has null break-even values, and more volume does not fix it.
        """
        registry = get_registry(run_context)
        if not registry.take_call():
            return _limit_result(registry)
        result = calculate(
            scenario_label,
            average_order_revenue_aed,
            food_cost_per_order_aed,
            packaging_per_order_aed,
            other_variable_cost_per_order_aed,
            delivery_share,
            delivery_commission_rate,
            monthly_fixed_costs_aed,
            operating_days_per_month,
        )
        if result["status"] == "ok":
            calculation_id = result["data"]["calculation_id"]
            registry.calculations[calculation_id] = result
        registry.note_warnings(result["warnings"])
        return json.dumps(result)

    return [lookup_market_evidence, calculate_unit_economics]
