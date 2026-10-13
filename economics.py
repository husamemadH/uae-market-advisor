from decimal import Decimal, InvalidOperation
from typing import Any


def calculate_unit_economics(
    scenario_label: str,
    average_order_revenue_aed: Decimal,
    food_cost_per_order_aed: Decimal,
    packaging_per_order_aed: Decimal,
    other_variable_cost_per_order_aed: Decimal,
) -> dict[str, Any]:
    """Calculate contribution per order from explicit scenario inputs in AED.

    Use when the user supplies revenue and all three variable costs. Ask for
    missing inputs or clearly label an agreed hypothetical example. Costs
    must include any variable fees the user wants modeled; this early version
    does not model delivery mix or fixed costs and cannot calculate break-even.
    This tool computes supplied assumptions, not researched market figures.

    Args:
        scenario_label: Short name identifying this supplied scenario.
        average_order_revenue_aed: Positive revenue per order, after discounts
            and excluding pass-through tax.
        food_cost_per_order_aed: Nonnegative ingredient cost per order.
        packaging_per_order_aed: Nonnegative packaging cost per order.
        other_variable_cost_per_order_aed: Nonnegative other variable costs
            per order. Explicit zero is valid; do not silently assume zero.
    """
    raw_inputs = {
        "average_order_revenue_aed": average_order_revenue_aed,
        "food_cost_per_order_aed": food_cost_per_order_aed,
        "packaging_per_order_aed": packaging_per_order_aed,
        "other_variable_cost_per_order_aed": other_variable_cost_per_order_aed,
    }
    try:
        values = {name: Decimal(str(value)) for name, value in raw_inputs.items()}
    except (InvalidOperation, ValueError):
        return {"status": "invalid_input", "data": None, "warnings": ["Supply numeric AED inputs."]}

    warnings = []
    if not scenario_label.strip() or len(scenario_label) > 100:
        warnings.append("Supply a scenario label between 1 and 100 characters.")
    for name, value in values.items():
        if not value.is_finite():
            warnings.append(f"{name} must be finite.")
        elif value < 0 or (name == "average_order_revenue_aed" and value == 0):
            warnings.append(f"{name} must be positive for revenue or nonnegative for costs.")
    if warnings:
        return {"status": "invalid_input", "data": None, "warnings": warnings}

    revenue = values["average_order_revenue_aed"]
    variable_cost = sum(
        value for name, value in values.items() if name != "average_order_revenue_aed"
    )
    contribution = revenue - variable_cost
    if contribution <= 0:
        warnings.append(
            "Contribution is nonpositive under these inputs; volume alone does not fix it."
        )
    return {
        "status": "ok",
        "data": {
            "scenario_label": scenario_label.strip(),
            "currency": "AED",
            "inputs": {name: format(value, "f") for name, value in values.items()},
            "variable_cost_per_order_aed": format(variable_cost, "f"),
            "contribution_per_order_aed": format(contribution, "f"),
            "formula": "revenue - food_cost - packaging - other_variable_cost",
        },
        "warnings": warnings,
    }
