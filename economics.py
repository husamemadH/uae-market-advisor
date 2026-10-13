from decimal import Decimal, DecimalException, localcontext
from typing import Any
from uuid import uuid4

# A delivery share of 0.5 means 50 of every 100 orders use delivery.


def calculate_unit_economics(
    scenario_label: str,
    average_order_revenue_aed: Decimal,
    food_cost_per_order_aed: Decimal,
    packaging_per_order_aed: Decimal,
    other_variable_cost_per_order_aed: Decimal,
    delivery_share: Decimal,
    delivery_commission_rate: Decimal,
    monthly_fixed_costs_aed: Decimal,
    operating_days_per_month: int,
) -> dict[str, Any]:
    """Calculate contribution, margin, and operating break-even for a scenario.

    All inputs are required. Amounts are AED, rates are fractions from 0 to 1,
    and operating days must be an integer from 1 to 31. A valid calculation
    has status ok even when its viability is not_viable_under_inputs.
    Decimal results are serialized as strings without rounding to cents.

    Args:
        scenario_label: Short name identifying this supplied scenario.
        average_order_revenue_aed: Positive revenue per order, after discounts
            and excluding pass-through tax.
        food_cost_per_order_aed: Nonnegative ingredient cost per order.
        packaging_per_order_aed: Nonnegative packaging cost per order.
        other_variable_cost_per_order_aed: Nonnegative other variable costs
            per order, excluding the delivery commission calculated here.
        delivery_share: Fraction of orders subject to the delivery commission.
        delivery_commission_rate: Commission fraction of supplied revenue on
            delivery orders. For example, 25 percent is 0.25.
        monthly_fixed_costs_aed: Nonnegative monthly fixed costs, excluding
            costs already counted per order.
        operating_days_per_month: Number of operating days from 1 through 31.
    """
    raw_inputs = {
        "average_order_revenue_aed": average_order_revenue_aed,
        "food_cost_per_order_aed": food_cost_per_order_aed,
        "packaging_per_order_aed": packaging_per_order_aed,
        "other_variable_cost_per_order_aed": other_variable_cost_per_order_aed,
        "delivery_share": delivery_share,
        "delivery_commission_rate": delivery_commission_rate,
        "monthly_fixed_costs_aed": monthly_fixed_costs_aed,
    }
    try:
        values = {name: Decimal(str(value)) for name, value in raw_inputs.items()}
    except (DecimalException, ValueError):
        return {
            "status": "invalid_input",
            "data": None,
            "warnings": ["Supply numeric amounts and rates."],
        }

    warnings = []
    if (
        not isinstance(scenario_label, str)
        or not scenario_label.strip()
        or len(scenario_label) > 100
    ):
        warnings.append("Supply a scenario label between 1 and 100 characters.")
    if type(operating_days_per_month) is not int or not 1 <= operating_days_per_month <= 31:
        warnings.append("operating_days_per_month must be an integer from 1 to 31.")
    for name, value in values.items():
        if not value.is_finite():
            warnings.append(f"{name} must be finite.")
        elif name in {"delivery_share", "delivery_commission_rate"}:
            if not 0 <= value <= 1:
                warnings.append(f"{name} must be a fraction from 0 to 1; use 0.25 for 25%.")
        elif value < 0 or (name == "average_order_revenue_aed" and value == 0):
            warnings.append(f"{name} must be positive for revenue or nonnegative for costs.")
    if warnings:
        return {"status": "invalid_input", "data": None, "warnings": warnings}

    # Allow enough precision for products and sums of the supplied decimals,
    # including sub-cent costs. Only the renderer rounds currency amounts.
    try:
        with localcontext() as context:
            context.prec = max(
                28,
                sum(len(v.as_tuple().digits) + abs(v.as_tuple().exponent) for v in values.values())
                + 8,
            )
            revenue = values["average_order_revenue_aed"]
            weighted_fee = revenue * values["delivery_share"] * values["delivery_commission_rate"]
            variable_cost = (
                values["food_cost_per_order_aed"]
                + values["packaging_per_order_aed"]
                + values["other_variable_cost_per_order_aed"]
                + weighted_fee
            )
            contribution = revenue - variable_cost
            margin = contribution / revenue
            monthly_orders = daily_orders = None
            if contribution > 0:
                # Quotient/remainder avoids rounding a division down to an
                # integer before applying the ceiling, even near a boundary.
                whole_orders, remainder = divmod(values["monthly_fixed_costs_aed"], contribution)
                monthly_orders = int(whole_orders) + (remainder != 0)
                daily_orders = (
                    monthly_orders + operating_days_per_month - 1
                ) // operating_days_per_month
    except (DecimalException, ValueError, OverflowError):
        return {
            "status": "invalid_input",
            "data": None,
            "warnings": ["Inputs exceed the calculator's supported decimal range."],
        }

    if contribution <= 0:
        warnings.append(
            "Contribution is nonpositive under these inputs; ordinary break-even is unavailable. "
            "Increasing volume alone does not fix the scenario."
        )
    return {
        "status": "ok",
        "data": {
            "calculation_id": f"CALC-{uuid4().hex}",
            "scenario_label": scenario_label.strip(),
            "currency": "AED",
            "inputs": {
                **{name: format(value, "f") for name, value in values.items()},
                "operating_days_per_month": operating_days_per_month,
            },
            "viability": "positive_contribution" if contribution > 0 else "not_viable_under_inputs",
            "weighted_delivery_fee_per_order_aed": format(weighted_fee, "f"),
            "variable_cost_per_order_aed": format(variable_cost, "f"),
            "contribution_per_order_aed": format(contribution, "f"),
            "contribution_margin": format(margin, "f"),
            "break_even_orders_month": monthly_orders,
            "break_even_orders_day": daily_orders,
            "formulas": {
                "weighted_delivery_fee": "revenue * delivery_share * delivery_commission_rate",
                "variable_cost": "food_cost + packaging + other_variable_cost + weighted_delivery_fee",
                "contribution_per_order": "revenue - variable_cost",
                "contribution_margin": "contribution_per_order / revenue",
                "break_even_orders_month": "ceil(monthly_fixed_costs / contribution_per_order)",
                "break_even_orders_day": "ceil(break_even_orders_month / operating_days_per_month)",
            },
        },
        "warnings": warnings,
    }
