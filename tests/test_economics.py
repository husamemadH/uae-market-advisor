from decimal import Decimal, localcontext
from inspect import signature
from uuid import UUID

import pytest

from economics import calculate_unit_economics


def calculate(
    revenue="40",
    food="12",
    packaging="2",
    other="1",
    share="0.5",
    commission="0.25",
    fixed="30000",
    days=30,
    label="Synthetic test",
):
    return calculate_unit_economics(
        label, revenue, food, packaging, other, share, commission, fixed, days
    )


def test_worked_example_returns_inputs_formulas_and_exact_results():
    result = calculate()
    assert result["status"] == "ok"
    data = result["data"]
    assert data["currency"] == "AED"
    assert data["viability"] == "positive_contribution"
    assert data["inputs"] == {
        "average_order_revenue_aed": "40",
        "food_cost_per_order_aed": "12",
        "packaging_per_order_aed": "2",
        "other_variable_cost_per_order_aed": "1",
        "delivery_share": "0.5",
        "delivery_commission_rate": "0.25",
        "monthly_fixed_costs_aed": "30000",
        "operating_days_per_month": 30,
    }
    assert Decimal(data["weighted_delivery_fee_per_order_aed"]) == 5
    assert Decimal(data["variable_cost_per_order_aed"]) == 20
    assert Decimal(data["contribution_per_order_aed"]) == 20
    assert Decimal(data["contribution_margin"]) == Decimal("0.5")
    assert data["break_even_orders_month"] == 1500
    assert data["break_even_orders_day"] == 50
    assert len(data["formulas"]) == 6
    assert (
        data["formulas"]["break_even_orders_month"]
        == "ceil(monthly_fixed_costs / contribution_per_order)"
    )
    UUID(data["calculation_id"].removeprefix("CALC-"))
    assert calculate()["data"]["calculation_id"] != data["calculation_id"]


def test_all_inputs_are_required_without_hidden_defaults():
    params = signature(calculate_unit_economics).parameters.values()
    assert all(param.default is param.empty for param in params)
    with pytest.raises(TypeError):
        calculate_unit_economics("Missing inputs", 40, 12, 2, 1)


def test_decimal_arithmetic_keeps_sub_cent_precision_until_display():
    data = calculate(revenue="40.005", share="0", fixed="30000.01")["data"]
    assert Decimal(data["contribution_per_order_aed"]) == Decimal("25.005")
    # Rounding contribution to 25.00 before division would incorrectly yield 1201.
    edge = calculate(revenue="40.004", share="0", fixed="30004")["data"]
    assert Decimal(edge["contribution_per_order_aed"]) == Decimal("25.004")
    assert edge["break_even_orders_month"] == 1200
    assert edge["break_even_orders_day"] == 40
    small = calculate(revenue="0.30", food="0.10", packaging="0.10", other="0", share="0")["data"]
    assert Decimal(small["contribution_per_order_aed"]) == Decimal("0.10")


def test_fractional_orders_are_rounded_up_for_month_and_day():
    data = calculate(fixed="30000.01")["data"]
    assert data["break_even_orders_month"] == 1501
    assert data["break_even_orders_day"] == 51


def test_break_even_boundary_and_math_do_not_depend_on_global_decimal_precision():
    with localcontext() as context:
        context.prec = 6
        data = calculate(fixed="30000.000000000000000000000001")["data"]
    assert data["break_even_orders_month"] == 1501
    assert data["break_even_orders_day"] == 51
    assert Decimal(data["contribution_per_order_aed"]) == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revenue": "0"},
        {"revenue": "-1"},
        {"food": "-1"},
        {"packaging": "-1"},
        {"other": "-1"},
        {"fixed": "-1"},
        {"share": "-0.1"},
        {"share": "1.01"},
        {"share": "50"},
        {"commission": "-0.1"},
        {"commission": "1.01"},
        {"commission": "25"},
        {"days": 0},
        {"days": 32},
        {"days": 1.5},
        {"days": 30.0},
        {"days": "30"},
        {"days": True},
        {"days": None},
        {"label": "  "},
        {"label": "x" * 101},
        {"label": None},
        {"revenue": "forty"},
        {"food": None},
        {"other": True},
    ],
)
def test_invalid_inputs_return_no_calculation(kwargs):
    result = calculate(**kwargs)
    assert result["status"] == "invalid_input"
    assert result["data"] is None
    assert result["warnings"]


@pytest.mark.parametrize(
    "field", ["revenue", "food", "packaging", "other", "share", "commission", "fixed"]
)
@pytest.mark.parametrize("value", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_every_amount_and_rate_must_be_finite(field, value):
    result = calculate(**{field: value})
    assert result["status"] == "invalid_input"
    assert result["data"] is None
    assert any("finite" in warning for warning in result["warnings"])


@pytest.mark.parametrize("revenue", ["15", "10"])
@pytest.mark.parametrize("fixed", ["30000", "0"])
def test_nonpositive_contribution_has_no_break_even_even_with_zero_fixed_costs(revenue, fixed):
    result = calculate(revenue=revenue, share="0", fixed=fixed)
    assert result["status"] == "ok"
    assert result["data"]["viability"] == "not_viable_under_inputs"
    assert Decimal(result["data"]["contribution_per_order_aed"]) <= 0
    assert result["data"]["break_even_orders_month"] is None
    assert result["data"]["break_even_orders_day"] is None
    assert "nonpositive" in result["warnings"][0]


@pytest.mark.parametrize(
    "share,commission,fee",
    [("0", "0.25", "0"), ("1", "0", "0"), ("1", "0.25", "10"), ("0.5", "1", "20")],
)
def test_delivery_mix_and_rate_boundaries(share, commission, fee):
    result = calculate(share=share, commission=commission)
    assert result["status"] == "ok"
    assert Decimal(result["data"]["weighted_delivery_fee_per_order_aed"]) == Decimal(fee)


@pytest.mark.parametrize("days", [1, 31])
def test_explicit_zero_costs_and_positive_contribution_need_zero_break_even_orders(days):
    data = calculate(
        food="0", packaging="0", other="0", share="0", commission="0", fixed="0", days=days
    )["data"]
    assert Decimal(data["contribution_per_order_aed"]) == 40
    assert Decimal(data["contribution_margin"]) == 1
    assert data["break_even_orders_month"] == data["break_even_orders_day"] == 0
