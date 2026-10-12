from decimal import Decimal

import pytest

from economics import calculate_unit_economics


def calculate(revenue="40", food="12", packaging="2", other="1"):
    return calculate_unit_economics("Synthetic test", Decimal(revenue), Decimal(food), Decimal(packaging), Decimal(other))


def test_contribution_from_synthetic_inputs():
    result = calculate()
    assert result["status"] == "ok"
    assert result["data"]["contribution_per_order_aed"] == "25"
    assert result["data"]["variable_cost_per_order_aed"] == "15"
    assert result["data"]["inputs"]["average_order_revenue_aed"] == "40"


def test_decimal_arithmetic_keeps_precision_until_display():
    assert calculate("0.30", "0.10", "0.10", "0")["data"]["contribution_per_order_aed"] == "0.10"
    assert calculate("40.005", "12", "2", "1")["data"]["contribution_per_order_aed"] == "25.005"


@pytest.mark.parametrize("kwargs", [
    {"revenue": "0"}, {"revenue": "-1"}, {"food": "-1"},
    {"packaging": "-1"}, {"other": "NaN"}, {"revenue": "Infinity"},
])
def test_invalid_numbers_return_no_calculation(kwargs):
    result = calculate(**kwargs)
    assert result["status"] == "invalid_input"
    assert result["data"] is None
    assert result["warnings"]


@pytest.mark.parametrize("revenue", ["15", "10"])
def test_nonpositive_contribution_has_warning(revenue):
    result = calculate(revenue=revenue)
    assert Decimal(result["data"]["contribution_per_order_aed"]) <= 0
    assert "nonpositive" in result["warnings"][0]


def test_explicit_zero_costs_are_valid():
    assert calculate("40", "0", "0", "0")["data"]["contribution_per_order_aed"] == "40"
