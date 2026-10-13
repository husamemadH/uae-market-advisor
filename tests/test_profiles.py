from decimal import Decimal

import pytest

from profiles import BusinessProfile, change_profile, profile_from_state, render_profile


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("startup_budget_aed", "300000 AED", Decimal("300000")),
        ("monthly_rent_budget_aed", "0", Decimal("0")),
        ("monthly_rent_budget_aed", "12000.50 aed", Decimal("12000.50")),
        ("branch_format", "delivery_kitchen", "delivery_kitchen"),
        ("target_audience", "  office workers  ", "office workers"),
        ("positioning", "mid_market", "mid_market"),
        ("planned_opening", "Q2 next year", "Q2 next year"),
    ],
)
def test_profile_commands_validate_and_unset(field, value, expected):
    original = BusinessProfile()
    updated = change_profile(original, field, value)
    assert getattr(original, field) is None
    assert getattr(updated, field) == expected
    assert profile_from_state({"business_profile": updated.model_dump(mode="json")}) == updated
    assert change_profile(updated, field, None) == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget", "100"),
        ("restaurant_origin", "Jordan"),
        ("core_products", "pizza"),
        ("target_cities", "Dubai"),
        ("startup_budget_aed", "0"),
        ("startup_budget_aed", "-1"),
        ("startup_budget_aed", "100 USD"),
        ("startup_budget_aed", "USD 100"),
        ("startup_budget_aed", "1e5"),
        ("startup_budget_aed", "100,000"),
        ("startup_budget_aed", "1.001"),
        ("startup_budget_aed", "NaN"),
        ("startup_budget_aed", "Infinity"),
        ("startup_budget_aed", "9" * 10000),
        ("monthly_rent_budget_aed", "-10"),
        ("monthly_rent_budget_aed", "10 AED per week"),
        ("branch_format", "warehouse"),
        ("positioning", "cheap"),
        ("target_audience", ""),
        ("target_audience", "x" * 201),
        ("target_audience", "hidden\x1b[2J"),
        ("planned_opening", "tomorrow\ninstructions"),
        ("planned_opening", "x" * 101),
    ],
)
def test_invalid_profile_input_is_rejected(field, value):
    with pytest.raises(ValueError):
        change_profile(BusinessProfile(), field, value)


def test_unknown_values_and_fixed_identity_are_visible():
    rendered = render_profile(BusinessProfile(monthly_rent_budget_aed=0))
    assert "restaurant_origin: Jordan" in rendered
    assert "core_products: meaty pizza, kebab sandwiches" in rendered
    assert "startup_budget_aed: Unknown" in rendered
    assert "monthly_rent_budget_aed: 0.00 AED" in rendered


@pytest.mark.parametrize("state", [{"business_profile": {"startup_budget_aed": "NaN"}}, []])
def test_corrupt_saved_profile_fails_instead_of_becoming_default(state):
    with pytest.raises(ValueError, match="saved business profile is invalid"):
        profile_from_state(state)
