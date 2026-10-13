"""Explicit, validated business constraints; conversation never writes these fields."""

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

PROFILE_KEY = "business_profile"
EDITABLE_FIELDS = (
    "startup_budget_aed",
    "monthly_rent_budget_aed",
    "branch_format",
    "target_audience",
    "positioning",
    "planned_opening",
)


class BusinessProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    restaurant_origin: Literal["Jordan"] = "Jordan"
    core_products: tuple[Literal["meaty pizza"], Literal["kebab sandwiches"]] = (
        "meaty pizza",
        "kebab sandwiches",
    )
    target_cities: tuple[Literal["Dubai"], Literal["Abu Dhabi"]] = ("Dubai", "Abu Dhabi")
    startup_budget_aed: Decimal | None = Field(default=None, gt=0, max_digits=16, decimal_places=2)
    monthly_rent_budget_aed: Decimal | None = Field(
        default=None, ge=0, max_digits=16, decimal_places=2
    )
    branch_format: Literal["takeaway", "dine_in", "delivery_kitchen", "mixed"] | None = None
    target_audience: str | None = Field(default=None, min_length=1, max_length=200)
    positioning: Literal["value", "mid_market", "premium"] | None = None
    planned_opening: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("target_audience", "planned_opening")
    @classmethod
    def single_line_text(cls, value: str | None) -> str | None:
        if value is not None and any(not char.isprintable() for char in value):
            raise ValueError("Use a single line of printable text.")
        return value


def profile_from_state(state: dict | None) -> BusinessProfile:
    if state is not None and not isinstance(state, dict):
        raise ValueError("The saved business profile is invalid; it has not been changed.")
    try:
        return BusinessProfile.model_validate((state or {}).get(PROFILE_KEY, {}))
    except ValidationError as exc:
        raise ValueError("The saved business profile is invalid; it has not been changed.") from exc


def change_profile(profile: BusinessProfile, field: str, value: str | None) -> BusinessProfile:
    """Parse one explicit command. None means unset; monetary field names specify AED."""
    if field not in EDITABLE_FIELDS:
        raise ValueError("Supported profile fields: " + ", ".join(EDITABLE_FIELDS) + ".")
    if value is not None:
        value = value.strip()
        if field.endswith("_aed"):
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]{1,2})?)(?:\s+AED)?", value, re.IGNORECASE)
            if not match or len(match[1]) > 17:
                raise ValueError(
                    "Use an AED amount with at most two decimal places, e.g. 300000 AED. "
                    "Use at most 14 digits before the decimal point. "
                    "Other currencies, separators, and expressions are not accepted."
                )
            value = match[1]
    try:
        return BusinessProfile.model_validate({**profile.model_dump(), field: value})
    except ValidationError as exc:
        message = exc.errors(include_input=False)[0]["msg"]
        raise ValueError(f"Invalid {field}: {message}.") from exc


def render_profile(profile: BusinessProfile) -> str:
    lines = ["Business profile (explicitly saved constraints)"]
    for field, value in profile.model_dump().items():
        if value is None:
            display = "Unknown"
        elif isinstance(value, tuple):
            display = ", ".join(value)
        elif isinstance(value, Decimal):
            display = f"{value:.2f} AED"
        else:
            display = value
        lines.append(f"{field}: {display}")
    return "\n".join(lines)
