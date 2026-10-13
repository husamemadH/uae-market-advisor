"""Structured advice, reference validation, and a shared Markdown renderer."""

import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from evidence import City
from turn_tools import TurnRegistry

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=700)]
Reference = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class AnswerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Statement(AnswerModel):
    text: Text
    basis: Literal["fact", "inference", "proposal", "assumption", "calculation", "research_lead"]
    evidence_ids: list[Reference] = Field(max_length=12)
    calculation_ids: list[Reference] = Field(max_length=6)


class AnswerSection(AnswerModel):
    title: str = Field(min_length=1, max_length=120)
    items: list[Statement] = Field(min_length=1, max_length=6)


class AdvisorAnswer(AnswerModel):
    """The model supplies statements and IDs; the app supplies source metadata."""

    kind: Literal["advice", "clarification", "insufficient_evidence", "out_of_scope"]
    cities: list[City] = Field(max_length=2)
    headline: str = Field(min_length=1, max_length=300)
    sections: list[AnswerSection] = Field(max_length=6)
    assumptions: list[Text] = Field(max_length=6)
    next_steps: list[Text] = Field(max_length=5)
    questions: list[Text] = Field(max_length=2)

    @model_validator(mode="after")
    def meaningful_answer(self):
        if len(set(self.cities)) != len(self.cities):
            raise ValueError("Cities must be unique.")
        if self.kind == "advice" and (not self.cities or not self.sections or not self.next_steps):
            raise ValueError("Advice requires cities, substantive sections, and a next step.")
        if self.kind == "clarification" and not self.questions:
            raise ValueError("Clarification requires a question.")
        return self


class AnswerValidationError(ValueError):
    def __init__(self, category: str):
        self.category = category
        super().__init__("The answer could not be validated against this question's tool results. Please try again.")


def validate_answer(answer: AdvisorAnswer, registry: TurnRegistry) -> None:
    if not isinstance(answer, AdvisorAnswer):
        raise AnswerValidationError("untyped_answer")
    texts = [answer.headline, *answer.assumptions, *answer.next_steps, *answer.questions]
    for section in answer.sections:
        texts.append(section.title)
        for item in section.items:
            texts.append(item.text)
            for record_id in item.evidence_ids:
                record = registry.evidence.get(record_id)
                if record is None:
                    raise AnswerValidationError("unretrieved_evidence")
                if not set(record.cities).intersection(answer.cities):
                    raise AnswerValidationError("evidence_city_mismatch")
            if any(ref not in registry.calculations for ref in item.calculation_ids):
                raise AnswerValidationError("unregistered_calculation")
            if item.basis == "fact" and not item.evidence_ids:
                raise AnswerValidationError("fact_without_evidence")
            if item.basis == "calculation" and not item.calculation_ids:
                raise AnswerValidationError("calculation_without_result")
            # Search is a later milestone. Curated records cannot masquerade as snippets.
            if item.basis == "research_lead":
                raise AnswerValidationError("search_not_available")
    # Model text is plain text; only the renderer is allowed to create source links.
    if any(re.search(r"https?://|www\.", text, re.IGNORECASE) for text in texts):
        raise AnswerValidationError("model_supplied_url")


def _plain(text: str) -> str:
    """Keep model text and source labels from creating Markdown links or markup."""
    return re.sub(r"([\\`*_{}\[\]<>()#!|~])", r"\\\1", " ".join(text.splitlines()))


def _money(value: str) -> str:
    amount = Decimal(value)
    # Avoid the default decimal context rejecting otherwise valid large inputs.
    from decimal import localcontext
    with localcontext() as context:
        context.prec = max(28, len(amount.as_tuple().digits) + abs(amount.adjusted()) + 4)
        return format(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def render_answer(answer: AdvisorAnswer, registry: TurnRegistry) -> str:
    """Validate, then render references and exact calculator values from this turn."""
    validate_answer(answer, registry)
    evidence_ids = list(dict.fromkeys(ref for section in answer.sections for item in section.items for ref in item.evidence_ids))
    calculation_ids = list(dict.fromkeys(ref for section in answer.sections for item in section.items for ref in item.calculation_ids))
    # Deduplicate pages while retaining which observations, dates, and limits were cited.
    urls = list(dict.fromkeys(str(registry.evidence[ref].source_url) for ref in evidence_ids))
    numbers = {url: number for number, url in enumerate(urls, 1)}
    calc_numbers = {ref: number for number, ref in enumerate(calculation_ids, 1)}
    parts = [_plain(answer.headline)]
    for section in answer.sections:
        lines = [f"**{_plain(section.title)}**"]
        for item in section.items:
            suffix = ""
            for url in dict.fromkeys(str(registry.evidence[ref].source_url) for ref in item.evidence_ids):
                suffix += f" [{numbers[url]}](<{url}>)"
            for ref in dict.fromkeys(item.calculation_ids):
                suffix += f" (Calculation {calc_numbers[ref]})"
            lines.append(f"- **{item.basis.replace('_', ' ').capitalize()}:** {_plain(item.text)}{suffix}")
        parts.append("\n\n".join([lines[0], "\n".join(lines[1:])]))
    for ref, number in calc_numbers.items():
        data = registry.calculations[ref]
        labels = {
            "average_order_revenue_aed": "Revenue per order",
            "food_cost_per_order_aed": "Food cost",
            "packaging_per_order_aed": "Packaging",
            "other_variable_cost_per_order_aed": "Other variable costs",
        }
        rows = [f"| {label} | {_money(data['inputs'][key])} |" for key, label in labels.items()]
        rows.extend([
            f"| Total variable costs | {_money(data['variable_cost_per_order_aed'])} |",
            f"| Contribution per order | {_money(data['contribution_per_order_aed'])} |",
        ])
        parts.append(f"**Calculation {number}: {_plain(data['scenario_label'])}**\n\n"
                     "| Input / result | AED |\n| --- | ---: |\n" + "\n".join(rows))
    for title, items in (
        ("Assumptions and limitations", answer.assumptions),
        ("Next steps", answer.next_steps),
        ("Questions", answer.questions),
        ("Tool warnings", registry.warnings),
    ):
        if items:
            parts.append(f"**{title}**\n\n" + "\n".join(f"- {_plain(item)}" for item in items))
    if urls:
        sources = []
        for url, number in numbers.items():
            records = [registry.evidence[ref] for ref in evidence_ids if str(registry.evidence[ref].source_url) == url]
            first = records[0]
            sources.append(f"{number}. [{_plain(first.source_title)}](<{url}>) — {_plain(first.publisher)}")
            for record in records:
                dates = f"checked {record.checked_on.isoformat()} UTC"
                if record.published_on:
                    dates += f"; published {record.published_on.isoformat()}"
                if record.valid_until:
                    dates += f"; valid until {record.valid_until.isoformat()}"
                flags = "; EXPIRED" if record.expired else ""
                if record.needs_review:
                    flags += "; NEEDS REVIEW"
                sources.append(f"   - {_plain(record.entity)}: {dates}{flags}.")
                if record.locator:
                    sources.append(f"     Location in source: {_plain(record.locator)}.")
                for qualifier in record.qualifiers:
                    sources.append(f"     - {_plain(qualifier)}")
        parts.append("**Sources**\n\n" + "\n".join(sources))
    return "\n\n".join(parts)
