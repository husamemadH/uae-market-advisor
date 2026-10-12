from pydantic import BaseModel, ConfigDict, Field


class AdvisorAnswer(BaseModel):
    """Typed final answer for the advisor's current capabilities."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=4000, description="Concise answer; use actual tool results for arithmetic.")
    assumptions: list[str] = Field(max_length=6, description="Input basis and limitations, including hypothetical scenarios.")
    next_steps: list[str] = Field(max_length=5, description="Practical proposed next actions; empty when unnecessary.")
    questions: list[str] = Field(max_length=2, description="Focused questions for missing inputs; empty when unnecessary.")


def render_answer(answer: AdvisorAnswer) -> str:
    """Render only validated final-answer fields, without raw JSON or reasoning."""
    parts = [answer.answer]
    for title, items in (
        ("Assumptions and limitations", answer.assumptions),
        ("Next steps", answer.next_steps),
        ("Questions", answer.questions),
    ):
        if items:
            parts.append(f"**{title}**\n\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(parts)
