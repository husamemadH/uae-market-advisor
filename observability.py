"""Per-turn observations and local usage records, without model text or reasoning."""

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Literal

from agno.exceptions import ModelAuthenticationError, ModelProviderError
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import SQLAlchemyError

from sessions import SessionStorageError

ERROR_MESSAGES = {
    "authentication": "Authentication failed. Check OPENROUTER_API_KEY and model access.",
    "credits": "Insufficient credits. Check OpenRouter credits and your key spending limit.",
    "rate_limit": "Rate limit reached. Wait before retrying.",
    "timeout": "The model request timed out. You can retry the question.",
    "connection": "Could not connect to OpenRouter. Check your connection before retrying.",
    "model_configuration": "The model or requested parameters are unavailable. Check model access and configuration.",
    "provider": "OpenRouter could not complete the request. You can try again later.",
    "validation": "The answer could not be validated. Please try again.",
    "storage": "Could not save or read this conversation. Check local session storage before retrying.",
    "cancelled": "Request cancelled. You can ask another question.",
    "internal": "The request failed unexpectedly. You can try again; see the run ID for diagnostics.",
}


def error_category(exc: BaseException) -> str:
    """Use exception types and HTTP status, never provider message substring guesses."""
    chain, current = [], exc
    while current is not None and id(current) not in {id(item) for item in chain}:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(item, KeyboardInterrupt) for item in chain):
        return "cancelled"
    if any(isinstance(item, (SessionStorageError, SQLAlchemyError)) for item in chain):
        return "storage"
    if any(isinstance(item, (APITimeoutError, TimeoutError)) for item in chain):
        return "timeout"
    if any(isinstance(item, APIConnectionError) for item in chain):
        return "connection"
    if any(isinstance(item, (ValidationError, json.JSONDecodeError)) for item in chain):
        return "validation"
    for item in chain:
        status = getattr(item, "status_code", None)
        if status in {401, 403} or isinstance(item, ModelAuthenticationError):
            return "authentication"
        if status == 402:
            return "credits"
        if status == 429:
            return "rate_limit"
        if status in {408, 504}:
            return "timeout"
        if status in {400, 404, 422}:
            return "model_configuration"
    return "provider" if isinstance(exc, ModelProviderError) else "internal"


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RequestUsage(RecordModel):
    generation_id: str | None = None
    model_reported: str | None = None
    provider_reported: str | None = None
    duration_ms: float = 0
    outcome: Literal["completed", "error", "cancelled"] = "error"
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    reported_cost_usd: Decimal | None = Field(default=None, ge=0)


class UsageSummary(RecordModel):
    model_requests: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    reported_cost_usd: Decimal | None = None
    known_reported_cost_usd: Decimal | None = None
    estimated_cost_usd: Decimal | None = None
    estimate_basis: str | None = None
    usage_complete: bool = False


def summarize_usage(requests: list[RequestUsage]) -> UsageSummary:
    totals = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "reported_cost_usd",
    ):
        values = [getattr(request, name) for request in requests]
        totals[name] = (
            sum(values) if values and all(value is not None for value in values) else None
        )
    costs = [
        request.reported_cost_usd for request in requests if request.reported_cost_usd is not None
    ]
    return UsageSummary(
        **totals,
        model_requests=len(requests),
        known_reported_cost_usd=sum(costs) if costs else None,
        usage_complete=bool(requests)
        and all(
            totals[key] is not None
            for key in ("input_tokens", "output_tokens", "reported_cost_usd")
        ),
    )


def capture_usage(request: RequestUsage, response) -> None:
    """Read optional provider fields before Agno fills missing counters with zeros."""

    def label(value):
        return value if isinstance(value, str) and len(value) <= 200 else None

    request.generation_id = label(getattr(response, "id", None))
    request.model_reported = label(getattr(response, "model", None))
    request.provider_reported = label(getattr(response, "provider", None))
    usage = getattr(response, "usage", None)
    for target, obj, attribute in (
        ("input_tokens", usage, "prompt_tokens"),
        ("output_tokens", usage, "completion_tokens"),
        ("reasoning_tokens", getattr(usage, "completion_tokens_details", None), "reasoning_tokens"),
        ("cached_tokens", getattr(usage, "prompt_tokens_details", None), "cached_tokens"),
    ):
        value = getattr(obj, attribute, None)
        if type(value) is int and value >= 0:
            setattr(request, target, value)
    cost = getattr(usage, "cost", None)
    if isinstance(cost, (str, int, float, Decimal)) and not isinstance(cost, bool):
        try:
            parsed = Decimal(str(cost))
            if parsed.is_finite() and parsed >= 0:
                request.reported_cost_usd = parsed
        except InvalidOperation:
            pass


class ToolEvent(RecordModel):
    name: str
    duration_ms: float
    status: str
    result_count: int = 0


@dataclass
class TurnObservation:
    requests: list[RequestUsage] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    error: str | None = None
    storage_failed: bool = False


CURRENT_TURN: ContextVar[TurnObservation | None] = ContextVar("advisor_observation", default=None)


def timed_tool(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        start, status, count = perf_counter(), "error", 0
        try:
            result = function(*args, **kwargs)
            payload = json.loads(result)
            status = payload["status"]
            data = payload.get("data") or {}
            count = len(data.get("records", [])) + int("calculation_id" in data)
            return result
        except KeyboardInterrupt:
            status = "cancelled"
            raise
        finally:
            if trace := CURRENT_TURN.get():
                trace.tool_events.append(
                    ToolEvent(
                        name=function.__name__,
                        duration_ms=(perf_counter() - start) * 1000,
                        status=status,
                        result_count=count,
                    )
                )

    return wrapped


class RunRecord(UsageSummary):
    schema_version: int = 1
    run_id: str
    attempt_run_ids: list[str] = Field(default_factory=list)
    session_id: str
    timestamp_utc: datetime
    model_requested: str
    model_reported_if_available: list[str] | None = None
    provider_if_available: list[str] | None = None
    prompt_version: str
    evidence_dataset_version: str | None
    question: str
    outcome: Literal["completed", "error", "cancelled"]
    error_category: str | None = None
    tool_events: list[ToolEvent] = Field(default_factory=list)
    requests: list[RequestUsage] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    latency_ms: float
    tool_output_characters: int = 0
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunLogError(RuntimeError):
    pass


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: RunRecord) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(record.model_dump_json() + "\n")
        except OSError as exc:
            raise RunLogError(
                "Could not save the usage record; usage history is incomplete."
            ) from exc

    def records(self, session_id: str) -> tuple[list[RunRecord], int]:
        records, unreadable = [], 0
        try:
            with self.path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        record = RunRecord.model_validate_json(line)
                    except ValueError:
                        unreadable += 1
                        continue
                    if record.session_id == session_id:
                        records.append(record)
        except FileNotFoundError:
            return [], 0
        except (OSError, UnicodeError) as exc:
            raise RunLogError("Could not read usage history from local storage.") from exc
        return records, unreadable


def render_usage(records: list[RunRecord], unreadable: int = 0) -> str:
    if not records:
        text = "No recorded usage for this session. Usage from earlier, unlogged turns is unknown."
    else:
        latest = records[-1]
        combined = summarize_usage([request for record in records for request in record.requests])
        # A record with no observed requests is unknown, not a free turn.
        if any(not record.requests for record in records):
            for key in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "reported_cost_usd",
            ):
                setattr(combined, key, None)
            combined.usage_complete = False
        lines = [
            f"Recorded turns in this session: {len(records)}",
            "Includes failed and cancelled turns.",
        ]
        for title, usage in (
            (f"Latest run: {latest.run_id} ({latest.outcome})", latest),
            ("Recorded session total", combined),
        ):
            lines.extend(["", title, f"Observed model requests: {usage.model_requests}"])
            for label, field_name in (
                ("Input tokens", "input_tokens"),
                ("Output tokens", "output_tokens"),
                ("Reasoning tokens (part of output)", "reasoning_tokens"),
                ("Cached tokens (part of input)", "cached_tokens"),
            ):
                value = getattr(usage, field_name)
                lines.append(f"{label}: {value if value is not None else 'Unknown'}")
            cost = usage.reported_cost_usd
            lines.append(f"Reported cost (USD): {cost if cost is not None else 'Unknown'}")
            if cost is None and usage.known_reported_cost_usd is not None:
                lines.append(
                    f"Reported cost subtotal (USD): {usage.known_reported_cost_usd}; incomplete"
                )
            lines.append(f"Core usage complete: {'yes' if usage.usage_complete else 'no'}")
        lines.extend(
            [
                f"\nLatest latency: {latest.latency_ms / 1000:.2f} seconds",
                "Estimates: unavailable; no price basis configured.",
                "Totals cover recorded turns only; they are not an account spending cap.",
            ]
        )
        text = "\n".join(lines)
    if unreadable:
        text += f"\nWarning: {unreadable} unreadable log records were skipped; totals may be incomplete."
    return text
