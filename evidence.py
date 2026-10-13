import json
import re
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

City = Literal["dubai", "abu_dhabi"]
Topic = Literal[
    "locations",
    "competitors",
    "menu",
    "pricing",
    "marketing",
    "delivery",
    "operations",
    "suppliers",
]
DEFAULT_EVIDENCE_FILE = Path(__file__).resolve().parent / "data" / "market_evidence.json"
MAX_PAYLOAD_CHARACTERS = 8000
CITY_ORDER: tuple[City, ...] = ("dubai", "abu_dhabi")


class EvidenceDataError(ValueError):
    """The research dataset cannot be read or does not satisfy its contract."""


class EvidenceRecord(BaseModel):
    """A factual observation and the source information needed to inspect it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1)
    cities: list[City] = Field(min_length=1)
    topics: list[Topic] = Field(min_length=1)
    entity: str = Field(min_length=1)
    observation: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: HttpUrl
    publisher: str = Field(min_length=1)
    source_kind: Literal["official", "business", "platform", "editorial"]
    published_on: date | None = None
    checked_on: date
    locator: str | None = None
    qualifiers: list[str] = Field(default_factory=list)
    valid_until: date | None = None

    @field_validator("cities", "topics")
    @classmethod
    def unique_tags(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("City and topic tags must not contain duplicates.")
        return values

    @field_validator("published_on", "checked_on", "valid_until", mode="before")
    @classmethod
    def iso_dates(cls, value):
        if value is None or type(value) is date:
            return value
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Use an ISO date in YYYY-MM-DD format.")
        return date.fromisoformat(value)

    @model_validator(mode="after")
    def publication_before_check(self):
        if self.published_on and self.published_on > self.checked_on:
            raise ValueError("Publication date cannot be later than the checked date.")
        return self


class EvidenceItem(EvidenceRecord):
    """Loaded source record with freshness flags calculated by the application."""

    needs_review: bool
    expired: bool


class LookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cities: list[City] = Field(min_length=1)
    topics: list[Topic] = Field(min_length=1)
    query: str = ""
    limit: int = Field(default=8, ge=1, le=12)


class EvidencePayload(BaseModel):
    records: list[EvidenceItem]
    matched_count: int
    omitted_count: int


class EvidenceResult(BaseModel):
    status: Literal["ok", "no_results", "invalid_input", "limit_reached"]
    data: EvidencePayload
    warnings: list[str]


def _review_flags(record: EvidenceRecord, on_date: date) -> tuple[bool, bool]:
    # Menu/price/competitor records and business/platform location records
    # use the branch/menu review window. Other context uses 180 days.
    fast_changing = (
        bool(set(record.topics) & {"menu", "pricing", "competitors"})
        or record.valid_until is not None
        or ("locations" in record.topics and record.source_kind in {"business", "platform"})
    )
    review_days = 30 if fast_changing else 180
    expired = record.valid_until is not None and record.valid_until < on_date
    # Rechecking an old published report does not make the report current.
    oldest_date = min(record.checked_on, record.published_on or record.checked_on)
    return (on_date - oldest_date).days > review_days or expired, expired


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _result(records: list[EvidenceItem], matched_count: int, status="ok") -> EvidenceResult:
    omitted_count = matched_count - len(records)
    warnings = []
    if omitted_count:
        warnings.append("Some matching records were omitted to respect result or payload limits.")
    if any(record.needs_review for record in records):
        warnings.append(
            "Some returned records need review; inspect their dates and freshness flags."
        )
    return EvidenceResult(
        status=status,
        data=EvidencePayload(
            records=records, matched_count=matched_count, omitted_count=omitted_count
        ),
        warnings=warnings,
    )


@dataclass(frozen=True)
class EvidenceStore:
    records: tuple[EvidenceItem, ...]

    @classmethod
    def load(
        cls, path: Path = DEFAULT_EVIDENCE_FILE, *, on_date: date | None = None
    ) -> "EvidenceStore":
        """Validate all records and derive freshness using today's date or a supplied test date."""
        on_date = on_date or date.today()
        try:
            raw_records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceDataError(f"Could not read a valid research dataset at {path}.") from exc
        try:
            records = TypeAdapter(list[EvidenceRecord]).validate_python(raw_records)
        except ValidationError as exc:
            error = exc.errors()[0]
            location = ".".join(str(part) for part in error["loc"]) or "dataset"
            raise EvidenceDataError(
                f"Invalid research data at {path}, {location}: {error['msg']}"
            ) from exc

        seen = set()
        loaded = []
        for record in records:
            if record.id in seen:
                raise EvidenceDataError(f"Duplicate evidence ID: {record.id}")
            if record.checked_on > on_date:
                raise EvidenceDataError(f"Evidence {record.id} has a checked date in the future.")
            seen.add(record.id)
            needs_review, expired = _review_flags(record, on_date)
            loaded.append(
                EvidenceItem(**record.model_dump(), needs_review=needs_review, expired=expired)
            )
        return cls(records=tuple(loaded))

    def lookup_market_evidence(
        self, cities: list[City], topics: list[Topic], query: str = "", limit: int = 8
    ) -> EvidenceResult:
        """Match any requested city/topic, then rank entity/observation token overlap.

        A nonempty query requires at least one shared token. Both-city requests
        interleave city results, deduplicating records that cover both cities.
        The complete result serialized with model_dump_json stays within 8,000
        characters. Oversized records are omitted whole so smaller matches can
        still be returned. This function performs no network or model requests.
        """
        try:
            request = LookupRequest(cities=cities, topics=topics, query=query, limit=limit)
        except ValidationError as exc:
            return EvidenceResult(
                status="invalid_input",
                data=EvidencePayload(records=[], matched_count=0, omitted_count=0),
                warnings=[f"Invalid lookup input: {exc.errors()[0]['msg']}"],
            )

        query_tokens = _tokens(request.query)
        scored = []
        for record in self.records:
            if not set(record.cities).intersection(request.cities) or not set(
                record.topics
            ).intersection(request.topics):
                continue
            score = len(query_tokens & _tokens(f"{record.entity} {record.observation}"))
            if query_tokens and score == 0:
                continue
            scored.append((score, record))
        ranked = [
            record for score, record in sorted(scored, key=lambda item: (-item[0], item[1].id))
        ]
        if not ranked:
            return _result([], 0, status="no_results")

        queues = {
            city: deque(record for record in ranked if city in record.cities)
            for city in CITY_ORDER
            if city in request.cities
        }
        selected = []
        seen = set()
        while any(queues.values()) and len(selected) < request.limit:
            for queue in queues.values():
                while queue and queue[0].id in seen:
                    queue.popleft()
                if not queue or len(selected) == request.limit:
                    continue
                record = queue.popleft()
                seen.add(record.id)
                candidate = _result([*selected, record], len(ranked))
                if len(candidate.model_dump_json()) <= MAX_PAYLOAD_CHARACTERS:
                    selected.append(record)
        return _result(selected, len(ranked), status="ok" if selected else "limit_reached")
