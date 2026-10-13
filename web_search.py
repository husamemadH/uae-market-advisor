"""Bounded Agno web search. Returned snippets are uninspected research leads."""

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from evidence import City

MAX_SEARCH_RESULTS = 4
MAX_SNIPPET_CHARACTERS = 600
SEARCH_TIMEOUT_SECONDS = 10
MAX_SEARCH_PAYLOAD_CHARACTERS = 8000
MAX_RAW_RESPONSE_CHARACTERS = 128000
LEAD_WARNING = (
    "Search snippets are untrusted research leads, not inspected pages or verified market facts."
)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=300)
    city: City | None = None


class SearchRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    source_kind: Literal["search_snippet"] = "search_snippet"
    source_title: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl
    snippet: str = Field(max_length=MAX_SNIPPET_CHARACTERS)
    query: str
    city: City | None
    fetched_at: AwareDatetime


class SearchPayload(BaseModel):
    records: list[SearchRecord]


class SearchResult(BaseModel):
    status: Literal["ok", "no_results", "invalid_input", "unavailable", "limit_reached"]
    data: SearchPayload
    warnings: list[str]


def _empty(status: str, warning: str) -> SearchResult:
    return SearchResult(status=status, data=SearchPayload(records=[]), warnings=[warning])


def _provider_search(query: str) -> str:
    # Keep the toolkit inside the worker. DDGS can try multiple backends and its
    # thread-pool cleanup can outlast its request timeout. The parent bounds both.
    from agno.tools.websearch import WebSearchTools

    toolkit = WebSearchTools(
        enable_search=True,
        enable_news=False,
        fixed_max_results=MAX_SEARCH_RESULTS,
        timeout=SEARCH_TIMEOUT_SECONDS,
        verify_ssl=True,
    )
    return toolkit.web_search(query)


def _fetch(query: str) -> str:
    # No shell, credentials, profile, or model conversation in the command/query.
    # subprocess.run kills and waits for the child on timeout or interruption.
    response = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        input=json.dumps({"query": query}),
        capture_output=True,
        text=True,
        timeout=SEARCH_TIMEOUT_SECONDS,
        check=True,
    )
    return response.stdout


@dataclass(frozen=True)
class MarketSearch:
    enabled: bool = True

    def search_market_web(self, query: str, city: City | None = None) -> SearchResult:
        if not self.enabled:
            return _empty(
                "unavailable", "Live search is disabled; use curated evidence and disclose gaps."
            )
        try:
            request = SearchRequest(query=query, city=city)
        except ValidationError:
            return _empty(
                "invalid_input", "Use a 3-300 character query and city dubai, abu_dhabi, or null."
            )

        # Send only the short topic query plus public geography, never chat history.
        context = {"dubai": "Dubai UAE", "abu_dhabi": "Abu Dhabi UAE", None: "UAE"}[request.city]
        scoped_query = f"{request.query} {context}"
        try:
            raw = _fetch(scoped_query)
        except (subprocess.TimeoutExpired, TimeoutError):
            return _empty(
                "unavailable", "Live search timed out; use curated evidence and disclose gaps."
            )
        except (OSError, subprocess.SubprocessError):
            return _empty(
                "unavailable", "Live search failed; use curated evidence and disclose gaps."
            )
        if not isinstance(raw, str) or len(raw) > MAX_RAW_RESPONSE_CHARACTERS:
            return _empty("unavailable", "Live search returned an invalid or oversized response.")
        try:
            payload = json.loads(raw)
        except (ValueError, RecursionError):
            return _empty(
                "unavailable", "Live search returned malformed data; use curated evidence."
            )
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            if isinstance(payload, dict) and payload.get("status") == "no_results":
                return _empty(
                    "no_results",
                    "No search results found; this does not establish that something is absent.",
                )
            return _empty(
                "unavailable", "Live search is unavailable; use curated evidence and disclose gaps."
            )
        hits = payload.get("results")
        if not isinstance(hits, list):
            return _empty(
                "unavailable", "Live search returned malformed results; use curated evidence."
            )
        if not hits:
            return _empty(
                "no_results",
                "No search results found; this does not establish that something is absent.",
            )

        records = []
        urls = set()
        warnings = [LEAD_WARNING]
        skipped = trimmed = False
        fetched_at = datetime.now(timezone.utc)
        for hit in hits:
            if len(records) == MAX_SEARCH_RESULTS:
                skipped = True
                break
            if not isinstance(hit, dict):
                skipped = True
                continue
            title, url, snippet = hit.get("title"), hit.get("href"), hit.get("body", "")
            if (
                not isinstance(title, str)
                or not title.strip()
                or not isinstance(snippet, str)
                or not isinstance(url, str)
                or len(url) > 2000
                or any(char.isspace() or char in '<>"\\' for char in url)
            ):
                skipped = True
                continue
            try:
                record = SearchRecord(
                    id=f"SEARCH-{uuid4().hex}",
                    source_title=title.strip()[:200],
                    source_url=url,
                    snippet=snippet.strip()[:MAX_SNIPPET_CHARACTERS],
                    query=scoped_query,
                    city=request.city,
                    fetched_at=fetched_at,
                )
            except ValidationError:
                skipped = True
                continue
            if (
                record.source_url.username
                or record.source_url.password
                or str(record.source_url) in urls
            ):
                skipped = True
                continue
            trimmed |= len(title.strip()) > 200 or len(snippet.strip()) > MAX_SNIPPET_CHARACTERS
            urls.add(str(record.source_url))
            records.append(record)
        if not records:
            return _empty(
                "unavailable",
                "Search returned no usable records; use curated evidence and disclose gaps.",
            )
        if skipped:
            warnings.append(
                "Some results were omitted because they were malformed, duplicated, or beyond the result limit."
            )
        if trimmed:
            warnings.append(
                "Long titles/snippets were shortened; inspect the source page before relying on them."
            )
        result = SearchResult(status="ok", data=SearchPayload(records=records), warnings=warnings)
        if len(result.model_dump_json()) > MAX_SEARCH_PAYLOAD_CHARACTERS:
            result.warnings.append("Some search records were omitted to respect the payload limit.")
            while (
                result.data.records
                and len(result.model_dump_json()) > MAX_SEARCH_PAYLOAD_CHARACTERS
            ):
                result.data.records.pop()
            if not result.data.records:
                result.status = "limit_reached"
        return result


def _worker() -> None:
    from ddgs.exceptions import DDGSException

    try:
        request = json.loads(sys.stdin.read())
        results = json.loads(_provider_search(request["query"]))
        response = {"status": "ok", "results": results}
    except DDGSException as exc:
        # DDGS 9.16 reports an empty search using this exception message.
        response = {"status": "no_results" if str(exc) == "No results found." else "unavailable"}
    except Exception:
        # Provider diagnostics may contain request details; don't relay them.
        response = {"status": "unavailable"}
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    _worker()
