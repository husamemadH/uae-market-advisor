import json
from datetime import date, timedelta
from pathlib import Path
from typing import get_args
from unittest.mock import Mock

import pytest

import app
from evidence import City, Topic, DEFAULT_EVIDENCE_FILE, MAX_PAYLOAD_CHARACTERS, EvidenceDataError, EvidenceStore


ON_DATE = date(2026, 10, 12)
FIXTURE_FILE = Path(__file__).parent / "fixtures" / "synthetic_evidence.json"


@pytest.fixture
def records():
    return json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def store():
    return EvidenceStore.load(FIXTURE_FILE, on_date=ON_DATE)


def load_records(tmp_path, records):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return EvidenceStore.load(path, on_date=ON_DATE)


def test_production_dataset_does_not_load_test_fixtures(records):
    store = EvidenceStore.load(DEFAULT_EVIDENCE_FILE)
    fixture_ids = {record["id"] for record in records}
    assert not fixture_ids.intersection(record.id for record in store.records)


def test_production_research_is_retrievable_for_each_city_and_topic():
    store = EvidenceStore.load(DEFAULT_EVIDENCE_FILE)
    for city in get_args(City):
        for topic in get_args(Topic):
            result = store.lookup_market_evidence([city], [topic])
            assert result.status == "ok", (city, topic)
            assert result.data.matched_count >= 2, (city, topic)
            assert result.data.records, (city, topic)
            assert len(result.model_dump_json()) <= MAX_PAYLOAD_CHARACTERS
            for record in result.data.records:
                assert city in record.cities and topic in record.topics
                assert record.locator and record.qualifiers
                assert record.source_url.host != "example.invalid"


def test_research_coverage_summary_matches_dataset_and_references():
    store = EvidenceStore.load(DEFAULT_EVIDENCE_FILE)
    report = json.loads(DEFAULT_EVIDENCE_FILE.with_name("research_coverage.json").read_text(encoding="utf-8"))
    by_id = {record.id: record for record in store.records}
    assert report["record_count"] == len(by_id)
    assert report["source_count"] == len({str(record.source_url) for record in store.records})
    for city in get_args(City):
        assert set(report["coverage"][city]) == set(get_args(Topic))
        for topic, coverage in report["coverage"][city].items():
            expected = {record.id for record in store.records if city in record.cities and topic in record.topics}
            assert coverage["count"] == len(expected)
            assert set(coverage["record_ids"]) == expected
        for group, reference_key in [("candidate_areas", "basis_record_ids"), ("competitor_examples", "record_ids")]:
            assert len(report[group][city]) >= 3
            for entry in report[group][city]:
                assert entry[reference_key]
                assert all(city in by_id[record_id].cities for record_id in entry[reference_key])
    for sample in report["manually_compared_samples"]:
        assert set(sample["record_ids"]) <= by_id.keys()
    assert report["known_gaps"]


def test_empty_dataset_reports_no_results(tmp_path):
    store = load_records(tmp_path, [])
    result = store.lookup_market_evidence(["dubai"], ["pricing"])
    assert result.status == "no_results"
    assert result.data.matched_count == 0


def test_city_and_topic_filters_keep_source_metadata(store):
    result = store.lookup_market_evidence(["abu_dhabi"], ["pricing"])
    assert [record.id for record in result.data.records] == ["SYN-AUH-PRICE-001"]
    record = result.data.records[0]
    assert record.checked_on == ON_DATE
    assert str(record.source_url) == "https://example.invalid/fixtures/abu-dhabi-menu"
    assert record.publisher == "Test fixture"
    assert record.qualifiers == ["Synthetic fixture; not market data."]
    assert record.needs_review is False


def test_keyword_ranking_is_case_insensitive_with_stable_ties(store):
    result = store.lookup_market_evidence(["dubai"], ["pricing"], "PIZZA TAKEAWAY")
    assert [record.id for record in result.data.records] == [
        "SYN-DXB-PRICE-001", "SYN-DXB-PRICE-003", "SYN-DXB-PRICE-004",
    ]
    assert store.lookup_market_evidence(["dubai"], ["pricing"], "pizza takeaway") == result


def test_both_city_lookup_balances_an_uneven_dataset(store):
    result = store.lookup_market_evidence(["dubai", "abu_dhabi"], ["pricing"], limit=2)
    assert [record.cities for record in result.data.records] == [["dubai"], ["abu_dhabi"]]
    assert result.data.matched_count == 5
    assert result.data.omitted_count == 3


def test_shared_city_record_is_not_duplicated(store):
    result = store.lookup_market_evidence(["dubai", "abu_dhabi"], ["operations", "suppliers"])
    assert {record.id for record in result.data.records} == {"SYN-BOTH-SUP-001", "SYN-AUH-OPS-001"}
    assert len(result.data.records) == 2


@pytest.mark.parametrize("topics,query", [(["marketing"], ""), (["pricing"], "no_matching_keyword")])
def test_no_matches_are_reported_honestly(store, topics, query):
    result = store.lookup_market_evidence(["dubai"], topics, query)
    assert result.status == "no_results"
    assert result.data.records == []
    assert result.data.omitted_count == 0


@pytest.mark.parametrize("changes", [
    {"cities": ["amman"]}, {"cities": []}, {"topics": ["unknown"]},
    {"topics": []}, {"limit": 0}, {"limit": 13}, {"limit": "8"}, {"limit": True},
])
def test_invalid_lookup_input_returns_an_envelope(store, changes):
    result = store.lookup_market_evidence(**{"cities": ["dubai"], "topics": ["pricing"], **changes})
    assert result.status == "invalid_input"
    assert result.data.records == []
    assert result.warnings


@pytest.mark.parametrize("changes", [
    {"cities": ["amman"]}, {"cities": []}, {"topics": ["unknown"]},
    {"cities": ["dubai", "dubai"]}, {"observation": "  "},
    {"source_url": "ftp://example.invalid/test"}, {"source_kind": "unknown"},
    {"checked_on": "2026-02-30"}, {"checked_on": "20261012"},
    {"checked_on": 1791763200}, {"checked_on": "2026-10-13"},
    {"published_on": "2026-10-13"}, {"unexpected_field": True},
])
def test_loader_rejects_invalid_records(tmp_path, records, changes):
    with pytest.raises(EvidenceDataError):
        load_records(tmp_path, [{**records[0], **changes}])


def test_loader_rejects_duplicate_ids(tmp_path, records):
    with pytest.raises(EvidenceDataError, match="Duplicate evidence ID"):
        load_records(tmp_path, [records[0], records[0]])


@pytest.mark.parametrize("content", ["not JSON", "{}", "null"])
def test_loader_rejects_malformed_dataset(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EvidenceDataError):
        EvidenceStore.load(path, on_date=ON_DATE)


@pytest.mark.parametrize("days,topics,source_kind,expected", [
    (30, ["pricing"], "business", False),
    (31, ["pricing"], "business", True),
    (31, ["competitors"], "editorial", True),
    (31, ["locations"], "platform", True),
    (180, ["locations"], "official", False),
    (181, ["locations"], "official", True),
])
def test_review_windows(tmp_path, records, days, topics, source_kind, expected):
    record = {**records[0], "topics": topics, "source_kind": source_kind, "published_on": None,
              "checked_on": (ON_DATE - timedelta(days=days)).isoformat()}
    store = load_records(tmp_path, [record])
    assert store.records[0].needs_review is expected
    result = store.lookup_market_evidence(["dubai"], topics)
    assert result.data.records[0].needs_review is expected
    assert bool(result.warnings) is expected


def test_rechecking_an_old_report_does_not_make_it_current(tmp_path, records):
    store = load_records(tmp_path, [{**records[0], "topics": ["marketing"], "source_kind": "official",
                                   "published_on": "2025-01-01", "checked_on": ON_DATE.isoformat()}])
    record = store.records[0]
    assert record.published_on == date(2025, 1, 1)
    assert record.checked_on == ON_DATE
    assert record.needs_review is True


@pytest.mark.parametrize("days_ago,expired", [(0, False), (1, True)])
def test_offer_expiry_keeps_historical_record_and_flags_it(tmp_path, records, days_ago, expired):
    store = load_records(tmp_path, [{**records[0], "valid_until": (ON_DATE - timedelta(days=days_ago)).isoformat()}])
    result = store.lookup_market_evidence(["dubai"], ["pricing"])
    assert result.status == "ok"
    assert result.data.records[0].expired is expired
    assert result.data.records[0].needs_review is expired


def test_default_and_maximum_record_limits(tmp_path, records):
    template = {**records[0], "entity": "Fixture", "observation": "Synthetic test.",
                "source_title": "Fixture", "qualifiers": [], "locator": None}
    store = load_records(tmp_path, [{**template, "id": f"SYN-{number:02d}"} for number in range(20)])
    for kwargs, count in [({}, 8), ({"limit": 12}, 12)]:
        result = store.lookup_market_evidence(["dubai"], ["pricing"], **kwargs)
        assert len(result.data.records) == count
        assert result.data.omitted_count == 20 - count
        assert len(result.model_dump_json()) <= MAX_PAYLOAD_CHARACTERS


def test_payload_limit_omits_whole_records(tmp_path, records):
    observation = "Synthetic " + "x" * 5000
    store = load_records(tmp_path, [{**records[0], "id": f"SYN-{number}", "observation": observation} for number in range(3)])
    result = store.lookup_market_evidence(["dubai"], ["pricing"])
    assert result.status == "ok"
    assert len(result.model_dump_json()) <= MAX_PAYLOAD_CHARACTERS
    assert len(result.data.records) == 1
    assert result.data.records[0].observation == observation
    assert result.data.matched_count == 3
    assert result.data.omitted_count == 2


def test_oversized_record_does_not_hide_smaller_matches(tmp_path, records):
    store = load_records(tmp_path, [{**records[0], "observation": "x" * 9000}, records[1]])
    result = store.lookup_market_evidence(["dubai"], ["pricing"])
    assert [record.id for record in result.data.records] == ["SYN-DXB-PRICE-002"]
    assert result.data.omitted_count == 1


def test_no_record_fits_returns_limit_reached(tmp_path, records):
    store = load_records(tmp_path, [{**records[0], "observation": "x" * 9000}])
    result = store.lookup_market_evidence(["dubai"], ["pricing"])
    assert result.status == "limit_reached"
    assert result.data.records == []
    assert result.data.matched_count == result.data.omitted_count == 1
    assert len(result.model_dump_json()) <= MAX_PAYLOAD_CHARACTERS


def test_corrupt_evidence_stops_startup_before_agent_creation(tmp_path, monkeypatch, capsys):
    path = tmp_path / "corrupt.json"
    path.write_text("not JSON", encoding="utf-8")
    monkeypatch.setattr(app, "DEFAULT_EVIDENCE_FILE", path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    builder = Mock()
    monkeypatch.setattr(app, "create_agent", builder)
    assert app.main() == 1
    builder.assert_not_called()
    assert "research dataset" in capsys.readouterr().err
