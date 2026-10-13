from io import StringIO
from unittest.mock import Mock
from uuid import uuid4

import pytest
from agno.models.response import ToolExecution
from agno.run import RunStatus
from agno.run.agent import RunOutput
from agno.session.agent import AgentSession
from rich.console import Console

import app
from advisor import ApplicationResult, answer_sources, export_answer, last_answer
from answers import AdvisorAnswer, AnswerSection, Statement
from evidence import EvidenceStore


def sample_answer(**updates):
    data = dict(
        kind="insufficient_evidence",
        cities=["dubai"],
        headline="Synthetic answer.",
        sections=[],
        assumptions=["Hypothetical inputs."],
        next_steps=[],
        questions=[],
    )
    return AdvisorAnswer(**{**data, **updates})


def test_export_revalidates_and_renders_instead_of_trusting_display_text(tmp_path):
    result = ApplicationResult(
        status=RunStatus.completed, content=sample_answer(), markdown="UNVALIDATED_TEXT"
    )
    folder = tmp_path / "exports"
    first, second = export_answer(result, folder), export_answer(result, folder)
    assert first != second
    assert first.parent == folder
    assert first.suffix == ".md"
    assert "UNVALIDATED_TEXT" not in first.read_text()
    assert "Hypothetical inputs." in first.read_text()
    assert answer_sources(result) == "This answer cites no sources."


@pytest.mark.parametrize("status", [RunStatus.error, RunStatus.cancelled, RunStatus.running])
def test_incomplete_results_cannot_be_exported(tmp_path, status):
    result = ApplicationResult(status=status, content=sample_answer(), markdown="partial")
    with pytest.raises(ValueError):
        export_answer(result, tmp_path / "exports")
    assert not (tmp_path / "exports").exists()


def test_missing_references_cannot_be_exported(tmp_path):
    content = sample_answer(
        sections=[
            AnswerSection(
                title="Fact",
                items=[
                    Statement(
                        text="Unsupported.",
                        basis="fact",
                        evidence_ids=["forged"],
                        calculation_ids=[],
                    )
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="could not be validated"):
        export_answer(
            ApplicationResult(status=RunStatus.completed, content=content), tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_sources_include_only_cited_records_with_original_qualifiers():
    from datetime import date
    from pathlib import Path

    records = EvidenceStore.load(
        Path(__file__).parent / "fixtures" / "synthetic_evidence.json", on_date=date(2026, 10, 12)
    ).records
    cited = next(record for record in records if "dubai" in record.cities)
    content = sample_answer(
        sections=[
            AnswerSection(
                title="Evidence",
                items=[
                    Statement(
                        text="Synthetic observation.",
                        basis="fact",
                        evidence_ids=[cited.id],
                        calculation_ids=[],
                    )
                ],
            )
        ]
    )
    result = ApplicationResult(
        status=RunStatus.completed,
        content=content,
        sources={record.id: record for record in records},
    )
    sources = answer_sources(result)
    assert str(cited.source_url) in sources
    assert cited.checked_on.isoformat() in sources
    for qualifier in cited.qualifiers:
        assert qualifier in sources
    for record in records:
        if record.source_url != cited.source_url:
            assert str(record.source_url) not in sources


@pytest.mark.parametrize("payload", ["not JSON", '{"status":"ok","data":{}}'])
def test_corrupt_saved_tool_result_is_not_exportable(payload):
    session_id = str(uuid4())
    run = RunOutput(
        status=RunStatus.completed,
        content=sample_answer().model_dump(),
        tools=[ToolExecution(tool_name="calculate_unit_economics", result=payload)],
    )
    agent = Mock()
    agent.db.get_session.return_value = AgentSession(session_id=session_id, runs=[run])
    with pytest.raises(ValueError, match="saved answer could not be validated"):
        last_answer(session_id, agent)


def test_restore_does_not_borrow_evidence_from_earlier_run():
    session_id = str(uuid4())
    previous = RunOutput(status=RunStatus.completed, content=sample_answer().model_dump(), tools=[])
    invalid = sample_answer(
        sections=[
            AnswerSection(
                title="Fact",
                items=[
                    Statement(
                        text="Old observation.",
                        basis="fact",
                        evidence_ids=["old-evidence"],
                        calculation_ids=[],
                    )
                ],
            )
        ]
    )
    latest = RunOutput(status=RunStatus.completed, content=invalid.model_dump(), tools=[])
    agent = Mock()
    agent.db.get_session.return_value = AgentSession(session_id=session_id, runs=[previous, latest])
    with pytest.raises(ValueError, match="saved answer could not be validated"):
        last_answer(session_id, agent)


@pytest.mark.integration
def test_cli_resumed_answer_exports_and_new_clears_last_answer(tmp_path, monkeypatch):
    from agno.models.message import Message
    from agno.models.openrouter import OpenRouter
    from agno.models.response import ModelResponse

    from advisor import answer_question, create_agent
    from sessions import SessionStore

    def respond(self, messages, **kwargs):
        content = sample_answer().model_dump_json()
        messages.append(Message(role="assistant", content=content))
        return ModelResponse(role="assistant", content=content)

    monkeypatch.setattr(OpenRouter, "response", respond)
    writer = create_agent("fixture", tmp_path / "sessions.db")
    session_id = SessionStore(writer.db).new()
    answer_question("Synthetic question", session_id, writer)
    writer.db.close()
    reader = create_agent("fixture", tmp_path / "sessions.db")
    console = Console(file=StringIO(), color_system=None)
    monkeypatch.setattr(
        console,
        "input",
        Mock(
            side_effect=[
                "/sources",
                "/export ../arbitrary.md",
                "/export",
                "/new",
                "/sources",
                "/export",
                "/quit",
            ]
        ),
    )
    model = Mock(side_effect=AssertionError("Commands must not call inference"))
    monkeypatch.setattr(OpenRouter, "response", model)
    try:
        assert (
            app.chat_loop(reader, console, session_id=session_id, export_dir=tmp_path / "exports")
            == 0
        )
        model.assert_not_called()
        paths = list((tmp_path / "exports").iterdir())
        assert len(paths) == 1
        assert "Synthetic answer" in paths[0].read_text()
        assert "This answer cites no sources" in console.file.getvalue()
        assert "No validated answer" in console.file.getvalue()
        assert "Unknown command or arguments" in console.file.getvalue()
    finally:
        reader.db.close()
