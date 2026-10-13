from decimal import Decimal
from io import StringIO
from unittest.mock import Mock
from uuid import uuid4

import pytest
from agno.db.sqlite import SqliteDb
from agno.models.message import Message
from agno.models.openrouter import OpenRouter
from agno.models.response import ModelResponse
from agno.run import RunStatus
from rich.console import Console
from sqlalchemy.exc import OperationalError

import app
from advisor import answer_question, create_agent, last_answer
from answers import AdvisorAnswer
from profiles import BusinessProfile
from sessions import SessionStorageError, SessionStore, validate_session_id


@pytest.fixture
def store(tmp_path):
    db = SqliteDb(db_file=str(tmp_path / "sessions.db"))
    yield SessionStore(db)
    db.close()


@pytest.mark.parametrize("value", ["", "wrong", "../sessions.db", " " + str(uuid4()), uuid4().hex])
def test_invalid_session_id_is_rejected(value):
    with pytest.raises(ValueError, match="Invalid session ID"):
        validate_session_id(value)


@pytest.mark.integration
def test_profile_survives_reopen_before_any_model_call_and_sessions_are_isolated(tmp_path):
    path = str(tmp_path / "sessions.db")
    writer = SessionStore(SqliteDb(db_file=path))
    first, second = writer.new(), writer.new()
    writer.update_profile(first, "startup_budget_aed", "300000 AED")
    writer.update_profile(first, "target_audience", "families")
    writer.update_profile(second, "startup_budget_aed", "900000 AED")
    writer.db.close()

    reader = SessionStore(SqliteDb(db_file=path))
    try:
        assert reader.profile(first).startup_budget_aed == Decimal("300000")
        assert reader.profile(first).target_audience == "families"
        assert reader.profile(second).startup_budget_aed == Decimal("900000")
        assert reader.profile(second).target_audience is None
        with pytest.raises(ValueError):
            reader.update_profile(first, "startup_budget_aed", "100 USD")
        assert reader.profile(first).startup_budget_aed == Decimal("300000")
        reader.update_profile(first, "startup_budget_aed", None)
        assert reader.profile(first).startup_budget_aed is None
        assert reader.profile(reader.new()) == BusinessProfile()
        with pytest.raises(ValueError, match="Unknown session"):
            reader.profile(str(uuid4()))
    finally:
        reader.db.close()


@pytest.mark.integration
def test_profile_reaches_model_beyond_history_and_never_leaks_between_sessions(
    tmp_path, monkeypatch
):
    contexts = []

    def respond(self, messages, **kwargs):
        contexts.append([(message.role, message.content) for message in messages])
        content = AdvisorAnswer(
            kind="insufficient_evidence",
            cities=["dubai"],
            headline="Synthetic acknowledgement.",
            sections=[],
            assumptions=[],
            next_steps=[],
            questions=[],
        ).model_dump_json()
        messages.append(Message(role="assistant", content=content))
        return ModelResponse(role="assistant", content=content)

    monkeypatch.setattr(OpenRouter, "response", respond)
    path = tmp_path / "sessions.db"
    agent = create_agent("fixture", path)
    store = SessionStore(agent.db)
    first, second = store.new(), store.new()
    store.update_profile(first, "startup_budget_aed", "123456 AED")
    store.update_profile(first, "target_audience", "Distinctive audience A")
    store.update_profile(second, "startup_budget_aed", "654321 AED")
    for number in range(5):
        answer_question(f"Hypothetical question {number}", first, agent)
    assert len([role for role, _ in contexts[-1] if role == "user"]) == 4
    assert '"startup_budget_aed":"123456"' in contexts[-1][0][1]
    assert store.profile(first).startup_budget_aed == Decimal("123456")
    answer_question("Session B question", second, agent)
    assert '"startup_budget_aed":"654321"' in contexts[-1][0][1]
    assert "Distinctive audience A" not in str(contexts[-1])
    assert "Hypothetical question" not in str(contexts[-1])
    store.update_profile(first, "startup_budget_aed", None)
    assert len(store.load(first).runs) == 5  # Profile writes preserve every saved run.
    agent.db.close()

    reader = create_agent("fixture", path)
    try:
        answer_question("Resume A", first, reader)
        assert '"startup_budget_aed":null' in contexts[-1][0][1]
        assert "Distinctive audience A" in contexts[-1][0][1]
        assert "Session B question" not in str(contexts[-1])
        assert "Hypothetical question 4" in str(contexts[-1])
        assert last_answer(first, reader).status == RunStatus.completed
    finally:
        reader.db.close()


@pytest.mark.integration
def test_cli_profile_commands_and_new_do_not_call_model(store, monkeypatch, tmp_path):
    session_id = store.new()
    console = Console(file=StringIO(), color_system=None, width=120)
    monkeypatch.setattr(
        console,
        "input",
        Mock(
            side_effect=[
                "/set startup_budget_aed 500000 AED",
                "/set branch_format takeaway",
                "/set startup_budget_aed 100 USD",
                "/profile",
                "/unset branch_format",
                "/set",
                "/unset",
                "/sources",
                "/export",
                "/new",
                "/profile",
                "/export",
                "/quit",
            ]
        ),
    )
    responder = Mock()
    monkeypatch.setattr(app, "answer_question", responder)
    agent = Mock(db=store.db)
    assert (
        app.chat_loop(agent, console, session_id=session_id, export_dir=tmp_path / "exports") == 0
    )
    responder.assert_not_called()
    output = console.file.getvalue()
    assert "startup_budget_aed: 500000.00 AED" in output
    assert "Other currencies" in output
    assert "branch_format is now unknown" in output
    assert "No validated answer" in output
    assert "startup_budget_aed: Unknown" in output.split("New conversation.")[1]
    assert store.profile(session_id).startup_budget_aed == Decimal("500000")
    assert store.profile(session_id).branch_format is None
    assert not (tmp_path / "exports").exists()


@pytest.mark.integration
@pytest.mark.parametrize("session_id", ["invalid", str(uuid4())])
def test_cli_bad_resume_exits_before_prompt_or_model(store, session_id):
    console = Console(file=StringIO(), color_system=None)
    console.input = Mock(side_effect=AssertionError("Should not prompt"))
    agent = Mock(db=store.db)
    assert app.chat_loop(agent, console, session_id=session_id) == 1
    agent.run.assert_not_called()
    assert "session ID" in console.file.getvalue()


def test_session_storage_errors_are_explicit():
    db = Mock()
    db.get_session.side_effect = OperationalError("read", {}, Exception("fixture"))
    with pytest.raises(SessionStorageError):
        SessionStore(db).load(str(uuid4()))
    db.upsert_session.return_value = None
    with pytest.raises(SessionStorageError):
        SessionStore(db).new()


def test_main_passes_explicit_resume_id_and_closes_storage(monkeypatch):
    session_id = str(uuid4())
    agent = Mock()
    loop = Mock(return_value=0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setattr(app, "create_agent", Mock(return_value=agent))
    monkeypatch.setattr(app, "chat_loop", loop)
    assert app.main(["--session", session_id]) == 0
    assert loop.call_args.kwargs["session_id"] == session_id
    agent.db.close.assert_called_once()
    agent.run.assert_not_called()
