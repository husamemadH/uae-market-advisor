from io import StringIO
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from agno.models.message import Message
from agno.models.openrouter import OpenRouter
from agno.models.response import ModelResponse
from agno.run import RunStatus
from agno.run.agent import RunOutput
from rich.console import Console

import app
from advisor import MAX_INPUT_LENGTH, answer_question, create_agent
from answers import AdvisorAnswer


def typed_answer(text):
    return AdvisorAnswer(answer=text, assumptions=[], next_steps=[], questions=[])


def run_cli(monkeypatch, inputs, answer=None):
    console = Console(file=StringIO(), color_system=None, width=100)
    monkeypatch.setattr(console, "input", Mock(side_effect=inputs))
    responder = Mock(
        return_value=answer or RunOutput(status=RunStatus.completed, content=typed_answer("A completed answer."))
    )
    monkeypatch.setattr(app, "answer_question", responder)
    assert app.chat_loop(Mock(), console) == 0
    return responder, console.file.getvalue()


def test_commands_and_blank_input_do_not_call_model(monkeypatch):
    responder, output = run_cli(monkeypatch, ["", "  ", "/help", "/unknown", "/new", "/quit"])
    responder.assert_not_called()
    assert "Unknown command" in output
    assert "New conversation" in output
    assert "Goodbye" in output


def test_follow_ups_share_session_and_new_resets_it(monkeypatch):
    responder, output = run_cli(monkeypatch, ["First question", "Follow-up", "/new", "New question", "/quit"])
    session_ids = [call.args[1] for call in responder.call_args_list]
    assert session_ids[0] == session_ids[1]
    assert session_ids[2] != session_ids[0]
    for session_id in session_ids:
        UUID(session_id)
    assert output.count("Advisor:") == 3


@pytest.mark.parametrize("end", [EOFError(), KeyboardInterrupt()])
def test_exit_at_prompt(monkeypatch, end):
    responder, output = run_cli(monkeypatch, [end])
    responder.assert_not_called()
    assert "Goodbye" in output


@pytest.mark.parametrize("status", [RunStatus.cancelled, RunStatus.error])
def test_incomplete_runs_are_not_displayed_as_answers(monkeypatch, status):
    responder, output = run_cli(
        monkeypatch, ["Question", "/quit"],
        RunOutput(status=status, content="PARTIAL_CONTENT", reasoning_content="RAW_REASONING"),
    )
    assert responder.call_count == 1
    assert "PARTIAL_CONTENT" not in output
    assert "RAW_REASONING" not in output
    assert "Advisor:" not in output


def test_request_interrupt_returns_to_prompt(monkeypatch):
    console = Console(file=StringIO(), color_system=None)
    monkeypatch.setattr(console, "input", Mock(side_effect=["Question", "Another question", "/quit"]))
    responder = Mock(side_effect=[KeyboardInterrupt(), RunOutput(status=RunStatus.completed, content=typed_answer("Recovered."))])
    monkeypatch.setattr(app, "answer_question", responder)
    assert app.chat_loop(Mock(), console) == 0
    assert responder.call_args_list[0].args[1] == responder.call_args_list[1].args[1]
    assert "Request cancelled" in console.file.getvalue()
    assert "Recovered." in console.file.getvalue()


@pytest.mark.parametrize("question", ["", "  ", "x" * (MAX_INPUT_LENGTH + 1)])
def test_invalid_question_never_reaches_agent(question):
    agent = Mock()
    with pytest.raises(ValueError):
        answer_question(question, str(uuid4()), agent)
    agent.run.assert_not_called()


def test_missing_key_never_constructs_agent(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    builder = Mock()
    monkeypatch.setattr(app, "create_agent", builder)
    assert app.main() == 1
    builder.assert_not_called()
    assert "Set OPENROUTER_API_KEY" in capsys.readouterr().err


def test_sqlite_history_survives_agent_recreation_and_stays_bounded(tmp_path, monkeypatch):
    # Replace only inference: Agno still builds context and writes/reads real SQLite.
    contexts = []

    def respond(self, messages, **kwargs):
        contexts.append([(message.role, message.content) for message in messages])
        content = typed_answer("Acknowledged.").model_dump_json()
        messages.append(Message(role="assistant", content=content))
        return ModelResponse(role="assistant", content=content)

    monkeypatch.setattr(OpenRouter, "response", respond)
    db_file = tmp_path / "sessions.db"
    session_id = str(uuid4())
    for number in range(5):
        # Recreating the agent proves history comes from the database.
        agent = create_agent("test-key", db_file)
        result = answer_question(f"Question {number}", session_id, agent)
        assert result.status == RunStatus.completed

    last_questions = [content for role, content in contexts[-1] if role == "user"]
    assert last_questions == ["Question 1", "Question 2", "Question 3", "Question 4"]
    assert len(agent.get_session(session_id=session_id).runs) == 5

    # Reuse the agent with a different UUID, as /new does.
    result = answer_question("Fresh session", str(uuid4()), agent)
    assert result.status == RunStatus.completed
    assert [content for role, content in contexts[-1] if role == "user"] == ["Fresh session"]


def test_agno_cancellation_is_not_a_success(tmp_path, monkeypatch):
    def interrupt(self, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(OpenRouter, "response", interrupt)
    agent = create_agent("test-key", tmp_path / "sessions.db")
    result = answer_question("Interrupted question", str(uuid4()), agent)
    assert result.status == RunStatus.cancelled
