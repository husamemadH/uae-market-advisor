from uuid import uuid4

from agno.models.message import Message
from agno.models.openrouter import OpenRouter
from agno.models.response import ModelResponse
from agno.run import RunStatus

from advisor import answer_question, create_agent
from answers import AdvisorAnswer


def test_saved_messages_and_answers_survive_database_reopen(tmp_path, monkeypatch):
    # Only inference is replaced. Every write/read uses the real SQLite adapter.
    model_calls = []

    def respond(self, messages, **kwargs):
        question = next(message.content for message in reversed(messages) if message.role == "user")
        model_calls.append(question)
        answer = AdvisorAnswer(
            answer=f"Saved answer for: {question}",
            assumptions=["Synthetic database test."],
            next_steps=["Review this scenario."],
            questions=["Would you like to adjust the scenario?"],
        )
        content = answer.model_dump_json()
        messages.append(Message(role="assistant", content=content))
        return ModelResponse(role="assistant", content=content)

    monkeypatch.setattr(OpenRouter, "response", respond)
    db_file = tmp_path / "sessions.db"
    writer = create_agent("test-key", db_file)
    scenarios = {
        str(uuid4()): ["Dubai takeaway scenario", "Revise the Dubai scenario"],
        str(uuid4()): ["Abu Dhabi dine-in scenario"],
    }
    expected = {}
    for session_id, questions in scenarios.items():
        expected[session_id] = {}
        for question in questions:
            result = answer_question(question, session_id, writer)
            assert result.status == RunStatus.completed
            expected[session_id][result.run_id] = (question, result.content)

    assert db_file.is_file()
    # Release the original database connections and discard the writer.
    writer.db.close()
    del writer

    reader = create_agent("test-key", db_file)
    try:
        for session_id, expected_runs in expected.items():
            session = reader.get_session(session_id=session_id)
            assert session is not None
            assert session.session_id == session_id
            assert {run.run_id for run in session.runs} == set(expected_runs)

            for run in session.runs:
                question, expected_answer = expected_runs[run.run_id]
                assert run.status == RunStatus.completed
                assert run.session_id == session_id
                assert [message.content for message in run.messages if message.role == "user"] == [question]

                # Check the saved final answer and the saved assistant message.
                # Agno can deserialize structured content as a dict or JSON text.
                stored_answer = (
                    AdvisorAnswer.model_validate_json(run.content)
                    if isinstance(run.content, str)
                    else AdvisorAnswer.model_validate(run.content)
                )
                assert stored_answer == expected_answer
                assistant_messages = [message for message in run.messages if message.role == "assistant"]
                assert len(assistant_messages) == 1
                assert AdvisorAnswer.model_validate_json(assistant_messages[0].content) == expected_answer

        assert reader.get_session(session_id=str(uuid4())) is None
        assert model_calls == [question for questions in scenarios.values() for question in questions]
    finally:
        reader.db.close()
