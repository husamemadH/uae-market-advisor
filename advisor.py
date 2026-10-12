from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.run.agent import RunOutput


MODEL_ID = "deepseek/deepseek-v4.1-flash"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_FILE = PROJECT_DIR / "runtime" / "sessions.db"
MAX_INPUT_LENGTH = 8000


def create_agent(api_key: str, db_file: Path = DEFAULT_DB_FILE) -> Agent:

    db_file.parent.mkdir(parents=True, exist_ok=True)
    return Agent(
        name="UAE Market Expansion Advisor",
        model=OpenRouter(
            id=MODEL_ID,
            api_key=api_key,
            max_tokens=4096,
            timeout=45,
            client_params={"max_retries": 0},
        ),
        db=SqliteDb(db_file=str(db_file)),
        add_history_to_context=True,
        num_history_runs=3,
        cache_session=False,
        retries=0,
        instructions=[
            "You advise a Jordanian restaurant specializing in meaty pizza and kebab sandwiches.",
            "The restaurant is considering expansion into Dubai and Abu Dhabi.",
            "Keep answers concise and distinguish assumptions from verified facts.",
            "You have no research or calculation tools yet. Do not invent market data, "
            "prices, source links, or claim to have verified current facts.",
            "Use the recent conversation for follow-ups. If a detail is missing, "
            "ask rather than guessing or claiming unlimited memory.",
        ],
        markdown=True,
        telemetry=False,
    )


def answer_question(question: str, session_id: str, agent: Agent) -> RunOutput:
    """Execute one turn; callers inspect status before rendering its content."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a question before sending it.")
    if len(question) > MAX_INPUT_LENGTH:
        raise ValueError(f"Keep questions within {MAX_INPUT_LENGTH:,} characters.")
    return agent.run(question, session_id=session_id, stream=False)
