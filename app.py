import os
import sys
from uuid import uuid4

from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.run import RunStatus
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from advisor import PROJECT_DIR, answer_question, create_agent
from answers import AdvisorAnswer
from evidence import DEFAULT_EVIDENCE_FILE, EvidenceDataError, EvidenceStore

HELP_TEXT = """Ask about expanding our Jordanian pizza and kebab restaurant into Dubai or Abu Dhabi.
Follow-up questions use the three most recent completed turns.
This early version can calculate contribution per order from supplied inputs.
Curated research supports advice with source links, inspection dates, and limitations.
Coverage is incomplete; there is no live web search yet.

/help  Show this help
/new   Start a fresh conversation; previous chats remain stored locally
/quit  Exit

Questions are limited to 8,000 characters. Ctrl+C cancels a request;
Ctrl+C at the prompt or Ctrl+D exits."""


def chat_loop(agent: Agent, console: Console) -> int:
    session_id = str(uuid4())
    console.print("UAE Market Expansion Advisor — Dubai and Abu Dhabi", markup=False)
    console.print(
        "Type /help for commands. Research-backed answers include sources and limitations."
    )
    console.print(f"Session: {session_id}")

    while True:
        try:
            question = console.input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question == "/quit":
            console.print("Goodbye.")
            return 0
        if question == "/help":
            console.print(HELP_TEXT, markup=False)
            continue
        if question == "/new":
            session_id = str(uuid4())
            console.print(f"New conversation. Session: {session_id}")
            continue
        if question.startswith("/"):
            console.print("Unknown command. Type /help for available commands.")
            continue

        try:
            with console.status("Preparing your answer..."):
                response = answer_question(question, session_id, agent)
        except KeyboardInterrupt:
            console.print("Request cancelled. You can ask another question.")
            continue
        except ValueError as exc:
            console.print(str(exc), markup=False)
            continue
        except ModelProviderError as exc:
            guidance = {
                401: "Authentication failed. Check OPENROUTER_API_KEY.",
                402: "Check your available OpenRouter credits and key spending limit.",
                429: "Rate limit reached. Wait before retrying.",
            }.get(exc.status_code, "Check your model access and connection.")
            console.print(f"OpenRouter request failed (HTTP {exc.status_code}). {guidance}")
            continue

        if response.status == RunStatus.cancelled:
            console.print("Request cancelled. You can ask another question.")
        elif response.status != RunStatus.completed:
            console.print(
                "The request failed. Check your API key, credits, model access, "
                "connection, and local session storage before trying again."
            )
        elif not isinstance(response.content, AdvisorAnswer):
            console.print("The answer could not be validated. Please try again.")
        else:
            console.print("\nAdvisor:")
            console.print(Markdown(response.markdown))


def main() -> int:
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        print(
            "Set OPENROUTER_API_KEY in .env or your environment before running the app.",
            file=sys.stderr,
        )
        return 1

    try:
        store = EvidenceStore.load(DEFAULT_EVIDENCE_FILE)
    except EvidenceDataError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        agent = create_agent(api_key, evidence_store=store)
    except OSError:
        print(
            "Could not create local session storage under runtime/.",
            file=sys.stderr,
        )
        return 1
    return chat_loop(agent, Console())


if __name__ == "__main__":
    raise SystemExit(main())
