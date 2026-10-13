import argparse
import os
import sys
from pathlib import Path

from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.run import RunStatus
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from advisor import (
    DEFAULT_EXPORT_DIR,
    PROJECT_DIR,
    answer_question,
    answer_sources,
    create_agent,
    export_answer,
    last_answer,
)
from answers import AdvisorAnswer
from evidence import DEFAULT_EVIDENCE_FILE, EvidenceDataError, EvidenceStore
from profiles import render_profile
from sessions import SessionStorageError, SessionStore
from settings import Settings

HELP_TEXT = """Ask about expanding our Jordanian pizza and kebab restaurant into Dubai or Abu Dhabi.
Follow-up questions use the three most recent completed turns.
Calculate contribution, margin, and break-even from explicit scenario inputs.
Curated research supports advice with source links, inspection dates, and limitations.
Optional live search finds research leads; snippets are not verified market facts.
Set ADVISOR_SEARCH_ENABLED=false to use curated evidence without live search.

/help  Show this help
/profile  Show saved business constraints and unknown fields
/set FIELD VALUE  Save a constraint; no model call
/unset FIELD  Return a constraint to unknown
/new   Reset profile, history, and last answer; previous sessions remain stored
/sources  Show sources cited by the latest validated answer in this session
/export  Save that answer, sources, assumptions, and calculations as Markdown
/quit  Exit

Fields: startup_budget_aed (positive), monthly_rent_budget_aed (zero or positive),
branch_format (takeaway, dine_in, delivery_kitchen, mixed), target_audience,
positioning (value, mid_market, premium), planned_opening.
Amounts use AED, e.g. /set startup_budget_aed 300000 AED (or 300000).
Use at most two decimal places; no commas or other currencies.
Example: /set branch_format takeaway
Ordinary chat and hypothetical calculations do not change saved constraints.
Resume a saved conversation with: uv run python app.py --session SESSION_ID

Questions are limited to 8,000 characters. Ctrl+C cancels a request;
Ctrl+C at the prompt or Ctrl+D exits."""


def chat_loop(
    agent: Agent,
    console: Console,
    *,
    session_id: str | None = None,
    export_dir: Path = DEFAULT_EXPORT_DIR,
) -> int:
    sessions = SessionStore(agent.db)
    resumed = session_id is not None
    try:
        if session_id is None:
            session_id = sessions.new()
        else:
            sessions.profile(session_id)  # Validate resume before accepting any questions.
    except (ValueError, SessionStorageError) as exc:
        console.print(str(exc), markup=False)
        return 1
    console.print("UAE Market Expansion Advisor — Dubai and Abu Dhabi", markup=False)
    console.print(
        "Type /help for commands. Research-backed answers include sources and limitations."
    )
    console.print(f"{'Resumed session' if resumed else 'Session'}: {session_id}")

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
        if question.startswith("/"):
            command, *arguments = question.split(maxsplit=2)
            try:
                if command == "/new" and not arguments:
                    session_id = sessions.new()
                    console.print(f"New conversation. Session: {session_id}")
                elif command == "/profile" and not arguments:
                    console.print(render_profile(sessions.profile(session_id)), markup=False)
                elif command == "/set" and len(arguments) == 2:
                    profile = sessions.update_profile(session_id, arguments[0], arguments[1])
                    console.print("Profile saved.", markup=False)
                    console.print(render_profile(profile), markup=False)
                elif command == "/unset" and len(arguments) == 1:
                    sessions.update_profile(session_id, arguments[0], None)
                    console.print(f"{arguments[0]} is now unknown.", markup=False)
                elif command in {"/sources", "/export"} and not arguments:
                    result = last_answer(session_id, agent)
                    if result is None:
                        console.print("No validated answer in this session yet.")
                    elif command == "/sources":
                        console.print(Markdown(answer_sources(result)))
                    else:
                        path = export_answer(result, export_dir)
                        console.print(f"Exported answer to {path}", markup=False)
                else:
                    console.print(
                        "Unknown command or arguments. Type /help for available commands."
                    )
            except (ValueError, SessionStorageError) as exc:
                console.print(str(exc), markup=False)
            except OSError:
                console.print("Could not write the answer export. Check runtime/exports/.")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UAE Market Expansion Advisor")
    parser.add_argument("--session", metavar="SESSION_ID", help="Resume a known local session UUID")
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        print(
            "Set OPENROUTER_API_KEY in .env or your environment before running the app.",
            file=sys.stderr,
        )
        return 1

    try:
        settings = Settings.from_environment()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        store = EvidenceStore.load(DEFAULT_EVIDENCE_FILE)
    except EvidenceDataError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        agent = create_agent(api_key, evidence_store=store, settings=settings)
    except OSError:
        print(
            "Could not create local session storage under runtime/.",
            file=sys.stderr,
        )
        return 1
    try:
        return chat_loop(agent, Console(), session_id=args.session)
    finally:
        agent.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
