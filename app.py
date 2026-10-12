import os
import sys
from pathlib import Path

from agno.agent import Agent
from agno.exceptions import ModelProviderError
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv


MODEL_ID = "deepseek/deepseek-v4.1-flash"


def main() -> int:
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        print(
            "Set OPENROUTER_API_KEY in .env or your environment before running the app.",
            file=sys.stderr,
        )
        return 1

    agent = Agent(
        name="UAE Market Expansion Advisor",
        model=OpenRouter(
            id=MODEL_ID,
            api_key=api_key,
            max_tokens=4096,
            timeout=45,
            client_params={"max_retries": 0},
        ),
        instructions=[
            "You advise a Jordanian restaurant specializing in meaty pizza and kebab sandwiches.",
            "The restaurant is considering expansion into Dubai and Abu Dhabi.",
            "Keep answers concise and distinguish assumptions from verified facts.",
        ],
        markdown=True,
    )

    try:
        agent.print_response(
            "In two sentences, explain how you would help us evaluate expansion into "
            "Dubai and Abu Dhabi. Describe your approach without making market claims."
        )
    except ModelProviderError:
        print(
            "OpenRouter request failed. Check your key, credits, model access, and connection.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
