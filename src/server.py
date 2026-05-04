"""A2A server entrypoint for the MLE-Bench purple agent."""
from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.applications import Starlette

from .executor import Executor


def build_app(*, host: str, port: int, card_url: str | None = None) -> Starlette:
    skill = AgentSkill(
        id="mle_bench_solver",
        name="MLE-Bench Kaggle competition solver",
        description=(
            "Receives a Kaggle-style competition tarball, drafts a solution.py "
            "with the configured OpenAI model in an iterative plan/code/run/"
            "debug loop, and returns submission.csv as a task artifact."
        ),
        tags=["mle-bench", "kaggle", "ml-engineering", "code-agent"],
        examples=[
            "Solve spaceship-titanic and produce submission.csv",
        ],
    )

    agent_card = AgentCard(
        name=f"MLE-Bench Purple Agent ({os.environ.get('OPENAI_MODEL', 'gpt-5.4')})",
        description=(
            "AIDE-style ML engineering agent for MLE-Bench. Plans, writes, "
            "executes, and debugs a solution.py against the provided dataset, "
            "then returns submission.csv as an A2A artifact."
        ),
        url=card_url or f"http://{host}:{port}/",
        version="0.1.0",
        skills=[skill],
        default_input_modes=["text", "file"],
        default_output_modes=["text", "file"],
        capabilities=AgentCapabilities(streaming=True),
    )

    handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    max_content_length = int(
        os.environ.get("A2A_MAX_CONTENT_LENGTH", str(512 * 1024 * 1024))
    )
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
        max_content_length=max_content_length,
    )
    return app.build()


def main() -> None:
    parser = argparse.ArgumentParser(description="MLE-Bench Purple Agent (AIDE-style)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    uvicorn.run(
        build_app(host=args.host, port=args.port, card_url=args.card_url),
        host=args.host,
        port=args.port,
        timeout_keep_alive=3600,
    )


if __name__ == "__main__":
    main()
