"""A2A server entrypoint for the MLE-Bench purple agent."""
from __future__ import annotations

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .executor import Executor


def main() -> None:
    parser = argparse.ArgumentParser(description="MLE-Bench Purple Agent (gpt-5-mini, AIDE-style)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    skill = AgentSkill(
        id="mle_bench_solver",
        name="MLE-Bench Kaggle competition solver",
        description=(
            "Receives a Kaggle-style competition tarball, drafts a solution.py "
            "with gpt-5-mini in an iterative plan/code/run/debug loop, and "
            "returns submission.csv as a task artifact."
        ),
        tags=["mle-bench", "kaggle", "ml-engineering", "code-agent"],
        examples=[
            "Solve spaceship-titanic and produce submission.csv",
        ],
    )

    agent_card = AgentCard(
        name="MLE-Bench Purple Agent (gpt-5-mini)",
        description=(
            "AIDE-style ML engineering agent for MLE-Bench. Plans, writes, "
            "executes, and debugs a solution.py against the provided dataset, "
            "then returns submission.csv as an A2A artifact."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
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
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=3600)


if __name__ == "__main__":
    main()
