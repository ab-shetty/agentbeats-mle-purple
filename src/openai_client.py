"""Thin OpenAI Responses-API wrapper used by the agent loop.

Uses OPENAI_API_KEY plus the model from OPENAI_MODEL (default `gpt-5.4`).
Exposes a `complete(system, user, ...)` helper that returns plain text.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # picks up OPENAI_API_KEY from env
    return _client


def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int = 16000,
) -> str:
    """Call the Responses API and return the concatenated text output."""
    client = _client_singleton()
    model = model or os.environ.get("OPENAI_MODEL", "gpt-5.4")
    effort = reasoning_effort or os.environ.get("REASONING_EFFORT", "medium")

    kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_output_tokens": max_output_tokens,
    }
    # gpt-5-* accept a reasoning param; ignore for non-reasoning models.
    if model.startswith(("gpt-5", "o")):
        kwargs["reasoning"] = {"effort": effort}

    resp = client.responses.create(**kwargs)
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Fallback: assemble from output items.
    pieces: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                pieces.append(t)
    return "".join(pieces)
