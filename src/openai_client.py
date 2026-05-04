"""Thin OpenAI Responses-API wrapper used by the agent loop.

Uses OPENAI_API_KEY plus the model from OPENAI_MODEL (default `gpt-5.4`).
Exposes a `complete(system, user, ...)` helper that returns plain text.
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # picks up OPENAI_API_KEY from env
    return _client


def _reset_client_singleton() -> None:
    global _client
    _client = None


def _retry_config() -> tuple[int, float, float]:
    attempts = max(1, int(os.environ.get("OPENAI_RETRY_ATTEMPTS", "4")))
    base_delay = max(0.1, float(os.environ.get("OPENAI_RETRY_BASE_SEC", "2.0")))
    max_delay = max(base_delay, float(os.environ.get("OPENAI_RETRY_MAX_SEC", "20.0")))
    return attempts, base_delay, max_delay


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code in {408, 409, 429} or bool(status_code and status_code >= 500)
    return False


def _retry_delay(attempt: int, *, base_delay: float, max_delay: float) -> float:
    capped = min(max_delay, base_delay * (2 ** max(0, attempt - 1)))
    return capped * random.uniform(0.8, 1.2)


def _response_text(resp: object) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text
    pieces: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                pieces.append(t)
    return "".join(pieces)


def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int = 16000,
) -> str:
    """Call the Responses API and return the concatenated text output."""
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

    attempts, base_delay, max_delay = _retry_config()
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            client = _client_singleton()
            resp = client.responses.create(**kwargs)
            return _response_text(resp)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc) or attempt >= attempts:
                raise
            delay = _retry_delay(attempt, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                "OpenAI request failed with retryable %s on attempt %d/%d; retrying in %.1fs.",
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            _reset_client_singleton()
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI completion failed without an exception")
