"""Shared utilities: OpenAI client factory, retry logic, cost tracking."""

from __future__ import annotations

import logging
import os
import time

import openai
from openai import OpenAI

from dreamink.config import AppConfig, get_config
from dreamink.models import TokenUsage

logger = logging.getLogger(__name__)

# Status codes that are safe to retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


def get_openai_client(config: AppConfig | None = None) -> OpenAI:
    """Create an OpenAI client using the API key from the environment.

    Args:
        config: App config. If None, loads from default config file.

    Raises:
        RuntimeError: If the API key environment variable is not set.
    """
    config = config or get_config()
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key: set the {config.api_key_env} environment variable."
        )
    return OpenAI(api_key=api_key)


def retry_api_call(func, *, max_retries: int = 3):
    """Call an OpenAI API function with exponential backoff on transient errors.

    Retries on 429 (rate limit) and 5xx server errors.
    Does NOT retry on 400 (bad request) or 401 (auth) errors.

    Args:
        func: A zero-argument callable that makes the API request.
        max_retries: Maximum number of retry attempts.

    Returns:
        The API response.

    Raises:
        openai.APIError: If all retries are exhausted or a non-retryable error occurs.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except openai.APIStatusError as e:
            last_error = e
            if e.status_code not in _RETRYABLE_STATUS_CODES:
                logger.error("Non-retryable API error (HTTP %d): %s", e.status_code, e)
                raise
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Retryable API error (HTTP %d), attempt %d/%d, waiting %ds: %s",
                    e.status_code,
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                logger.error("All %d retries exhausted: %s", max_retries, e)
                raise
        except openai.APIConnectionError as e:
            last_error = e
            if attempt < max_retries:
                wait = 2**attempt
                logger.warning(
                    "Connection error, attempt %d/%d, waiting %ds: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                logger.error("All %d retries exhausted: %s", max_retries, e)
                raise


def extract_token_usage(response) -> TokenUsage:
    """Extract token usage from an OpenAI chat completion response."""
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )


# GPT-4 pricing per 1K tokens (as of Nov 2023)
_GPT4_PROMPT_COST_PER_1K = 0.03
_GPT4_COMPLETION_COST_PER_1K = 0.06


def calculate_llm_cost(usage: TokenUsage) -> float:
    """Calculate USD cost for a GPT-4 API call from token usage."""
    return (
        usage.prompt_tokens * _GPT4_PROMPT_COST_PER_1K / 1000
        + usage.completion_tokens * _GPT4_COMPLETION_COST_PER_1K / 1000
    )
