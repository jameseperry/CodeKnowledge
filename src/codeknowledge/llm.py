"""LLM client for code description generation."""

from __future__ import annotations

import logging
import os

import anthropic

log = logging.getLogger(__name__)


def _parse_custom_headers() -> dict[str, str]:
    """Parse ANTHROPIC_CUSTOM_HEADERS env var (comma-separated key: value pairs)."""
    raw = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    headers: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


def _get_model(tier: str) -> str:
    """Resolve a model tier to a model name, respecting env overrides."""
    env_map = {
        "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    }
    env_key = env_map.get(tier)
    if env_key:
        return os.environ.get(env_key, tier)
    return tier  # assume it's a direct model name


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client with custom headers from env."""
    headers = _parse_custom_headers()
    return anthropic.Anthropic(default_headers=headers)


def describe_file(
    prompt: str,
    model_tier: str = "haiku",
    max_tokens: int = 4096,
) -> str:
    """Send a description prompt to the LLM and return the response text.

    Args:
        prompt: The full prompt built by describe.build_prompt().
        model_tier: One of "haiku", "sonnet", "opus".
        max_tokens: Maximum response tokens.

    Returns:
        The raw response text from the model.
    """
    model = _get_model(model_tier)
    client = get_client()

    log.info("Calling %s for file description...", model)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        system="You are a code analysis assistant. You produce clear, accurate descriptions of source code structure and behavior. Be concise but thorough. Focus on what the code does and why, not how to use it.",
    )

    return message.content[0].text
