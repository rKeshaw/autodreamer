"""
LLM Utilities — Robust JSON parsing, multi-model support, and shared LLM interface.

All modules should use these utilities instead of raw ollama.Client calls
and bare json.loads() to ensure consistent behavior and error handling.
"""

import re
import json
import time
import os
import threading
from collections import OrderedDict
from ollama import Client

# ── Singleton client ──────────────────────────────────────────────────────────

_client = None

_OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
_OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
_OLLAMA_RETRY_BACKOFF_SECONDS = float(
    os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "1.5")
)
_OLLAMA_CACHE_ENABLED = os.getenv("OLLAMA_CACHE_ENABLED", "1").lower() in (
    "1", "true", "yes", "on"
)
_OLLAMA_CACHE_MAX_TEMPERATURE = float(
    os.getenv("OLLAMA_CACHE_MAX_TEMPERATURE", "0.2")
)
_OLLAMA_CACHE_MAX_ENTRIES = int(os.getenv("OLLAMA_CACHE_MAX_ENTRIES", "2048"))

_response_cache: OrderedDict[tuple, str] = OrderedDict()
_cache_lock = threading.Lock()

def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(timeout=_OLLAMA_TIMEOUT_SECONDS)
    return _client


def _cacheable_request(temperature: float) -> bool:
    return (
        _OLLAMA_CACHE_ENABLED and
        float(temperature) <= _OLLAMA_CACHE_MAX_TEMPERATURE and
        _OLLAMA_CACHE_MAX_ENTRIES > 0
    )


def _build_cache_key(model: str, messages: list[dict],
                     temperature: float) -> tuple:
    normalized_messages = tuple(
        (str(msg.get("role", "")), str(msg.get("content", "")))
        for msg in messages
    )
    return (
        model,
        round(float(temperature), 3),
        normalized_messages,
    )


def _cache_get(cache_key: tuple) -> str | None:
    with _cache_lock:
        value = _response_cache.get(cache_key)
        if value is not None:
            _response_cache.move_to_end(cache_key)
        return value


def _cache_set(cache_key: tuple, content: str):
    with _cache_lock:
        _response_cache[cache_key] = content
        _response_cache.move_to_end(cache_key)
        while len(_response_cache) > _OLLAMA_CACHE_MAX_ENTRIES:
            _response_cache.popitem(last=False)


def _chat_with_retries(model: str, messages: list[dict],
                       temperature: float) -> str:
    """
    Execute a chat request with bounded retries.

    This protects long-running autonomous loops from transient transport
    failures and hard timeouts while still surfacing a clear final error.
    """
    client = _get_client()
    cache_key = None
    if _cacheable_request(temperature):
        cache_key = _build_cache_key(model, messages, temperature)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    last_error = None

    for attempt in range(_OLLAMA_MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature}
            )
            content = response.get('message', {}).get('content', '')
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model returned empty content")
            content = content.strip()
            if cache_key is not None:
                _cache_set(cache_key, content)
            return content
        except Exception as exc:
            last_error = exc
            if attempt >= _OLLAMA_MAX_RETRIES:
                break
            backoff = _OLLAMA_RETRY_BACKOFF_SECONDS * (attempt + 1)
            print(
                f"[LLM] Call failed (attempt {attempt + 1}/"
                f"{_OLLAMA_MAX_RETRIES + 1}): {exc}. "
                f"Retrying in {backoff:.1f}s..."
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"LLM call failed after {_OLLAMA_MAX_RETRIES + 1} attempt(s): "
        f"{last_error}"
    ) from last_error


# ── Robust JSON parsing ──────────────────────────────────────────────────────

def _repair_json_candidate(candidate: str) -> str:
    """
    Repair common JSON corruption from LLM outputs.

    Most frequently this is caused by LaTeX-ish backslashes such as \\chi or
    \\lambda appearing inside JSON strings, which are invalid JSON escapes.
    """
    if not candidate:
        return candidate

    # Escape invalid backslash sequences while preserving valid JSON escapes.
    candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
    # Remove NULs or other control chars that occasionally appear in outputs.
    candidate = ''.join(
        ch for ch in candidate
        if ch == '\n' or ch == '\r' or ch == '\t' or ord(ch) >= 32
    )
    return candidate

def parse_llm_json(raw: str):
    """
    Extract JSON from LLM output, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Preamble text before JSON
    - Trailing text after JSON
    - Single-quoted strings (common with smaller models)

    Returns parsed object or None if truly unparseable.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            return json.loads(_repair_json_candidate(text))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find JSON object or array within the text
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start < 0:
            continue

        # Find matching closing bracket, handling nesting
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        # Try fixing single quotes
                        try:
                            fixed = candidate.replace("'", '"')
                            return json.loads(_repair_json_candidate(fixed))
                        except (json.JSONDecodeError, ValueError):
                            try:
                                return json.loads(_repair_json_candidate(candidate))
                            except (json.JSONDecodeError, ValueError):
                                break

    return None


def require_json(raw: str, default=None):
    """Parse LLM JSON output, returning default if unparseable."""
    result = parse_llm_json(raw)
    return result if result is not None else default


# ── Multi-model LLM calls ────────────────────────────────────────────────────

def llm_call(prompt: str, temperature: float = 0.7,
             model: str = None, system: str = None,
             role: str = "creative") -> str:
    """
    Unified LLM call with model selection based on task role.

    Roles:
        creative  — dreaming, synthesis, analogies (higher temp, creative model)
        precise   — JSON extraction, factual questions (low temp, precise model)
        code      — code generation for sandbox
        reasoning — deliberate thinking, chain-of-thought
        notebook  — structured research memos
        publication — conservative publication drafting
        verifier  — strict admissibility and validation checks
    """
    from config import MODELS

    if model is None:
        model = getattr(MODELS, role.upper(), MODELS.CREATIVE)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _chat_with_retries(model, messages, temperature)


def llm_json(prompt: str, temperature: float = 0.1,
             model: str = None, default=None,
             system: str = None) -> any:
    """
    LLM call that expects JSON output. Uses precise model by default.

    Always uses the JSON system message for better compliance.
    Returns parsed JSON or default if unparseable.
    """
    json_system = (
        "You are a structured data extractor. You respond ONLY with valid JSON. "
        "No preamble, no explanation, no markdown code blocks, no trailing text. "
        "Just the raw JSON object or array."
    )
    if system:
        json_system = system + "\n\n" + json_system

    try:
        raw = llm_call(
            prompt,
            temperature=temperature,
            model=model,
            system=json_system,
            role="precise"
        )
    except RuntimeError:
        return default
    return require_json(raw, default=default)


def llm_chat(messages: list[dict], temperature: float = 0.7,
             model: str = None, role: str = "creative") -> str:
    """
    Multi-turn LLM call for conversation-style interactions.
    """
    from config import MODELS

    if model is None:
        model = getattr(MODELS, role.upper(), MODELS.CREATIVE)

    return _chat_with_retries(model, messages, temperature)
