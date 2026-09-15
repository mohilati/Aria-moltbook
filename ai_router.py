from __future__ import annotations

import os
from typing import Any

import httpx

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()
TIMEOUT = float(os.getenv("AI_ROUTER_TIMEOUT", "45"))


def _extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned: {data}")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if isinstance(text, list):
        text = "\n".join(
            str(item.get("text", ""))
            for item in text
            if isinstance(item, dict) and item.get("text")
        )
    text = str(text or "").strip()
    if not text:
        raise RuntimeError(f"Provider returned empty content: {data}")
    return text


def _gemini(system: str, user: str, max_tokens: int) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": min(max_tokens, 1400),
            "temperature": 0.9,
        },
    }
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10)) as client:
        response = client.post(
            url, params={"key": GEMINI_API_KEY}, json=payload
        )
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"Gemini HTTP {response.status_code}",
            request=response.request,
            response=response,
        )
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(
        str(part.get("text", "")).strip()
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()
    if not text:
        raise RuntimeError(f"Gemini returned empty content: {data}")
    return text


def _openrouter(system: str, user: str, max_tokens: int) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": min(max_tokens, 1400),
        "temperature": 0.9,
    }
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10)) as client:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "AriaPsi",
            },
            json=payload,
        )
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"OpenRouter HTTP {response.status_code}",
            request=response.request,
            response=response,
        )
    return _extract_text(response.json())


def _cerebras(system: str, user: str, max_tokens: int) -> str:
    if not CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY missing")
    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": min(max_tokens, 1400),
        "temperature": 0.9,
    }
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10)) as client:
        response = client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"Cerebras HTTP {response.status_code}",
            request=response.request,
            response=response,
        )
    return _extract_text(response.json())


def llm_text(client: bool, system: str, user: str,
             max_output_tokens: int = 220) -> str:
    del client
    providers = []
    if GEMINI_API_KEY:
        providers.append(("gemini", _gemini))
    if OPENROUTER_API_KEY:
        providers.append(("openrouter", _openrouter))
    if CEREBRAS_API_KEY:
        providers.append(("cerebras", _cerebras))

    if not providers:
        raise RuntimeError(
            "No AI provider configured. Set GEMINI_API_KEY, "
            "OPENROUTER_API_KEY, or CEREBRAS_API_KEY."
        )

    last_error = None
    for name, provider in providers:
        try:
            print(f"AI router: trying provider={name}")
            result = provider(system, user, max_output_tokens)
            print(f"AI router: provider={name} succeeded")
            return result
        except httpx.HTTPStatusError as error:
            last_error = error
            print(
                f"AI router: provider={name} HTTP "
                f"{error.response.status_code}; failover"
            )
        except (httpx.TimeoutException, httpx.RequestError) as error:
            last_error = error
            print(f"AI router: provider={name} network error; failover")
        except Exception as error:
            last_error = error
            print(f"AI router: provider={name} failed; failover - {error}")

    raise RuntimeError(f"All configured AI providers failed: {last_error!r}")
