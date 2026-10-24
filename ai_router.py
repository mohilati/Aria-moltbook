from __future__ import annotations

import os
import time
from typing import Any, Callable

import httpx

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip()

TIMEOUT = float(os.getenv("AI_ROUTER_TIMEOUT", "45"))

# Order in which providers are tried. Change it without touching code by
# setting AI_PROVIDER_ORDER, e.g. "groq,gemini,openrouter,cerebras".
PROVIDER_ORDER = [
    p.strip().lower()
    for p in os.getenv("AI_PROVIDER_ORDER", "gemini,groq,openrouter,cerebras").split(",")
    if p.strip()
]

# Temporary errors: worth a short retry on the same provider.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
# Permanent errors (bad key, no credit, model gone): skip this provider for
# the rest of the run so we do not waste time on it again.
DISABLE_STATUS = {400, 401, 402, 403, 404}

RETRIES = 2  # attempts per provider for temporary errors

_disabled: set[str] = set()
# Providers / models that failed temporarily in this run: tried last next time,
# so one overloaded model does not slow down every single call.
_slow_providers: set[str] = set()
_slow_models: set[str] = set()


def _http_error(name: str, response: httpx.Response) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        f"{name} HTTP {response.status_code}",
        request=response.request,
        response=response,
    )


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


# ---------------------------------------------------------------- Gemini

def _gemini_once(model: str, system: str, user: str, max_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": min(max_tokens * 4, 2400),
            "temperature": 1.0,
        },
    }
    with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10)) as client:
        response = client.post(url, params={"key": GEMINI_API_KEY}, json=payload)

    if response.status_code >= 400:
        raise _http_error("Gemini", response)

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


def _gemini(system: str, user: str, max_tokens: int) -> str:
    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL not in models:
        models.append(GEMINI_FALLBACK_MODEL)
    models.sort(key=lambda m: m in _slow_models)  # models that failed earlier go last

    last_error: Exception | None = None
    for model in models:
        try:
            return _with_retries("gemini", lambda m=model: _gemini_once(m, system, user, max_tokens))
        except httpx.HTTPStatusError as error:
            last_error = error
            _slow_models.add(model)
            print(f"AI router: gemini model={model} failed with HTTP {error.response.status_code}")
        except (httpx.TimeoutException, httpx.RequestError) as error:
            last_error = error
            _slow_models.add(model)
            print(f"AI router: gemini model={model} network error")
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------- OpenAI-compatible providers

def _openai_compat(
    name: str,
    url: str,
    key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    extra_headers: dict[str, str] | None = None,
) -> str:
    budget = min(max_tokens, 1400)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 1.0,
    }

    # gpt-oss models "think" first and the thinking uses up the token budget,
    # which can leave the answer empty or cut off. Keep thinking short and
    # leave extra room for the answer.
    if "gpt-oss" in model.lower():
        payload["reasoning_effort"] = "low"
        budget = min(max_tokens * 3, 2000)

    payload["max_tokens"] = budget

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    with httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10)) as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        raise _http_error(name, response)

    return _extract_text(response.json())


def _groq(system: str, user: str, max_tokens: int) -> str:
    return _with_retries(
        "groq",
        lambda: _openai_compat(
            "Groq",
            "https://api.groq.com/openai/v1/chat/completions",
            GROQ_API_KEY,
            GROQ_MODEL,
            system,
            user,
            max_tokens,
        ),
    )


def _openrouter(system: str, user: str, max_tokens: int) -> str:
    return _with_retries(
        "openrouter",
        lambda: _openai_compat(
            "OpenRouter",
            "https://openrouter.ai/api/v1/chat/completions",
            OPENROUTER_API_KEY,
            OPENROUTER_MODEL,
            system,
            user,
            max_tokens,
            extra_headers={"X-Title": "AriaPsi"},
        ),
    )


def _cerebras(system: str, user: str, max_tokens: int) -> str:
    return _with_retries(
        "cerebras",
        lambda: _openai_compat(
            "Cerebras",
            "https://api.cerebras.ai/v1/chat/completions",
            CEREBRAS_API_KEY,
            CEREBRAS_MODEL,
            system,
            user,
            max_tokens,
        ),
    )


# ---------------------------------------------------------------- routing

def _with_retries(name: str, call: Callable[[], str]) -> str:
    """Retry short-lived problems (overload, rate limit, network) a couple of times."""
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            return call()

        except httpx.HTTPStatusError as error:
            last_error = error
            code = error.response.status_code
            if code not in RETRY_STATUS or attempt == RETRIES:
                raise
            print(f"AI router: {name} HTTP {code}, retry {attempt}/{RETRIES - 1}")
            time.sleep(2 * attempt)

        except (httpx.TimeoutException, httpx.RequestError) as error:
            last_error = error
            if attempt == RETRIES:
                raise
            print(f"AI router: {name} network error, retry {attempt}/{RETRIES - 1}")
            time.sleep(2 * attempt)

    assert last_error is not None
    raise last_error


def _available_providers() -> list[tuple[str, Callable[[str, str, int], str]]]:
    table: dict[str, tuple[str, Callable[[str, str, int], str]]] = {}

    if GEMINI_API_KEY:
        table["gemini"] = ("gemini", _gemini)
    if GROQ_API_KEY:
        table["groq"] = ("groq", _groq)
    if OPENROUTER_API_KEY:
        table["openrouter"] = ("openrouter", _openrouter)
    if CEREBRAS_API_KEY:
        table["cerebras"] = ("cerebras", _cerebras)

    ordered = [table[name] for name in PROVIDER_ORDER if name in table]
    # Any configured provider missing from AI_PROVIDER_ORDER goes last.
    ordered += [item for name, item in table.items() if name not in PROVIDER_ORDER]
    # Providers that had temporary trouble earlier in this run are tried last.
    ordered.sort(key=lambda item: item[0] in _slow_providers)
    return ordered


def llm_text(client: bool, system: str, user: str, max_output_tokens: int = 220) -> str:
    del client

    providers = _available_providers()

    if not providers:
        raise RuntimeError(
            "No AI provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, "
            "OPENROUTER_API_KEY, or CEREBRAS_API_KEY."
        )

    configured = [name for name, _ in providers]
    print(f"AI router: providers in order = {configured}")

    last_error: Exception | None = None

    for name, provider in providers:
        if name in _disabled:
            continue

        try:
            print(f"AI router: trying provider={name}")
            result = provider(system, user, max_output_tokens)
            print(f"AI router: provider={name} succeeded")
            return result

        except httpx.HTTPStatusError as error:
            last_error = error
            code = error.response.status_code
            print(f"AI router: provider={name} HTTP {code}; failover")
            _slow_providers.add(name)
            if code in DISABLE_STATUS:
                _disabled.add(name)
                print(f"AI router: provider={name} disabled for the rest of this run")

        except (httpx.TimeoutException, httpx.RequestError) as error:
            last_error = error
            print(f"AI router: provider={name} network error; failover")
            _slow_providers.add(name)

        except Exception as error:  # noqa: BLE001
            last_error = error
            print(f"AI router: provider={name} failed; failover - {error}")

    raise RuntimeError(f"All configured AI providers failed: {last_error!r}")
