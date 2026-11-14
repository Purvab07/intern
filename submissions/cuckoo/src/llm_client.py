"""LLM baseline: local Ollama model by default, or a frontier model (Claude) when configured.

Returns {"label": str|None, "confidence": int|None, "latency_ms", "input_tokens", "output_tokens",
         "phases": {...ms...}, "model": str, "attempts": int}

  Local (default):  needs `ollama serve` and `ollama pull <LOCAL_MODEL>`.
  Frontier:         set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY (or LLM_API_KEY) in .env.
Run directly for a smoke test:  python src/llm_client.py
"""
import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv
from tenacity import (AsyncRetrying, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
FRONTIER_MODEL = "claude-opus-5-5"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2")
MODEL = FRONTIER_MODEL if PROVIDER == "anthropic" else LOCAL_MODEL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TIMEOUT_S = 120.0   # local generation is slower than a hosted API, and requests queue
NUM_CTX = 8192      # Ollama's default context can silently cut long pages
PROMPT = ('Classify this document as exactly one of: {classes}. Reply with JSON only:\n'
          '{{"label": "<one of the classes>", "confidence": <0-100>}}. Document: {text}')

_client: httpx.AsyncClient | None = None
_anthropic = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=TIMEOUT_S)
    return _client


def _schema(classes: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": classes},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["label", "confidence"],
    }


def _parse(raw: str, classes: list[str]) -> tuple[str, int]:
    raw = raw.strip()
    if raw.startswith("```"):  # tolerate a fenced JSON block
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    obj = json.loads(raw.strip())
    label, conf = obj["label"], obj["confidence"]
    if label not in classes:
        raise ValueError(f"label {label!r} not in classes")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 100:
        raise ValueError(f"bad confidence {conf!r}")
    return label, int(round(conf))


async def _chat(body: dict) -> dict:
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential_jitter(initial=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    ):
        with attempt:
            r = await _get_client().post("/api/chat", json=body)
            if r.status_code == 404:
                raise RuntimeError(f"model {LOCAL_MODEL!r} not found; run: ollama pull {LOCAL_MODEL}")
            r.raise_for_status()
            return r.json()


async def _classify_ollama(text: str, classes: list[str]) -> dict:
    body = {
        "model": LOCAL_MODEL,
        "messages": [{"role": "user",
                      "content": PROMPT.format(classes=", ".join(classes), text=text)}],
        "format": _schema(classes),
        "stream": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    in_tok = out_tok = attempts = 0
    ph = {"load_ms": 0.0, "read_ms": 0.0, "write_ms": 0.0}
    t0 = time.perf_counter()
    label = conf = None
    for _ in range(2):  # one try + one retry on parse failure
        attempts += 1
        data = await _chat(body)
        p_tok = int(data.get("prompt_eval_count", 0))
        if p_tok >= NUM_CTX - 64:
            print(f"[llm] WARNING prompt hit context limit ({p_tok}/{NUM_CTX}); Ollama may have cut the text")
        in_tok += p_tok
        out_tok += int(data.get("eval_count", 0))
        # Ollama reports its own phase timings in nanoseconds
        ph["load_ms"] += data.get("load_duration", 0) / 1e6
        ph["read_ms"] += data.get("prompt_eval_duration", 0) / 1e6
        ph["write_ms"] += data.get("eval_duration", 0) / 1e6
        raw = data.get("message", {}).get("content", "")
        try:
            label, conf = _parse(raw, classes)
            break
        except (ValueError, KeyError, TypeError) as e:  # json.JSONDecodeError is a ValueError
            print(f"[llm] parse failure ({e}); raw={raw[:120]!r}")
    latency = int((time.perf_counter() - t0) * 1000)
    ph = {k: round(v) for k, v in ph.items()}
    ph["other_ms"] = max(0, latency - sum(ph.values()))
    return {"label": label, "confidence": conf, "latency_ms": latency, "input_tokens": in_tok,
            "output_tokens": out_tok, "phases": ph, "model": LOCAL_MODEL, "attempts": attempts}


async def _classify_anthropic(text: str, classes: list[str]) -> dict:
    global _anthropic
    import anthropic
    if _anthropic is None:
        _anthropic = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY") or os.environ["LLM_API_KEY"], timeout=120.0, max_retries=3)
    msg = PROMPT.format(classes=", ".join(classes), text=text)
    in_tok = out_tok = attempts = 0
    ph = {"wait_first_word_ms": 0.0, "write_ms": 0.0}
    t0 = time.perf_counter()
    label = conf = None
    for _ in range(2):
        attempts += 1
        ts = time.perf_counter()
        first = None
        # Opus 5.5 rejects temperature; low effort keeps a classification call quick.
        async with _anthropic.messages.stream(model=FRONTIER_MODEL, max_tokens=2048,
                                              output_config={"effort": "low"},
                                              messages=[{"role": "user", "content": msg}]) as stream:
            async for _chunk in stream.text_stream:
                if first is None:
                    first = time.perf_counter()
            final = await stream.get_final_message()
        end = time.perf_counter()
        first = first or end
        ph["wait_first_word_ms"] += (first - ts) * 1000
        ph["write_ms"] += (end - first) * 1000
        in_tok += final.usage.input_tokens
        out_tok += final.usage.output_tokens
        if final.stop_reason == "refusal":
            break
        raw = "".join(b.text for b in final.content if b.type == "text")
        try:
            label, conf = _parse(raw, classes)
            break
        except (ValueError, KeyError, TypeError) as e:
            print(f"[llm] parse failure ({e}); raw={raw[:120]!r}")
    latency = int((time.perf_counter() - t0) * 1000)
    ph = {k: round(v) for k, v in ph.items()}
    return {"label": label, "confidence": conf, "latency_ms": latency, "input_tokens": in_tok,
            "output_tokens": out_tok, "phases": ph, "model": FRONTIER_MODEL, "attempts": attempts}


async def classify(text: str, classes: list[str]) -> dict:
    if PROVIDER == "anthropic":
        return await _classify_anthropic(text, classes)
    return await _classify_ollama(text, classes)


if __name__ == "__main__":
    demo = "INVOICE #4411\nBill to: Acme Corp\nAmount due: $1,200.00\nDue date: 2026-12-01"
    print(MODEL, asyncio.run(classify(demo, ["invoice", "letter", "form", "resume", "report", "memo"])))
