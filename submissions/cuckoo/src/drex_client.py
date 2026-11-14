"""Drex (NACE) client: one choice question per document, returns per-class probabilities.

API per https://console.nace.ai/docs/drex/quickstart :
  POST https://console.nace.ai/v1/systemone
  body {"model", "state": <text>, "questions": {id: {"type": "choice", "instructions", "criteria": {label: desc}}}}
  resp answers[id].probabilities {label: p}, usage.input_tokens (billed), evaluation_time_ms

Run directly:  python src/drex_client.py --check   (key check, free)
               python src/drex_client.py [N] [--true-col COL]   (go/no-go on first N docs)
"""
import asyncio
import os
import time

import httpx
from dotenv import load_dotenv
from tenacity import (AsyncRetrying, retry_if_exception, stop_after_attempt,
                      wait_exponential_jitter)

load_dotenv()

BASE_URL = os.getenv("NACE_BASE_URL", "https://console.nace.ai")
MODEL = "drex-latest"
QUESTION = "What type of document is this?"
TIMEOUT_S = 30.0
RETRYABLE = {429, 500, 503, 529}

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        key = os.environ["NACE_API_KEY"]
        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=TIMEOUT_S,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
    return _client


class RetryableError(Exception):
    def __init__(self, msg: str, wait_s: float = 0.0):
        super().__init__(msg)
        self.wait_s = wait_s


def _is_retryable(e: BaseException) -> bool:
    return isinstance(e, (RetryableError, httpx.TimeoutException, httpx.TransportError))


_backoff = wait_exponential_jitter(initial=1, max=20)


def _wait(retry_state) -> float:
    """Exponential backoff, but never shorter than the server's retry-after-ms hint."""
    hint = getattr(retry_state.outcome.exception(), "wait_s", 0.0)
    return max(hint, _backoff(retry_state))


async def _post(body: dict) -> dict:
    r = await _get_client().post("/v1/systemone", json=body)
    if r.status_code in RETRYABLE:
        try:
            wait_s = float(r.headers.get("retry-after-ms") or 0) / 1000
        except ValueError:
            wait_s = 0.0
        raise RetryableError(f"HTTP {r.status_code}: {r.text[:200]}", wait_s)
    if r.status_code != 200:
        raise RuntimeError(f"Drex HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


async def classify(text: str, classes: list[str], descriptions: dict | None = None) -> dict:
    """Return {"probs": {class: float}, "latency_ms": int, "input_tokens": int}."""
    body = {
        "model": MODEL,
        "state": text,
        "questions": {
            "doc_type": {
                "type": "choice",
                "instructions": QUESTION,
                "criteria": {c: (descriptions or {}).get(c) for c in classes},
            }
        },
    }
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable),
        wait=_wait,
        stop=stop_after_attempt(5),
        reraise=True,
    ):
        with attempt:
            t0 = time.perf_counter()  # time the successful request only, not retry waits
            data = await _post(body)
            latency_ms = int((time.perf_counter() - t0) * 1000)

    raw = data["answers"]["doc_type"]["probabilities"]
    missing = [c for c in classes if c not in raw]
    if missing:
        raise RuntimeError(f"Drex response missing classes {missing}")
    total = sum(float(raw[c]) for c in classes)
    if total <= 0:
        raise RuntimeError("Drex probabilities sum to 0")
    probs = {c: float(raw[c]) / total for c in classes}  # renormalise rounding drift
    return {
        "probs": probs,
        "latency_ms": latency_ms,
        "input_tokens": int(data["usage"]["input_tokens"]),
        "server_ms": data.get("evaluation_time_ms"),  # Drex's own compute time for this request
        "model": data.get("model"),
    }


TRUE_COL_CANDIDATES = ("true_label", "true_class", "clean_label", "original_label", "true")


async def _go_no_go(n: int, true_col: str | None) -> None:
    """Classify the first n parsed docs and score against the TRUE class in labels.csv.
    labels.csv is read only for scoring here; nothing from it is sent to Drex."""
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    try:
        import config
    except ModuleNotFoundError:
        sys.exit("config.py not found in the repo root; pull Person B's work first.")

    def p(x) -> Path:  # config paths may be relative to the repo root
        x = Path(x)
        return x if x.is_absolute() else root / x

    labels = pd.read_csv(p(config.LABELS_CSV))
    if true_col is None:
        found = [c for c in TRUE_COL_CANDIDATES if c in labels.columns]
        others = [c for c in labels.columns if c != "doc_id"]
        if found:
            true_col = found[0]
        elif len(others) == 1:
            true_col = others[0]  # only one label column: no noise applied yet
        else:
            sys.exit(f"Can't tell which labels.csv column is the TRUE class: {list(labels.columns)}.\n"
                     f"Re-run with --true-col <name> (not the noisy/given label column).")
    if true_col not in labels.columns:
        sys.exit(f"--true-col {true_col!r} not in labels.csv columns {list(labels.columns)}")
    truth = dict(zip(labels["doc_id"].astype(str), labels[true_col].astype(str)))
    print(f"(scoring against labels.csv column: {true_col!r})")

    files = sorted(p(config.PARSED_DIR).glob("*.json"))[:n]
    if not files:
        sys.exit(f"No parsed documents in {p(config.PARSED_DIR)}")
    right = unknown = 0
    for f in files:
        doc = json.loads(f.read_text())
        res = await classify(doc["text"], config.CLASSES)
        top = max(res["probs"], key=res["probs"].get)
        true = truth.get(str(doc["doc_id"]))
        unknown += true is None
        right += top == true
        print(f"{str(doc['doc_id']):<20} top={top:<14} p={res['probs'][top]:.3f}  true={true or '?'}"
              f"  {'OK' if top == true else 'MISS'}  {res['latency_ms']}ms")
    if unknown:
        print(f"WARNING: {unknown} doc_ids not found in labels.csv (counted as MISS)")
    print(f"\n{right}/{len(files)} correct -> {'GO' if right >= 0.8 * len(files) else 'NO-GO'}")


async def _key_check() -> None:
    """GET /v1/models: confirms the key works. Not billed, not rate limited (per docs)."""
    r = await _get_client().get("/v1/models")
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return
    for m in r.json().get("models", []):
        alias = f" -> {m['alias_for']}" if m.get("alias_for") else ""
        print(f"  {m.get('name')}{alias}")
    print("Drex key OK")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Drex key check or go/no-go run")
    ap.add_argument("n", nargs="?", type=int, default=20, help="docs for go/no-go (default 20)")
    ap.add_argument("--check", action="store_true", help="only verify the API key via /v1/models")
    ap.add_argument("--true-col", help="labels.csv column holding the TRUE class (auto-detected)")
    a = ap.parse_args()
    asyncio.run(_key_check() if a.check else _go_no_go(a.n, a.true_col))
