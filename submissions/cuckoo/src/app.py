"""Cuckoo demo app: serves the demo page and analyses uploaded documents live.

  python src/app.py            -> http://localhost:8000/demo/

POST /api/analyze  JSON {"name": "scan.pdf", "data": "<base64>", "filed_as": "memo" | ""}
  -> NDI parse (page 1) -> Drex and the local LLM classify the text (in parallel)
  -> {"drex": {...}, "llm": {...}, "ndi_ms", "text_preview", ...}
The filed_as label is NOT sent to either model; it is only echoed back for the verdict.
The API key stays on this server; the browser never sees it.
"""
import asyncio
import base64
import re
import json
import os
import sys
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import drex_client  # noqa: E402
import llm_client  # noqa: E402
from ingest_ndi import parse_one  # noqa: E402

UPLOADS = ROOT / "data" / "uploads"
MIME = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".tif": "image/tiff", ".tiff": "image/tiff", ".webp": "image/webp"}
MAX_BYTES = 15 * 1024 * 1024
PDF_PAGES = 3            # read the first 3 pages of a PDF (a cover page alone can mislead)
MAX_CHARS = 15000        # keep the text within the local LLM's context window
# Live uploads can be anything, so both models may also answer "other" instead of forcing one of the six.
LIVE_CLASSES = config.CLASSES + ["other"]
DREX_DESCRIPTIONS = {"other": "none of the listed document types"}

# One long-lived event loop: the model clients keep async HTTP clients bound to it.
LOOP = asyncio.new_event_loop()
threading.Thread(target=LOOP.run_forever, daemon=True).start()
_ndi: dict = {}


async def _ndi_clients():
    if not _ndi:
        _ndi["c"] = httpx.AsyncClient(headers={"Authorization": f"Bearer {os.environ['NACE_API_KEY']}"}, timeout=120)
        _ndi["up"] = httpx.AsyncClient(timeout=120)  # upload host gets only its one-time token
    return _ndi["c"], _ndi["up"]


async def analyze(name: str, raw: bytes, filed_as: str) -> dict:
    ext = Path(name).suffix.lower()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"upload_{uuid.uuid4().hex[:12]}{ext}"  # neutral name: never hints at a class
    path.write_bytes(raw)
    c, up = await _ndi_clients()
    t0 = time.perf_counter()
    parsed = await parse_one(c, up, path, mime=MIME[ext], max_pages=PDF_PAGES if ext == ".pdf" else None)
    ndi_ms = int((time.perf_counter() - t0) * 1000)
    text = re.sub(r"^--- Page \d+ ---\s*$", "", parsed["text"], flags=re.M).strip()
    truncated = len(text) > MAX_CHARS
    text = text[:MAX_CHARS]
    out = {"name": name, "filed_as": filed_as or None, "ndi_ms": ndi_ms, "pages_read": parsed.get("page_count"),
           "truncated": truncated, "text_preview": text[:700], "empty_text": not text.strip()}
    if not text.strip():
        return out

    async def run_drex():
        try:
            r = await drex_client.classify(text, LIVE_CLASSES, DREX_DESCRIPTIONS)
            top = max(r["probs"], key=r["probs"].get)
            return {"top": top, "confidence": r["probs"][top], "probs": r["probs"], "latency_ms": r["latency_ms"],
                    "server_ms": r.get("server_ms"), "input_tokens": r.get("input_tokens"), "model": r.get("model")}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    async def run_llm():
        try:
            r = await llm_client.classify(text, LIVE_CLASSES)
            return {"top": r["label"], "confidence": None if r["confidence"] is None else r["confidence"] / 100,
                    "latency_ms": r["latency_ms"], "model": r.get("model", llm_client.MODEL),
                    "provider": llm_client.PROVIDER, "phases": r.get("phases"), "attempts": r.get("attempts"),
                    "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens")}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:160]}", "model": llm_client.MODEL}

    out["drex"], out["llm"] = await asyncio.gather(run_drex(), run_llm())
    return out


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (self.path or ""):
            sys.stderr.write("%s\n" % (fmt % args))

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/demo/")
            self.end_headers()
            return
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "classes": config.CLASSES, "llm_model": llm_client.MODEL,
                                    "llm_provider": llm_client.PROVIDER, "frontier": llm_client.PROVIDER == "anthropic"})
        # Never serve hidden files (.env, .git, ...), however the path is spelled.
        fs = Path(self.translate_path(self.path)).resolve()
        try:
            rel = fs.relative_to(ROOT)
        except ValueError:
            return self._json(404, {"error": "not found"})
        if any(part.startswith(".") for part in rel.parts) or rel.parts[:2] == ("data", "uploads"):
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_HEAD(self):
        return self._json(405, {"error": "method not allowed"})

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BYTES * 1.4:
                return self._json(413, {"error": "File too large (max 15 MB)."})
            req = json.loads(self.rfile.read(n))
            name = str(req.get("name", "upload"))
            if Path(name).suffix.lower() not in MIME:
                return self._json(400, {"error": "Use a PDF, PNG, JPG, TIFF or WEBP file."})
            raw = base64.b64decode(req.get("data", ""), validate=True)
            if not raw or len(raw) > MAX_BYTES:
                return self._json(400, {"error": "Empty or too large file (max 15 MB)."})
            filed = req.get("filed_as") or ""
            if filed and filed not in config.CLASSES:
                filed = ""
            fut = asyncio.run_coroutine_threadsafe(analyze(name, raw, filed), LOOP)
            return self._json(200, fut.result(timeout=240))
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT)))
    print(f"Cuckoo running: http://localhost:{port}/demo/   (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
