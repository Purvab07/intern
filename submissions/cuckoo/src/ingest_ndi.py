"""Parse scans with NDI (Nace Document Intelligence) into data/parsed/<doc_id>.json.

  python src/ingest_ndi.py [--limit N] [--concurrency 8]

Flow (console.nace.ai/docs/perception): upload grant -> multipart upload -> POST
/v1/documents/parse (wait up to 60 s) -> poll /v1/documents/jobs/{id}. Resumable: skips docs
already parsed. Writes atomically so run_eval never reads a half-written file.
Only the image goes to NDI; the file is named <doc_id>.png and doc_ids are neutral.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

load_dotenv(ROOT / ".env")
BASE = "https://console.nace.ai"
DONE = {"succeeded", "failed", "cancelled"}


async def _req(c: httpx.AsyncClient, method: str, url: str, **kw) -> httpx.Response:
    for attempt in range(6):
        try:
            r = await c.request(method, url, **kw)
        except httpx.TransportError:
            await asyncio.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 529):
            await asyncio.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
            continue
        return r
    raise RuntimeError(f"{method} {url} kept failing")


async def parse_one(c: httpx.AsyncClient, up_c: httpx.AsyncClient, img: Path,
                    mime: str = "image/png", max_pages: int | None = None) -> dict:
    name = img.name
    g = await _req(c, "POST", f"{BASE}/v1/documents/upload-grants",
                   json={"path": f"cuckoo/{name}", "ttl_seconds": 600})
    g.raise_for_status()
    g = g.json()
    with img.open("rb") as fh:
        # upload host gets only its one-time token, never the API key
        up = await _req(up_c, "POST", g["upload_url"], headers={"X-Upload-Token": g["token"]},
                        files={"file": (name, fh, mime)},
                        data={"metadata": json.dumps({"path": f"cuckoo/{name}"})})
    if up.status_code >= 400:
        raise RuntimeError(f"upload {up.status_code}: {up.text[:200]}")
    file_id = up.json()["result"]["file"]["file_id"]
    r = await _req(c, "POST", f"{BASE}/v1/documents/parse", params={"wait_seconds": 60},
                   headers={"Idempotency-Key": f"cuckoo-parse-{img.stem}"},
                   json={"source": {"type": "workspace_file", "workspace_id": g["workspace_id"],
                                    "file_id": file_id},
                         "output": {"formats": ["text"]}, "parse_mode": "low",
                         **({"page_ranges": [{"start": 1, "end": max_pages}]} if max_pages else {})})
    if r.status_code >= 400:
        raise RuntimeError(f"parse {r.status_code}: {r.text[:200]}")
    job = r.json()
    while job.get("status") not in DONE:
        await asyncio.sleep(2)
        jr = await _req(c, "GET", f"{BASE}/v1/documents/jobs/{job['job_id']}")
        jr.raise_for_status()
        job = jr.json()
    if job["status"] != "succeeded":
        raise RuntimeError(f"job {job['status']}: {job.get('error')}")
    doc = job["result"]["document"]
    text = doc.get("text")
    if text is None and doc.get("content_url"):
        text = (await _req(c, "GET", doc["content_url"])).json().get("text")
    return {"text": text or "", "ocr_applied": job["result"].get("ocr_applied"), "page_count": doc.get("page_count"),
            "job_id": job["job_id"], "cost": job.get("cost")}


async def main(limit: int | None, concurrency: int) -> None:
    scans = sorted((ROOT / config.RAW_DIR / "scans").glob("*.png"))
    out_dir = ROOT / config.PARSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in scans if not (out_dir / f"{p.stem}.json").exists()]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(scans)} scans, {len(scans) - len(todo) if limit is None else '?'} already parsed, "
          f"{len(todo)} to parse")
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "fail": 0, "empty": 0}
    headers = {"Authorization": f"Bearer {os.environ['NACE_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers, timeout=120) as c, \
            httpx.AsyncClient(timeout=120) as up_c:
        async def one(p: Path) -> None:
            async with sem:
                try:
                    res = await parse_one(c, up_c, p)
                except Exception as e:
                    stats["fail"] += 1
                    print(f"  FAIL {p.stem}: {type(e).__name__}: {str(e)[:200]}")
                    return
            rec = {"doc_id": p.stem, "text": res["text"],
                   "source": {"tool": "ndi-parse", "mode": "low", "file": p.name,
                              "ocr_applied": res["ocr_applied"], "job_id": res["job_id"]}}
            tmp = out_dir / f".{p.stem}.json.tmp"
            tmp.write_text(json.dumps(rec))
            tmp.rename(out_dir / f"{p.stem}.json")  # atomic: never a half-written .json
            stats["ok"] += 1
            stats["empty"] += not res["text"].strip()
            n = stats["ok"] + stats["fail"]
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")
        await asyncio.gather(*(one(p) for p in todo))
    print(f"Done: {stats['ok']} parsed, {stats['fail']} failed, {stats['empty']} with empty text")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(main(a.limit, a.concurrency))
