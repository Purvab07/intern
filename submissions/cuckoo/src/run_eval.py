"""python src/run_eval.py --model drex|llm [--limit N] [--concurrency 8]

Appends one JSON line per document to results/<model>.jsonl; skips doc_ids already present.
Only doc["text"] is sent to a model. Labels are never read here.
"""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

try:
    import config  # noqa: E402  (Person B's file)
except ModuleNotFoundError:
    sys.exit("config.py not found in the repo root. It comes from Person B's `data` branch: "
             "git pull origin main (after B merges), then re-run.")


def cfg_path(x) -> Path:
    """config paths may be relative; resolve them against the repo root, not the cwd."""
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


def out_path(model: str) -> Path:
    return cfg_path(config.RESULTS_DIR) / f"{model}.jsonl"


def done_ids(path: Path) -> set[str]:
    ids = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    ids.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # ignore a torn last line from an interrupted run
    return ids


def to_line(model: str, doc_id: str, r: dict) -> dict:
    if model == "drex":
        return {"doc_id": doc_id, "probs": r["probs"], "latency_ms": r["latency_ms"],
                "input_tokens": r["input_tokens"]}
    return {"doc_id": doc_id, "label": r["label"], "confidence": r["confidence"],
            "latency_ms": r["latency_ms"], "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"]}


async def main(model: str, limit: int | None, concurrency: int) -> None:
    if model == "drex":
        from drex_client import classify
    else:
        from llm_client import classify

    out = out_path(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    # If a previous run was killed mid-write, end the torn line so the next append stays valid.
    if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
        with out.open("a") as fh:
            fh.write("\n")
    skip = done_ids(out)
    parsed_dir = cfg_path(config.PARSED_DIR)
    files = sorted(parsed_dir.glob("*.json"))
    if not files:
        print(f"  WARNING no parsed documents found in {parsed_dir}")
    todo, unreadable, dup_ids, seen = [], [], [], set()
    for f in files:
        try:  # B may still be writing a file; skip it now, the next run picks it up
            doc = json.loads(f.read_text())
            doc["doc_id"], doc["text"]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError, TypeError):
            unreadable.append(f.name)
            continue
        if doc["doc_id"] in seen:  # two files with the same doc_id would make duplicate lines
            dup_ids.append(doc["doc_id"])
            continue
        seen.add(doc["doc_id"])
        if doc["doc_id"] not in skip:
            todo.append(doc)
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(files)} parsed, {len(skip)} already done, {len(todo)} to run ({model})")
    if unreadable:
        print(f"  WARNING skipped {len(unreadable)} unreadable/partial files: {unreadable[:10]}")
    if dup_ids:
        print(f"  WARNING {len(dup_ids)} duplicate doc_ids across parsed files, kept the first: "
              f"{dup_ids[:10]}")
    empty = [d["doc_id"] for d in todo if not str(d["text"]).strip()]
    if empty:
        print(f"  WARNING {len(empty)} docs have empty text (models will still guess; tell B): "
              f"{empty[:10]}")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    latencies: list[int] = []
    failures: list[str] = []
    count = 0

    finished = 0

    def progress() -> None:
        nonlocal finished
        finished += 1
        if finished % 10 == 0 or finished == len(todo):
            print(f"  {finished}/{len(todo)}")

    async def one(doc: dict) -> None:
        nonlocal count
        async with sem:
            try:
                r = await classify(doc["text"], config.CLASSES)
            except Exception as e:  # keep going; failed docs are retried on the next run
                failures.append(doc["doc_id"])
                print(f"  FAIL {doc['doc_id']}: {type(e).__name__}: {str(e)[:150]}")
                progress()
                return
        async with lock:
            with out.open("a") as fh:
                fh.write(json.dumps(to_line(model, doc["doc_id"], r)) + "\n")
            latencies.append(r["latency_ms"])
            if model == "llm" and r["label"] is None:
                failures.append(doc["doc_id"])
            count += 1
            progress()

    await asyncio.gather(*(one(d) for d in todo))
    med = statistics.median(latencies) if latencies else None
    print(f"\nDone: {count} written, {len(failures)} failures "
          f"(llm null labels count as failures), median latency {med} ms")
    print(f"{out}: {len(done_ids(out))} unique doc_ids total")


def check(model: str) -> None:
    """Task 5 check: line count, duplicate doc_ids, null probs/labels."""
    out = out_path(model)
    if not out.exists():
        sys.exit(f"{out} does not exist yet; run: python src/run_eval.py --model {model}")
    rows, corrupt = [], 0
    for line in out.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                corrupt += 1
    ids = [r.get("doc_id") for r in rows]
    dups = len(ids) - len(set(ids))
    if model == "drex":
        bad = [r.get("doc_id") for r in rows
               if not isinstance(r.get("probs"), dict)
               or any(r["probs"].get(c) is None for c in config.CLASSES)]
    else:
        bad = [r.get("doc_id") for r in rows if r.get("label") is None]
    n_parsed = len(list(cfg_path(config.PARSED_DIR).glob("*.json")))
    ok = len(set(ids)) == n_parsed and not dups and not bad and not corrupt
    print(f"{out.name}: {len(rows)} lines (parsed docs: {n_parsed}), {dups} duplicate doc_ids, "
          f"{len(bad)} null {'probs' if model == 'drex' else 'labels'} {bad[:10]}, "
          f"{corrupt} corrupt lines -> {'OK' if ok else 'NOT OK'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["drex", "llm"], required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int,
                    help="default 8 for drex; 1 for llm (local Ollama runs requests one at a "
                         "time, so more just adds queue time to latency_ms)")
    ap.add_argument("--check", action="store_true", help="validate the output file and exit")
    a = ap.parse_args()
    if a.check:
        check(a.model)
    else:
        conc = a.concurrency or (8 if a.model == "drex" else 1)
        asyncio.run(main(a.model, a.limit, conc))
