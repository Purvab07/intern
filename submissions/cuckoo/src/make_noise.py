"""Build the 300-document dataset and plant label errors.

  python src/make_noise.py

Reads data/raw/train.parquet (RVL-CDIP, 50 scans per class, from
huggingface.co/datasets/jordyvl/rvl_cdip_100_examples_per_class), keeps our six classes,
writes data/raw/scans/<doc_id>.png and data/labels.csv with columns:
  doc_id, true_label, given_label, flipped, split
given_label = true_label except for NOISE_RATE of docs, flipped to a different random class.
"""
import csv
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

URL = ("https://huggingface.co/datasets/jordyvl/rvl_cdip_100_examples_per_class/resolve/main/"
       "data/train-00000-of-00001-81f1d229db782541.parquet")
RVL = {0: "letter", 1: "form", 5: "report", 11: "invoice", 14: "resume", 15: "memo"}  # RVL id -> ours


def main() -> None:
    raw = ROOT / config.RAW_DIR
    parquet = raw / "train.parquet"
    if not parquet.exists():
        sys.exit(f"Missing {parquet}. Download it:\n  curl -L -o {parquet} {URL}")
    scans = raw / "scans"
    scans.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(parquet).to_pylist()
    by_class: dict[str, list[dict]] = {c: [] for c in config.CLASSES}
    for r in rows:
        c = RVL.get(r["label"])
        if c:
            by_class[c].append(r)

    rng = random.Random(config.SEED)
    picked = []
    for c in config.CLASSES:
        rs = by_class[c][: config.N_PER_CLASS]
        if len(rs) < config.N_PER_CLASS:
            sys.exit(f"only {len(rs)} scans for {c}")
        picked += [(c, r) for r in rs]
    rng.shuffle(picked)  # neutral ids in random order: a doc_id never reveals the class
    docs = []
    for i, (c, r) in enumerate(picked):
        doc_id = f"doc_{i:03d}"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("L").save(scans / f"{doc_id}.png")
        docs.append({"doc_id": doc_id, "true_label": c})

    rng.shuffle(docs)
    n_flip = round(config.NOISE_RATE * len(docs))
    for i, d in enumerate(docs):
        d["flipped"] = i < n_flip
        d["given_label"] = (rng.choice([c for c in config.CLASSES if c != d["true_label"]])
                            if d["flipped"] else d["true_label"])
    rng.shuffle(docs)  # split independently of which docs were flipped
    n_dev = round(config.DEV_FRACTION * len(docs))
    for i, d in enumerate(docs):
        d["split"] = "dev" if i < n_dev else "test"

    docs.sort(key=lambda d: d["doc_id"])
    with open(ROOT / config.LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "true_label", "given_label", "flipped", "split"])
        w.writeheader()
        w.writerows(docs)
    n_dev_flip = sum(d["flipped"] for d in docs if d["split"] == "dev")
    print(f"{len(docs)} scans -> {scans}\n{n_flip} labels flipped "
          f"({n_dev_flip} in dev, {n_flip - n_dev_flip} in test); "
          f"{n_dev} dev / {len(docs) - n_dev} test -> {config.LABELS_CSV}")


if __name__ == "__main__":
    main()
