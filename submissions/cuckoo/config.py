"""Shared settings. Paths are relative to the repo root."""

CLASSES = ["invoice", "letter", "form", "resume", "report", "memo"]
SEED = 42

RAW_DIR = "data/raw"            # RVL-CDIP parquet + extracted scans (data/raw/scans/*.png)
PARSED_DIR = "data/parsed"      # NDI output, one <doc_id>.json per scan
LABELS_CSV = "data/labels.csv"  # doc_id, true_label, given_label, flipped, split
RESULTS_DIR = "results"

N_PER_CLASS = 50     # 6 x 50 = 300 documents
NOISE_RATE = 0.10    # fraction of labels deliberately flipped
DEV_FRACTION = 1 / 3 # dev split is used only to choose the flag threshold

# Prices in USD. None = unknown: cost shows as "-" instead of a made-up number.
# Drex: read the per-input-token price from console.nace.ai -> Usage -> CSV (cost_usd / input_tokens).
DREX_PRICE_PER_INPUT_TOKEN = None
LLM_PRICE_PER_INPUT_TOKEN = 0.0   # local Ollama: no per-token charge
LLM_PRICE_PER_OUTPUT_TOKEN = 0.0
