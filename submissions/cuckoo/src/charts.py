"""Budget curve: planted errors caught vs. number of flags a reviewer checks (test split).

  python src/charts.py   -> results/charts/budget_curve.png

Uses the same flag scores and labels as score.py. Two series, fixed colours
(Drex = categorical slot 1 blue, LLM = slot 2 orange), legend + direct end labels.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def main() -> None:
    labels = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / config.LABELS_CSV))}
    res = ROOT / config.RESULTS_DIR
    load = lambda p: {r["doc_id"]: r for r in map(json.loads, filter(str.strip, p.read_text().splitlines()))}
    drex, llm = load(res / "drex.jsonl"), load(res / "llm.jsonl")
    test = [d for d, r in labels.items() if r["split"] == "test" and d in drex and d in llm]
    given = {d: labels[d]["given_label"] for d in test}
    flipped = {d: labels[d]["flipped"] == "True" for d in test}
    s_drex = {d: 1 - drex[d]["probs"][given[d]] for d in test}
    s_llm = {d: ((llm[d]["confidence"] or 0) / 100
                 if llm[d]["label"] is not None and llm[d]["label"] != given[d] else 0.0) for d in test}
    n_err = sum(flipped.values())
    k_max = min(60, len(test))

    sys.path.insert(0, str(ROOT / "src"))
    from score import found_at_budget  # same tie handling as metrics.json

    def curve(score):
        return [found_at_budget(test, score, flipped, k) for k in range(k_max + 1)]

    xs = list(range(k_max + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for ys, color, name in ((curve(s_drex), BLUE, "Drex"), (curve(s_llm), ORANGE, f"Local LLM ({config_model()})")):
        ax.plot(xs, ys, color=color, lw=2, label=name, solid_capstyle="round")
        ax.annotate(f"{name.split(' (')[0]}: {ys[-1]:g}", (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.axhline(n_err, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(1, n_err, f"  {n_err} planted errors in test set", va="bottom", color=INK2, fontsize=8)
    ax.set_xlim(0, k_max + 9)
    ax.set_ylim(0, n_err + 2)
    ax.set_xlabel("Documents reviewed (highest flag score first)", color=INK2, fontsize=9)
    ax.set_ylabel("Planted errors found", color=INK2, fontsize=9)
    ax.set_title(f"Label errors caught per review budget ({len(test)} test documents)",
                 loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    out = res / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "budget_curve.png", facecolor=SURFACE)
    print(f"-> {out / 'budget_curve.png'}")


def config_model() -> str:
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return os.getenv("LOCAL_MODEL", "llama3.2")


if __name__ == "__main__":
    main()
