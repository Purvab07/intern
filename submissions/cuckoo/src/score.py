"""Score both models' label-error flags and write results/metrics.json.

  python src/score.py

Flag scores (labels are used here, after the models have run):
  Drex: flag_score = 1 - P_drex(given_label)        (probability mass against the given label)
  LLM:  flag_score = confidence/100 if LLM label != given_label else 0   (null label -> 0)
A doc is flagged when flag_score >= threshold. Each model's threshold is chosen on the dev
split (max F1, ties -> higher threshold); every reported number is on the test split only.
budget_curve[K] = number of planted errors among the K highest-scored test docs.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99
BUDGETS = (5, 10, 20)


def load_jsonl(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"missing {p}; run src/run_eval.py first")
    return {r["doc_id"]: r for r in map(json.loads, filter(str.strip, p.read_text().splitlines()))}


def pct(xs: list, q: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def counts(ids, score, flipped, t):
    flagged = [d for d in ids if score[d] >= t]
    tp = sum(flipped[d] for d in flagged)
    return tp, len(flagged) - tp


def f1(tp, fp, n_pos):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + (n_pos - tp))


def pick_threshold(dev, score, flipped):
    n_pos = sum(flipped[d] for d in dev)
    best = max(GRID, key=lambda t: (f1(*counts(dev, score, flipped, t), n_pos), t))
    return best


def found_at_budget(ids, score, flipped, k):
    """Expected planted errors among the top-k docs, breaking score ties at random.
    (The LLM reports confidence 100 on most docs, so a fixed tie order would be arbitrary.)"""
    found, taken = 0.0, 0
    for s in sorted({score[d] for d in ids}, reverse=True):
        group = [d for d in ids if score[d] == s]
        room = k - taken
        if room <= 0:
            break
        if len(group) <= room:
            found += sum(flipped[d] for d in group)
            taken += len(group)
        else:
            found += room * sum(flipped[d] for d in group) / len(group)
            taken = k
    return round(found, 1)


def side(name, test, dev, score, flipped, rows, cost_per_doc, top_ok):
    t = pick_threshold(dev, score, flipped)
    tp, fp = counts(test, score, flipped, t)
    n_pos = sum(flipped[d] for d in test)
    lat = [rows[d]["latency_ms"] for d in test if d in rows]
    return {
        "threshold": t,
        "errors_found": tp,
        "false_alarms": fp,
        "recall": round(tp / n_pos, 4) if n_pos else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "budget_curve": {str(k): found_at_budget(test, score, flipped, k) for k in BUDGETS},
        "curve": [found_at_budget(test, score, flipped, k) for k in range(0, min(60, len(test)) + 1)],
        "n_flagged": tp + fp,
        "latency_p50_ms": pct(lat, 0.50),
        "latency_p95_ms": pct(lat, 0.95),
        "cost_per_1k_docs": None if cost_per_doc is None else round(cost_per_doc * 1000, 6),
        "accuracy_vs_true": round(sum(top_ok(d) for d in test) / len(test), 4),
        "n_scored": sum(d in rows for d in test),
    }


def main() -> None:
    labels = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / config.LABELS_CSV))}
    res = ROOT / config.RESULTS_DIR
    drex, llm = load_jsonl(res / "drex.jsonl"), load_jsonl(res / "llm.jsonl")
    ids = [d for d in labels if d in drex and d in llm]
    missing = len(labels) - len(ids)
    if missing:
        print(f"WARNING {missing} docs missing from drex/llm results; scoring the rest")
    given = {d: labels[d]["given_label"] for d in ids}
    true = {d: labels[d]["true_label"] for d in ids}
    flipped = {d: labels[d]["flipped"] == "True" for d in ids}
    dev = [d for d in ids if labels[d]["split"] == "dev"]
    test = [d for d in ids if labels[d]["split"] == "test"]

    s_drex = {d: 1 - drex[d]["probs"][given[d]] for d in ids}
    s_llm = {d: ((llm[d]["confidence"] or 0) / 100
                 if llm[d]["label"] is not None and llm[d]["label"] != given[d] else 0.0) for d in ids}
    top = {d: max(drex[d]["probs"], key=drex[d]["probs"].get) for d in ids}

    p_in = config.DREX_PRICE_PER_INPUT_TOKEN
    drex_cost = None if p_in is None else statistics.mean(drex[d]["input_tokens"] for d in test) * p_in
    llm_cost = statistics.mean(llm[d]["input_tokens"] * config.LLM_PRICE_PER_INPUT_TOKEN
                               + llm[d]["output_tokens"] * config.LLM_PRICE_PER_OUTPUT_TOKEN
                               for d in test)

    m = {
        "n_docs": len(ids),
        "n_test": len(test),
        "n_dev": len(dev),
        "n_planted_errors_test": sum(flipped[d] for d in test),
        "n_planted_errors_total": sum(flipped.values()),
        "n_planted_errors_dev": sum(flipped[d] for d in dev),
        "noise_rate": config.NOISE_RATE,
        "classes": config.CLASSES,
        "llm_conf_100_count": sum(1 for d in ids if llm[d]["confidence"] == 100),
        "threshold": None,
        "drex": side("drex", test, dev, s_drex, flipped, drex, drex_cost, lambda d: top[d] == true[d]),
        "llm": side("llm", test, dev, s_llm, flipped, llm, llm_cost,
                    lambda d: llm[d]["label"] == true[d]),
        "top_flags": [
            {"doc_id": d, "given_label": given[d], "drex_top": top[d], "true_label": true[d],
             "drex_probs": {c: round(drex[d]["probs"][c], 4) for c in config.CLASSES},
             "flag_score": round(s_drex[d], 4), "flipped": flipped[d]}
            for d in sorted(test, key=lambda d: s_drex[d], reverse=True)[:20]],
        "notes": {"flag_score_drex": "1 - P(given_label)",
                  "flag_score_llm": "confidence/100 if label != given_label else 0",
                  "threshold_chosen_on": "dev split (max F1)", "reported_on": "test split",
                  "budget_curve": "planted errors in the top-K flags (expected value, score ties "
                                  "broken at random)",
                  "cost": "null = price unknown"},
    }
    # Per-category view and per-document speed, over all docs both models ran on.
    llm_lab = {d: llm[d]["label"] for d in ids}
    m["per_class"] = {c: {
        "n_true": sum(true[d] == c for d in ids),
        "drex_pred": sum(top[d] == c for d in ids),
        "llm_pred": sum(llm_lab[d] == c for d in ids),
        "drex_correct": sum(true[d] == c and top[d] == c for d in ids),
        "llm_correct": sum(true[d] == c and llm_lab[d] == c for d in ids),
    } for c in config.CLASSES}
    m["overall_accuracy"] = {"drex": round(sum(top[d] == true[d] for d in ids) / len(ids), 4),
                             "llm": round(sum(llm_lab[d] == true[d] for d in ids) / len(ids), 4)}
    dl = [drex[d]["latency_ms"] for d in ids]
    ll = [llm[d]["latency_ms"] for d in ids]
    m["speed"] = {
        "drex_ms": dl, "llm_ms": ll,
        "drex_mean_ms": round(statistics.mean(dl)), "llm_mean_ms": round(statistics.mean(ll)),
        "drex_median_ms": statistics.median(dl), "llm_median_ms": statistics.median(ll),
        "drex_faster_count": sum(drex[d]["latency_ms"] < llm[d]["latency_ms"] for d in ids),
        "n": len(ids),
        # does answer time grow with page length? split at the median Drex input-token count
        "long_page_tokens": statistics.median(drex[d]["input_tokens"] for d in ids),
        "short_pages": {"drex_median_ms": statistics.median(drex[d]["latency_ms"] for d in ids if drex[d]["input_tokens"] <= statistics.median(drex[x]["input_tokens"] for x in ids)),
                        "llm_median_ms": statistics.median(llm[d]["latency_ms"] for d in ids if drex[d]["input_tokens"] <= statistics.median(drex[x]["input_tokens"] for x in ids))},
        "long_pages": {"drex_median_ms": statistics.median(drex[d]["latency_ms"] for d in ids if drex[d]["input_tokens"] > statistics.median(drex[x]["input_tokens"] for x in ids)),
                       "llm_median_ms": statistics.median(llm[d]["latency_ms"] for d in ids if drex[d]["input_tokens"] > statistics.median(drex[x]["input_tokens"] for x in ids))},
        "note": "Drex: hosted API over the network. LLM: llama3.2 on this laptop via Ollama, one request at a time.",
    }
    m["threshold"] = m["drex"]["threshold"]
    out = res / "metrics.json"
    out.write_text(json.dumps(m, indent=2))
    d, l = m["drex"], m["llm"]
    print(f"test docs {m['n_test']}, planted errors {m['n_planted_errors_test']}")
    for name, s in (("drex", d), ("llm ", l)):
        print(f"{name}: t={s['threshold']} found {s['errors_found']} | false alarms {s['false_alarms']} | "
              f"P={s['precision']} R={s['recall']} | top5/10/20 {list(s['budget_curve'].values())} | "
              f"acc {s['accuracy_vs_true']} | p50 {s['latency_p50_ms']} ms")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
