# Cuckoo: 5-minute pitch script

Spoken at about 150 words per minute. Times in brackets are targets.
**Every figure is from `results/metrics.json`** (run on 2026-11-14, test split of 200 documents).
The source key is shown next to each figure so you can check it. Lines marked ⚠️ need confirming.

---

## 1. Problem (0:00 to 0:20)

> Every document AI system learns from labels, and some labels are wrong: an invoice filed as a letter.
> Like a cuckoo's egg in the nest, a mislabeled document looks like it belongs. It quietly skews training
> and evaluation, and checking every label by hand doesn't scale.

## 2. How it's done today (0:20 to 0:50)

> Today's best approach uses LLMs as judges. At EMNLP 2025, researchers from the Technion and Google
> showed an *ensemble* of LLMs, each run with four prompts, can flag examples where it confidently
> disagrees with the label. Above 95% confidence, over two thirds of those flags were real human errors.
> But that's several models and several prompts for every example.

*Source: Nahum et al., "Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect
on Model Performance", EMNLP 2025 (aclanthology.org/2025.emnlp-main.1360).*

## 3. Our method (0:50 to 1:50)

> We do it in one pass. NDI, Nace Document Intelligence, reads each scanned page into text. Then we ask
> Drex one typed question: "What type of document is this?" with six classes as the options. Drex
> doesn't write an answer we have to parse. It returns a probability for every class, in one call.
>
> And it's blind. Drex never sees the label, only the page. Afterwards we compare: the flag score is
> one minus the probability Drex gives the label the document was filed under. High score, strong
> disagreement.
>
> To measure it, we took 300 real scanned documents from the RVL-CDIP dataset, fifty each of invoices,
> letters, forms, resumes, reports and memos, and deliberately flipped ten percent of the labels. We
> tuned the threshold on 100 of them and tested on the other 200, which hold 21 planted errors.
> The baseline is a local open model, Llama 3.2, given exactly the same text and classes.

*(300 = `n_docs`, 200 = `n_test`, 21 = `n_planted_errors_test`, 100 = `n_dev`)*

## 4. Live demo (1:50 to 3:20)

*Before you start: run `python src/app.py` in the repo and open `http://localhost:8000/demo/`. Keep Ollama
running. Do one practice upload first so the local model is already loaded.*

> **[① Add your files: click "Resume filed as memo", then Analyze]** Here's a scanned page that someone filed
> as a memo. Cuckoo reads it with NDI and asks Drex and the LLM what it is. Neither sees the label.
>
> **[② Drex vs the LLM: the two answer boxes]** Both say resume. But look at how they say it: Drex gives a
> real probability for every option, while the LLM just names one answer and calls itself 100% sure.
>
> **[the timeline]** And here's where the time goes on this exact file. The LLM writes its answer one token
> at a time. Drex doesn't write anything: it scores every option in a single pass. *(Read the two numbers
> off the screen.)*
>
> **[③ The result]** And the verdict, in plain words: probably filed under the wrong label. Check it.
>
> **[back to ①: drop in 3 files at once, e.g. the 3 sample buttons]** It works on a whole dataset too: one
> line per file, a summary of what's mislabeled, and what each model thinks the dataset contains.

*On-screen timings change from run to run, so read them off the page; don't memorise them. If the LLM wins
on a single file, say so: the page explains why (Drex's internet round trip vs a model on this laptop). The
fair speed comparison is the 300-document one in section 5.*

## 5. Numbers (3:20 to 4:20)

> Of the 21 planted errors, Drex flagged 18, with 7 false alarms: 72% precision and 86% recall.
> The baseline flagged 16, but with 61 false alarms, about 21% precision.
>
> On a fixed budget it's starker. If a reviewer checks only the top 20 flags, Drex hands them 15 real
> errors. The baseline hands them about 4.
>
> And it's fast: Drex's median is 306 milliseconds per document, against 561 for the local model.

*(18/7/0.72/0.8571 = `drex.errors_found/false_alarms/precision/recall`; 16/61/0.2078 = `llm.*`;
15 and 4.1 = `drex.budget_curve["20"]` and `llm.budget_curve["20"]`, ties broken at random;
306 / 561 = `latency_p50_ms`.)*

⚠️ **Cost:** `drex.cost_per_1k_docs` is null because the Drex price isn't set yet. Leave cost out, or set
`DREX_PRICE_PER_INPUT_TOKEN` in `config.py` from your console Usage CSV, rerun `python src/score.py`,
and only then quote it.

## 6. Limits, stated openly (4:20 to 4:40)

> Three honest limits. The errors are planted, so real label noise may be subtler. Drex isn't
> perfect on its own either: it classifies 78.5% of test pages correctly, so some flags are its
> mistakes, and those are our seven false alarms. And our baseline is a small open model on a laptop,
> not a frontier LLM.

*(78.5% = `drex.accuracy_vs_true` 0.785.)*

## 7. NDI and Adobe (4:40 to 5:00)

> ⚠️ *[Confirm the wording with the organisers before saying this.]* Anywhere documents get labelled at
> scale, this slots in right after parsing. NDI turns scans into text, and Drex adds a blind second
> opinion on every label, in one call per document. For document tools like Adobe's, that means cleaner
> training data and a twenty-document review queue instead of a full re-audit. Thank you.
