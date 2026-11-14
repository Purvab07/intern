// Cuckoo: upload files, compare Drex vs the LLM on THOSE files, plain-words verdict.
// Everything shown comes from the live /api/analyze results (src/app.py). textContent only.
(function () {
  "use strict";
  var DASH = "–", CLASSES = ["invoice", "letter", "form", "resume", "report", "memo"];
  var health = { ok: false }, queue = [], results = [], sel = -1;

  // ---------- helpers ----------
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function pct(v) { return isNum(v) ? Math.round(v * 100) + "%" : DASH; }
  function t(ms) { return isNum(ms) ? (ms < 1000 ? Math.round(ms) + " ms" : (ms / 1000).toFixed(1) + " s") : DASH; }
  function $(id) { return document.getElementById(id); }
  function clear(b) { while (b.firstChild) b.removeChild(b.firstChild); return b; }
  function el(tag, a, kids) {
    var e = document.createElement(tag);
    for (var k in a || {}) {
      if (k === "text") e.textContent = a[k]; else if (k === "class") e.className = a[k];
      else if (k === "style") e.style.cssText = a[k]; else e.setAttribute(k, a[k]);
    }
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined && c !== false) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function B(x, style) { return el("b", style ? { text: x, style: style } : { text: x }); }
  function bar(f, color) { return el("div", { "class": "bar" }, [el("i", { style: "width:" + Math.max(2, Math.min(100, (f || 0) * 100)).toFixed(1) + "%;background:" + color })]); }
  function name(c) { return c === "other" ? "none of these" : (c || DASH); }
  function aName(c) { return c === "other" ? "none of the known types" : (/^[aeiou]/i.test(c || "") ? "an " : "a ") + c; }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  function llmName() { return health.frontier ? "Frontier LLM" : "LLM"; }

  // ---------- verdict in plain words (one file) ----------
  function verdict(item) {
    var r = item.r || {}, D = r.drex || {}, L = r.llm || {}, f = item.filed;
    if (r.error) return { cls: "warn", e: "⚠️", chip: "Error", title: "Couldn't analyze this file", text: r.error };
    if (r.empty_text) return { cls: "warn", e: "📭", chip: "No text", title: "No readable text found", text: "NDI couldn't read any text on the first pages, so neither model can judge it." };
    if (!D.top) return { cls: "warn", e: "⚠️", chip: "Drex error", title: "Drex couldn't answer", text: D.error || "" };
    var llmBit = L.top ? (L.top === D.top ? " The " + llmName() + " agrees." : " The " + llmName() + " thinks it's " + aName(L.top) + ".") : "";
    if (f) {
      if (D.top !== f && D.confidence >= 0.95)
        return { cls: "bad", e: "⚠️", chip: "Mislabeled?", flag: true, title: "Probably filed under the wrong label",
          text: "It's filed as " + aName(f) + ", but it reads like " + aName(D.top) + " (Drex is " + pct(D.confidence) + " sure)." + llmBit + " Worth a quick human check." };
      if (D.top !== f)
        return { cls: "warn", e: "🤔", chip: "Unsure", title: "Possibly mislabeled",
          text: "Drex leans towards " + name(D.top) + " (" + pct(D.confidence) + " sure) rather than " + f + ", but not strongly enough to flag it." + llmBit };
      return { cls: "good", e: "✅", chip: "Label OK", title: "The label looks right",
        text: "Drex agrees it's " + aName(f) + " (" + pct(D.confidence) + " sure)." + llmBit };
    }
    return { cls: "info", e: "🔎", chip: cap(name(D.top)), title: D.top === "other" ? "Doesn't match any known type" : "This looks like " + aName(D.top),
      text: "Drex is " + pct(D.confidence) + " sure." + llmBit + " Pick a “filed as” label to check a label." };
  }

  // ---------- 1. add files ----------
  function initAdd() {
    $("classes").textContent = CLASSES.join(", ");
    var drop = $("drop"), picker = $("picker");
    drop.addEventListener("click", function () { picker.click(); });
    drop.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picker.click(); } });
    picker.addEventListener("change", function () { add(picker.files); picker.value = ""; });
    ["dragenter", "dragover"].forEach(function (x) { drop.addEventListener(x, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (x) { drop.addEventListener(x, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
    drop.addEventListener("drop", function (e) { e.stopPropagation(); add(e.dataTransfer.files); });
    window.addEventListener("dragover", function (e) { e.preventDefault(); });   // a stray drop must not open the file
    window.addEventListener("drop", function (e) { e.preventDefault(); if (e.dataTransfer && e.dataTransfer.files.length) { add(e.dataTransfer.files); $("add").scrollIntoView(); } });
    Array.prototype.forEach.call(document.querySelectorAll("[data-sample]"), function (b) {
      b.addEventListener("click", function () {
        fetch("../data/raw/scans/" + b.dataset.sample + ".png").then(function (r) { if (!r.ok) throw new Error(); return r.blob(); })
          .then(function (blob) { add([new File([blob], "sample-" + b.dataset.sample + ".png", { type: "image/png" })], b.dataset.filed); })
          .catch(function () { status("Sample scans aren't available here (data/raw is local only).", true); });
      });
    });
    $("go").addEventListener("click", run);
    fetch("/api/health").then(function (r) { return r.json(); }).then(function (h) { health = h || {}; if (h.classes) CLASSES = h.classes; $("classes").textContent = CLASSES.join(", "); renderQueue(); llmNote(); })
      .catch(function () { health = { ok: false }; renderQueue(); });
  }
  function llmNote() {
    var n = clear($("llm-note"));
    n.appendChild(el("div", { "class": "note" }, health.frontier
      ? ["Comparing Drex with a frontier LLM: ", B(health.llm_model), "."]
      : ["Heads-up: the LLM here is ", B(health.llm_model || "a local model"), " running on this laptop, not a frontier model. To compare with Claude, add ",
         el("code", { text: "LLM_PROVIDER=anthropic" }), " and ", el("code", { text: "ANTHROPIC_API_KEY=…" }), " to .env and restart the app."]));
  }
  function status(msg, warn) { var s = clear($("status")); if (msg) s.appendChild(el("div", { "class": warn ? "offline" : "hint", text: msg })); }
  function add(files, filed) {
    Array.prototype.forEach.call(files || [], function (f) {
      if (!/\.(pdf|png|jpe?g|tiff?|webp)$/i.test(f.name)) { status("Skipped " + f.name + ": use PDF, PNG, JPG, TIFF or WEBP.", true); return; }
      if (f.size > 15 * 1024 * 1024) { status("Skipped " + f.name + ": larger than 15 MB.", true); return; }
      queue.push({ file: f, filed: filed || "", url: /^image\/(png|jpe?g|webp)/.test(f.type) ? URL.createObjectURL(f) : null });
    });
    renderQueue();
  }
  function renderQueue() {
    var box = clear($("files"));
    queue.forEach(function (q, i) {
      var s = el("select", { "aria-label": "Filed as" }, [el("option", { value: "", text: "Filed as… (optional)" })]
        .concat(CLASSES.map(function (c) { var o = el("option", { value: c, text: "Filed as " + c }); if (c === q.filed) o.selected = true; return o; })));
      s.addEventListener("change", function () { q.filed = s.value; });
      var x = el("button", { "class": "x", "aria-label": "Remove", text: "×" });
      x.addEventListener("click", function () { queue.splice(i, 1); renderQueue(); });
      box.appendChild(el("div", { "class": "file" }, [el("div", { "class": "fi" }, [q.url ? el("img", { src: q.url, alt: "" }) : "📄"]), el("div", { "class": "nm", text: q.file.name }), s, x]));
    });
    $("go").disabled = !queue.length || !health.ok;
    $("go").textContent = queue.length > 1 ? "Analyze " + queue.length + " files" : "Analyze";
    if (!health.ok) {
      var st = clear($("status"));
      st.appendChild(el("div", { "class": "offline" }, ["Live analysis needs the Cuckoo app. In the repo folder run ", el("code", { text: "python src/app.py" }),
        " and open ", el("code", { text: "http://localhost:8000/demo/" }), "."]));
    }
  }
  function b64(file) { return new Promise(function (res, rej) { var r = new FileReader(); r.onload = function () { res(String(r.result).split(",")[1]); }; r.onerror = rej; r.readAsDataURL(file); }); }

  function run() {
    var items = queue.map(function (q) { return { name: q.file.name, file: q.file, url: q.url, filed: q.filed, r: null }; });
    queue = []; renderQueue();
    var pg = $("progress"); if (pg.querySelector(".empty")) clear(pg);
    var chain = Promise.resolve(), done = 0;
    status("Analyzing " + items.length + " file" + (items.length > 1 ? "s" : "") + "…");
    items.forEach(function (it) {
      var row = el("div", { "class": "prow" }, [el("div", { "class": "fi" }, [it.url ? el("img", { src: it.url, alt: "" }) : "📄"]),
        el("div", null, [el("div", { "class": "nm", text: it.name }), el("div", { "class": "st" }, [el("span", { "class": "spinner" }), " waiting…"])]), el("span")]);
      pg.appendChild(row);
      chain = chain.then(function () {
        row.querySelector(".st").lastChild.textContent = " reading the page, then asking Drex and the " + llmName() + "…";
        return b64(it.file).then(function (data) {
          return fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: it.name, data: data, filed_as: it.filed }) });
        }).then(function (r) { return r.json(); }).catch(function (e) { return { error: String(e) }; })
          .then(function (r) {
            it.r = r; results.push(it); done++;
            var v = verdict(it), D = r.drex || {}, L = r.llm || {};
            var st = clear(row.querySelector(".st"));
            st.textContent = r.error || r.empty_text ? v.text : "Drex: " + name(D.top) + " · " + llmName() + ": " + name(L.top) + (it.filed ? " · filed as " + it.filed : "");
            row.replaceChild(el("span", { "class": "chip " + v.cls, text: v.chip }), row.lastChild);
            var idx = results.length - 1;
            row.addEventListener("click", function () { select(idx); $("compare").scrollIntoView(); });
            select(idx); renderResult();
            status(done < items.length ? "Analyzed " + done + " of " + items.length + "…" : "Done. Scroll down for the comparison and the verdict.");
          });
      });
    });
  }

  // ---------- 2. compare on this file ----------
  function select(i) { sel = i; renderCompare(); }
  function renderCompare() {
    var box = clear($("cmp")), it = results[sel];
    if (!it) { box.appendChild(el("div", { "class": "empty", text: "Analyze a file in step ① and the comparison appears here." })); return; }
    if (results.length > 1) {
      var tabs = el("div", { "class": "tabs" });
      results.forEach(function (x, i) { var b = el("button", { "class": "tab" + (i === sel ? " on" : ""), text: x.name }); b.addEventListener("click", function () { select(i); }); tabs.appendChild(b); });
      box.appendChild(tabs);
    }
    var r = it.r || {}, D = r.drex || {}, L = r.llm || {}, f = it.filed, LN = llmName();
    $("cmp-title").textContent = "How did each model handle “" + it.name + "”?";
    if (r.error || r.empty_text || !D.top) { box.appendChild(el("div", { "class": "empty", text: verdict(it).text })); return; }

    box.appendChild(el("div", { "class": "answers" }, [
      el("div", { "class": "ans d" }, [el("span", { "class": "who" }, [el("span", { "class": "sw d" }), "Drex says"]), el("div", { "class": "big", text: cap(name(D.top)) }),
        el("div", { "class": "sub", text: pct(D.confidence) + " probability, measured across all options" })]),
      el("div", { "class": "ans l" }, [el("span", { "class": "who" }, [el("span", { "class": "sw l" }), LN + " (" + (L.model || "") + ") says"]), el("div", { "class": "big", text: L.top ? cap(name(L.top)) : "No answer" }),
        el("div", { "class": "sub", text: L.error ? L.error : (isNum(L.confidence) ? pct(L.confidence) + " (its own guess at how sure it is)" : "") })])
    ]));

    var grid = el("div", { "class": "params" });
    function card(c, cs, ic, title, winner, lines, take) {
      var w = winner === "d" ? el("span", { "class": "win d", text: "Drex better" }) : winner === "l" ? el("span", { "class": "win l", text: LN + " better" }) : winner === "t" ? el("span", { "class": "win t", text: "Tie" }) : null;
      grid.appendChild(el("div", { "class": "card pc", style: "--c:" + c + ";--cs:" + cs }, [w, el("div", { "class": "q" }, [el("div", { "class": "ic", text: ic }), el("h3", { text: title })])]
        .concat(lines).concat([el("p", { "class": "take" }, take)])));
    }
    function line(who, frac, color, val) { return el("div", { "class": "line" }, [el("span", { "class": "who" }, [el("span", { "class": "sw " + who }), who === "d" ? "Drex" : "LLM"]), bar(frac, color), el("span", { "class": "v", text: val })]); }

    if (f) {
      var dOk = D.top === f, lOk = L.top === f;
      card("var(--violet)", "var(--violet-soft)", "🏷️", "Agrees with your label (" + f + ")?", null,
        [line("d", dOk ? 1 : 0.02, dOk ? "var(--good)" : "var(--bad)", dOk ? "Yes" : "No"), line("l", lOk ? 1 : 0.02, lOk ? "var(--good)" : "var(--bad)", L.top ? (lOk ? "Yes" : "No") : DASH)],
        [dOk === lOk ? "Both models " + (dOk ? "agree with" : "disagree with") + " the label." : "They split: Drex says " + name(D.top) + ", the " + LN + " says " + name(L.top) + "."]);
    } else {
      var same = D.top === L.top;
      card("var(--violet)", "var(--violet-soft)", "🤝", "Do the two models agree?", null,
        [line("d", 1, "var(--drex)", cap(name(D.top))), line("l", 1, "var(--llm)", cap(name(L.top)))],
        [same ? "Yes: both say " + name(D.top) + "." : "No: they disagree, so this file deserves a closer look."]);
    }
    var probs = D.probs || {}, opts = el("div", { "class": "opts" });
    Object.keys(probs).sort(function (a, b) { return probs[b] - probs[a]; }).slice(0, 4).forEach(function (c) {
      opts.appendChild(el("span", { text: cap(name(c)) })); opts.appendChild(bar(probs[c], "var(--drex)")); opts.appendChild(el("span", { style: "text-align:right", text: pct(probs[c]) }));
    });
    card("var(--drex)", "var(--drex-soft)", "🧮", "Shows its reasoning across options", "d", [opts],
      ["Drex scored all ", B(String(Object.keys(probs).length) + " options"), " at once (top 4 shown). The " + LN + " only names one answer, so you can't see what else it considered."]);
    card("var(--teal)", "var(--teal-soft)", "🎯", "How sure is it?", null,
      [line("d", D.confidence, "var(--drex)", pct(D.confidence)), line("l", L.confidence, "var(--llm)", pct(L.confidence))],
      isNum(L.confidence) && L.confidence >= 0.99 && D.confidence < 0.99
        ? ["The " + LN + " says it's certain. Drex's number is a real probability, so ", B(pct(D.confidence)), " means there is some doubt worth knowing about."]
        : ["Drex's ", B(pct(D.confidence)), " is a measured probability; the " + LN + "'s ", B(pct(L.confidence)), " is its own guess."]);
    var dT = D.latency_ms, lT = L.latency_ms, fastD = isNum(dT) && isNum(lT) && dT < lT, mx = Math.max(dT || 0, lT || 0) || 1;
    var close = isNum(dT) && isNum(lT) && Math.max(dT, lT) < 1.2 * Math.min(dT, lT);
    card("var(--llm)", "var(--llm-soft)", "⏱️", "Time to answer this file", close ? "t" : (fastD ? "d" : "l"),
      [line("d", dT / mx, "var(--drex)", t(dT)), line("l", lT / mx, "var(--llm)", t(lT))],
      close ? ["About the same on this file."] : [B(fastD ? "Drex" : "The " + LN), " answered " + (fastD ? (lT / dT) : (dT / lT)).toFixed(1) + "× faster. The timeline below shows where the time went."]);
    var dIn = D.input_tokens, lIn = L.input_tokens, lOut = L.output_tokens;
    card("var(--warn)", "var(--warn-soft)", "⚙️", "Work done to answer", isNum(lOut) && lOut > 0 ? "d" : null,
      [el("div", { "class": "line" }, [el("span", { "class": "who" }, [el("span", { "class": "sw d" }), "Drex"]), el("span", { style: "font-size:13px", text: "read " + (dIn || DASH) + " · wrote 0" }), el("span", { "class": "v", text: "0 tokens" })]),
       el("div", { "class": "line" }, [el("span", { "class": "who" }, [el("span", { "class": "sw l" }), "LLM"]), el("span", { style: "font-size:13px", text: "read " + (lIn || DASH) + " · wrote " + (lOut || DASH) }), el("span", { "class": "v", text: (lOut || DASH) + " tokens" })])],
      ["Both read the page (counted in tokens, roughly word pieces). Drex answers with numbers; the " + LN + " has to ", B("write " + (lOut || DASH) + " tokens"), " of text one by one, which takes time."]);
    var att = L.attempts || 1;
    card("var(--good)", "var(--good-soft)", "🔁", "Answered first time?", att > 1 ? "d" : "t",
      [line("d", 1, "var(--good)", "1 try"), line("l", 1, att > 1 ? "var(--bad)" : "var(--good)", att + (att > 1 ? " tries" : " try"))],
      att > 1 ? ["The " + LN + "'s first reply couldn't be read, so it had to be asked again."] : ["Both gave a usable answer on the first try."]);
    box.appendChild(grid);

    var tl = el("div", { "class": "card timeline" }, [el("h3", { text: "Where the time went, step by step" }),
      el("p", { style: "margin:0;color:var(--ink-2);font-size:14px" }, ["First, both waited ", B(t(r.ndi_ms)), " for NDI to read the page (" + (r.pages_read || 1) + " page" + ((r.pages_read || 1) > 1 ? "s" : "") + "). Then each model worked on the text, at the same time:"])]);
    var P = L.phases || {}, segsL;
    var net = isNum(D.server_ms) ? Math.max(0, dT - D.server_ms) : null;
    var segsD = isNum(D.server_ms) ? [["Internet round trip", net, "#93c5fd"], ["Thinking (one pass)", D.server_ms, "#2563eb"]] : [["Answering", dT, "#2563eb"]];
    if (isNum(P.wait_first_word_ms)) segsL = [["Waiting for first word", P.wait_first_word_ms, "#fdba74"], ["Writing the answer", P.write_ms, "#f97316"]];
    else segsL = [["Loading the model", P.load_ms, "#fde68a"], ["Reading the text", P.read_ms, "#fdba74"], ["Writing the answer", P.write_ms, "#f97316"], ["Other", P.other_ms, "#fed7aa"]];
    var scale = Math.max(dT || 0, lT || 0) || 1;
    function row(label, segs, total) {
      var tr = el("div", { "class": "track" });
      segs.forEach(function (s) { if (isNum(s[1]) && s[1] > 0) tr.appendChild(el("div", { "class": "seg", title: s[0] + ": " + t(s[1]), style: "width:" + (s[1] / scale * 100).toFixed(2) + "%;background:" + s[2] + ";color:" + (s[2] === "#2563eb" || s[2] === "#f97316" ? "#fff" : "#7c2d12") }, [s[1] / scale > 0.14 ? t(s[1]) : ""])); });
      return el("div", { "class": "tl-row" }, [el("span", { "class": "lab", text: label }), tr, el("span", { "class": "tot", text: t(total) })]);
    }
    tl.appendChild(row("Drex", segsD, dT));
    tl.appendChild(row(LN, segsL, lT));
    var lg = el("div", { "class": "legend" });
    segsD.concat(segsL).forEach(function (s) { if (isNum(s[1]) && s[1] > 0) lg.appendChild(el("span", null, [el("i", { style: "background:" + s[2] }), s[0] + " " + t(s[1])])); });
    tl.appendChild(lg);

    var ev = el("ul", { "class": "evidence" });
    function fact(e, bg, kids) { ev.appendChild(el("li", null, [el("span", { "class": "e", style: "background:" + bg, text: e }), el("div", null, kids)])); }
    if (isNum(P.load_ms) && P.load_ms > 50) fact("🧊", "#fef3c7", ["Before it could start, the " + LN + " spent ", B(t(P.load_ms)), " loading itself into memory. Drex had nothing to load."]);
    if (isNum(P.write_ms) && P.write_ms > 0) fact("✍️", "var(--llm-soft)", ["The " + LN + " wrote its answer piece by piece: ", B((lOut || "?") + " tokens in " + t(P.write_ms)), ". Drex doesn't write: it scored every option in one pass" + (isNum(D.server_ms) ? " in " : "."), isNum(D.server_ms) ? B(t(D.server_ms)) : null, isNum(D.server_ms) ? "." : null]);
    if (isNum(P.read_ms) && P.read_ms > 0) fact("📖", "#ffedd5", ["Reading the text took the " + LN + " ", B(t(P.read_ms)), " for " + (lIn || "?") + " tokens."]);
    if (isNum(P.wait_first_word_ms)) fact("⌛", "#ffedd5", ["The " + LN + " took ", B(t(P.wait_first_word_ms)), " before its first word arrived."]);
    if (isNum(net) && net > 0) fact("🌐", "var(--drex-soft)", ["Most of Drex's time was the trip over the internet (", B(t(net)), "); the model itself needed ", B(t(D.server_ms)), "."]);
    if (isNum(dT) && isNum(lT)) {
      if (close) fact("🤝", "var(--soft)", ["On this file both finished in about the same time."]);
      else if (fastD) fact("🏁", "var(--good-soft)", ["Overall, ", B("Drex finished " + t(lT - dT) + " sooner"), " (" + t(dT) + " vs " + t(lT) + ")."]);
      else fact("🏁", "var(--soft)", ["On this file the " + LN + " finished first (" + t(lT) + " vs " + t(dT) + ")" + (isNum(net) && net > lT ? ", mainly because Drex's answer had to travel over the internet while the " + LN + " runs on this laptop." : ".")]);
    }
    tl.appendChild(ev);
    box.appendChild(tl);
  }

  // ---------- 3. result ----------
  function count(list, key) { var m = {}; list.forEach(function (x) { var k = key(x); if (k) m[k] = (m[k] || 0) + 1; }); return m; }
  function describe(m) {
    var ks = Object.keys(m).sort(function (a, b) { return m[b] - m[a]; });
    return ks.length ? ks.map(function (k) { return m[k] + " " + (k === "other" ? "“none of these”" : k + (m[k] > 1 ? "s" : "")); }).join(", ") : DASH;
  }
  function renderResult() {
    var box = clear($("res")), ok = results.filter(function (x) { return x.r && x.r.drex && x.r.drex.top; });
    if (!results.length) { box.appendChild(el("div", { "class": "empty", text: "Your verdict appears here after the analysis." })); return; }
    var LN = llmName();
    if (results.length === 1) {
      var v = verdict(results[0]);
      box.appendChild(el("div", { "class": "verdict " + v.cls }, [el("span", { "class": "e", text: v.e }), el("div", null, [el("h3", { text: v.title }), el("p", { text: v.text })])]));
    } else {
      var flagged = results.filter(function (x) { return verdict(x).flag; });
      var agree = ok.filter(function (x) { return x.r.llm && x.r.llm.top === x.r.drex.top; }).length;
      var dSum = ok.reduce(function (s, x) { return s + (x.r.drex.latency_ms || 0); }, 0), lSum = ok.reduce(function (s, x) { return s + ((x.r.llm || {}).latency_ms || 0); }, 0);
      box.appendChild(el("div", { "class": "verdict " + (flagged.length ? "bad" : "good") }, [el("span", { "class": "e", text: flagged.length ? "⚠️" : "✅" }), el("div", null, [
        el("h3", { text: flagged.length ? flagged.length + " of your " + results.length + " files look mislabeled" : "No mislabeled files found in your " + results.length + " files" }),
        el("p", { text: (flagged.length ? "Check: " + flagged.map(function (x) { return x.name + " (filed as " + x.filed + ", looks like " + aName(x.r.drex.top) + ")"; }).join("; ") + ". " : "") +
          "Drex and the " + LN + " agree on " + agree + " of " + ok.length + " files. Total answer time: Drex " + t(dSum) + ", " + LN + " " + t(lSum) + "." })])]));
    }
    var tb = el("tbody");
    results.forEach(function (x, i) {
      var r = x.r || {}, v = verdict(x);
      var tr = el("tr", { style: "cursor:pointer" }, [el("td", { "class": "nm", text: x.name }), el("td", { "class": "hide-sm", text: x.filed ? cap(x.filed) : DASH }),
        el("td", null, [el("span", { "class": "lbl d", text: r.drex && r.drex.top ? cap(name(r.drex.top)) : DASH }), r.drex && isNum(r.drex.confidence) ? el("span", { style: "color:var(--ink-3);font-size:12.5px", text: " " + pct(r.drex.confidence) }) : null]),
        el("td", null, [el("span", { "class": "lbl l", text: r.llm && r.llm.top ? cap(name(r.llm.top)) : DASH })]),
        el("td", null, [el("span", { "class": "chip " + v.cls, text: v.chip })])]);
      tr.addEventListener("click", function () { select(i); $("compare").scrollIntoView(); });
      tb.appendChild(tr);
    });
    box.appendChild(el("div", { "class": "card", style: "padding:10px 14px;overflow-x:auto" }, [el("table", { "class": "rtable" }, [
      el("thead", null, [el("tr", null, [el("th", { text: "File" }), el("th", { "class": "hide-sm", text: "Filed as" }), el("th", { text: "Drex says" }), el("th", { text: LN + " says" }), el("th", { text: "Verdict" })])]), tb])]));
    if (ok.length > 1) {
      var dC = count(ok, function (x) { return x.r.drex.top; }), lC = count(ok, function (x) { return (x.r.llm || {}).top; }), fC = count(ok, function (x) { return x.filed; });
      var cats = Object.keys(Object.assign({}, dC, lC, fC)), mx = 1;
      cats.forEach(function (c) { mx = Math.max(mx, dC[c] || 0, lC[c] || 0, fC[c] || 0); });
      var mix = el("div", { "class": "mix" });
      cats.forEach(function (c) {
        function r(lab, v, col) { return el("div", { "class": "r" }, [el("span", { text: lab }), bar((v || 0) / mx, col), B(String(v || 0))]); }
        mix.appendChild(el("div", { "class": "m" }, [el("h4", { text: cap(name(c)) }), Object.keys(fC).length ? r("Filed", fC[c], "#94a3b8") : null, r("Drex", dC[c], "var(--drex)"), r("LLM", lC[c], "var(--llm)")]));
      });
      box.appendChild(el("div", { "class": "card", style: "margin-top:14px" }, [el("h3", { style: "margin:0 0 4px;font-size:18px", text: "What's in your dataset, according to each model" }),
        el("p", { style: "margin:0;color:var(--ink-2);font-size:14.5px" }, [B("Drex: ", "color:var(--drex)"), describe(dC) + ". ", B(LN + ": ", "color:#c2410c"), describe(lC) + "."]), mix]));
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initAdd); else initAdd();
})();
