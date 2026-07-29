/* ============================================================
   Interactive explainer — "Length Is Not Information"
   Companion to "Text Prompt Scaling Law in Visual Generation".
   Vanilla JS, no deps; reuses window.SVGH drawing primitives.
   Each section registers a mount via reg(id, fn); init() binds
   every registered widget whose host element is present.
   ============================================================ */
(function () {
  "use strict";
  const H = window.SVGH;
  if (!H) { console.error("svg-helpers.js (window.SVGH) must load before blog.js"); return; }

  // ---- widget registry: id -> mount(rootEl) ----
  const MOUNTS = {};
  function reg(id, fn) { MOUNTS[id] = fn; }

  // ---- reading-progress bar ----
  function progressBar() {
    const bar = document.getElementById("progress");
    if (!bar) return;
    function update() {
      const doc = document.documentElement;
      const max = doc.scrollHeight - doc.clientHeight;
      const p = max > 0 ? window.scrollY / max : 0;
      bar.style.transform = "scaleX(" + H.clamp(p, 0, 1) + ")";
    }
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  // ====================================================================
  //  Section widgets are appended below (one block per section).
  // ====================================================================

  // §2 — Length is not information.
  // Real data: natural-language ("dense") captions averaged over 300 prompts
  // (data/infomation_metrics/gpg_data_points.json). Words climb steeply with
  // detail level; Grounded Perplexity Gain barely moves.
  const NL_LADDER = [
    { lvl: "L6",  tok: 786,  gpg: 106.3 },
    { lvl: "L8",  tok: 956,  gpg: 110.1 },
    { lvl: "L10", tok: 1249, gpg: 112.5 },
  ];
  function nlLength(root) {
    const host = root.querySelector(".nll-bars");
    const slider = root.querySelector(".nll-slider");
    const verdict = root.querySelector(".nll-verdict");
    const base = NL_LADDER[0];
    const GMAX = 0.75; // axis spans 0..+75% growth over L6

    const W = 560, HT = 168, x0 = 150, x1 = 524, t = 14;
    const yW = 58, yI = 116, bh = 30; // bar y-centres + height
    const X = pct => x0 + (H.clamp(pct, 0, GMAX) / GMAX) * (x1 - x0);
    const svg = H.makeSVG(W, HT);

    // axis + gridlines at 0/25/50/75%
    [0, 0.25, 0.5, 0.75].forEach(p => {
      svg.appendChild(H.el("line", { x1: X(p), y1: t, x2: X(p), y2: HT - 24, stroke: H.GRID, "stroke-width": 0.6, opacity: 0.7 }));
      svg.appendChild(H.txt(X(p), HT - 9, "+" + Math.round(p * 100) + "%", { "text-anchor": "middle", class: "ax-lab" }));
    });
    svg.appendChild(H.txt(x0, t - 4, "more words / more information than the shortest caption →",
      { "text-anchor": "start", class: "ax-lab", "font-style": "italic" }));

    function row(yc, label, color) {
      svg.appendChild(H.txt(x0 - 12, yc + 5, label, { "text-anchor": "end", class: "ax-lab", "font-size": 12.5, fill: H.INK }));
      svg.appendChild(H.el("line", { x1: x0, y1: t, x2: x0, y2: HT - 24, stroke: H.AXIS, "stroke-width": 1 }));
      const track = H.rrect(x0, yc - bh / 2, 2, bh, { fill: color, opacity: 0.18 });
      const bar = H.rrect(x0, yc - bh / 2, 0, bh, { fill: color });
      const val = H.txt(x0 + 8, yc + 5, "", { "text-anchor": "start", "font-family": H.MONO, "font-size": 12, fill: "#fff", "font-weight": 500 });
      svg.appendChild(track); svg.appendChild(bar); svg.appendChild(val);
      return { bar, val };
    }
    const rW = row(yW, "WORDS", H.RED);
    const rI = row(yI, "INFORMATION", H.BLUE);
    host.innerHTML = ""; host.appendChild(svg);

    let cur = { w: 0, i: 0 };
    function paint(wPct, iPct, d) {
      [[rW, wPct, d.tok, "tokens"], [rI, iPct, d.gpg, "nats"]].forEach(([r, pct, abs, unit]) => {
        const w = X(pct) - x0;
        r.bar.setAttribute("width", Math.max(0, w));
        // keep value label readable: inside the bar if long enough, else just past its end
        const inside = w > 96;
        r.val.setAttribute("x", inside ? x0 + 8 : x0 + w + 8);
        r.val.setAttribute("fill", inside ? "#fff" : H.MUTED);
        r.val.textContent = `${abs} ${unit}  ·  +${Math.round(pct * 100)}%`;
      });
    }
    function setLevel(idx, animate) {
      const d = NL_LADDER[idx];
      const wTarget = d.tok / base.tok - 1, iTarget = d.gpg / base.gpg - 1;
      const w0 = cur.w, i0 = cur.i;
      const apply = (u) => paint(w0 + (wTarget - w0) * u, i0 + (iTarget - i0) * u, d);
      if (animate) H.tween(420, apply, () => { cur = { w: wTarget, i: iTarget }; });
      else { paint(wTarget, iTarget, d); cur = { w: wTarget, i: iTarget }; }
      verdict.innerHTML = idx === 0
        ? `<b>L6</b> — the shortest, most compact caption. Drag right to pile on detail.`
        : `From L6 to <b>${d.lvl}</b>: <span class="up-w">+${Math.round(wTarget * 100)}% words</span>, but only <span class="up-i">+${Math.round(iTarget * 100)}% information</span>.`;
    }
    slider.addEventListener("input", () => setLevel(+slider.value, true));
    setLevel(0, false);
  }
  reg("nl-length", nlLength);

  // §3 — GPG, the white-box meter. Per-token explorer.
  // ILLUSTRATIVE per-token surprisals for one caption (representative values,
  // chosen to make the mechanism legible — NOT measured). Real GPG is computed
  // by a frozen judge and aggregated over 300 prompts (see §5). Visibly marked
  // "illustrative data" in the figure caption, per project rule.
  // [token, surprisal WITH image (nats), grounding gain (nats)]
  const GPG_TOKENS = [
    ["A", 0.3, 0], ["silver", 0.5, 2.6], ["pickup", 0.6, 2.1], ["truck", 0.4, 2.9],
    ["transports", 1.1, 0.7], ["cardboard", 0.5, 3.1], ["boxes", 0.6, 2.8], ["a", 0.3, 0],
    ["wrapped", 0.9, 1.4], ["cylinder", 1.0, 1.2], ["and", 0.2, 0], ["Coca-Cola", 1.4, 0.4],
    ["cans", 0.8, 0.7], ["on", 0.2, 0], ["a", 0.3, 0], ["street", 0.7, 1.7],
  ];
  function gpgPerToken(root) {
    const toks = GPG_TOKENS;
    const gpgTotal = toks.reduce((a, t) => a + t[2], 0);
    const chart = root.querySelector(".gpgx-chart");
    const head = root.querySelector(".gpgx-head");
    const btn = root.querySelector(".gpgx-toggle");
    const imgwrap = root.querySelector(".gpgx-imgwrap");
    const AMBER = "#d8973c";

    const W = 680, HT = 300, m = { l: 38, r: 14, t: 22, b: 86 };
    const n = toks.length, yMax = 4.0, bw = (W - m.l - m.r) / n * 0.62;
    const Y = v => m.t + (1 - v / yMax) * (HT - m.t - m.b);
    const base = Y(0);
    const svg = H.makeSVG(W, HT);

    [0, 1, 2, 3, 4].forEach(v => {
      svg.appendChild(H.el("line", { x1: m.l, y1: Y(v), x2: W - m.r, y2: Y(v), stroke: H.GRID, "stroke-width": 0.6, opacity: 0.5 }));
      svg.appendChild(H.txt(m.l - 7, Y(v) + 4, v, { "text-anchor": "end", class: "ax-lab" }));
    });

    const bars = toks.map((d, i) => {
      const [tok, nll, gain] = d;
      const cx = m.l + (i + 0.5) / n * (W - m.l - m.r);
      const baseRect = H.el("rect", { x: cx - bw / 2, width: bw, fill: H.BLUE, y: Y(nll), height: Math.max(0, base - Y(nll)) });
      const gainRect = H.el("rect", { x: cx - bw / 2, width: bw, fill: AMBER, y: Y(nll), height: 0 });
      const tl = H.txt(cx + 3, base + 13, tok, { "text-anchor": "end", class: "ax-lab" });
      tl.setAttribute("transform", `rotate(-42 ${cx + 3} ${base + 13})`);
      svg.appendChild(baseRect); svg.appendChild(gainRect); svg.appendChild(tl);
      // transparent hit column for hover
      const hit = H.el("rect", { x: cx - bw / 2 - 3, y: m.t, width: bw + 6, height: base - m.t, fill: "#fff", opacity: 0 });
      hit.style.cursor = "default";
      hit.addEventListener("mouseenter", e => H.showTip(
        `<b>${tok}</b><br>surprisal w/ image: ${nll.toFixed(1)}<br>surprisal w/o image: ${(nll + gain).toFixed(1)}<br>grounding gain: <b>${gain.toFixed(1)}</b> nats`, e));
      hit.addEventListener("mousemove", H.moveTip);
      hit.addEventListener("mouseleave", H.hideTip);
      svg.appendChild(hit);
      return { nll, gain, baseRect, gainRect };
    });

    // legend + axis title
    svg.appendChild(H.el("rect", { x: m.l, y: 3, width: 10, height: 10, fill: H.BLUE }));
    svg.appendChild(H.txt(m.l + 15, 12, "surprisal given the image", { class: "ax-lab" }));
    svg.appendChild(H.el("rect", { x: m.l + 190, y: 3, width: 10, height: 10, fill: AMBER }));
    svg.appendChild(H.txt(m.l + 205, 12, "explained by the image  (grounding gain)", { class: "ax-lab" }));
    const yt = H.txt(11, m.t + (HT - m.t - m.b) / 2, "per-token surprisal  (nats)", { "text-anchor": "middle", class: "ax-title" });
    yt.setAttribute("transform", `rotate(-90 11 ${m.t + (HT - m.t - m.b) / 2})`);
    svg.appendChild(yt);
    chart.innerHTML = ""; chart.appendChild(svg);

    function paint(t) { // t: 1 = image on (amber collapsed) … 0 = image off (amber full)
      bars.forEach(b => {
        const top = Y(b.nll + b.gain * (1 - t));
        b.gainRect.setAttribute("y", top);
        b.gainRect.setAttribute("height", Math.max(0, Y(b.nll) - top));
      });
      head.innerHTML = `GPG<sub>total</sub> = &Sigma; grounding gain = <b>${gpgTotal.toFixed(1)}</b> nats`;
    }
    let on = true;
    function animateTo(target) {
      const from = on ? 1 : 0;
      H.tween(520, u => paint(from + (target - from) * u));
      on = target === 1;
      btn.innerHTML = on ? "image <b>on</b> &middot; hide it" : "image <b>off</b> &middot; reveal it";
      btn.classList.toggle("off", !on);
      if (imgwrap) imgwrap.classList.toggle("hidden", !on);
    }
    btn.addEventListener("click", () => animateTo(on ? 0 : 1));
    paint(1);
    btn.innerHTML = "image <b>on</b> &middot; hide it";
  }
  reg("gpg-explore", gpgPerToken);

  // §1 — renderer demo: one slide, prompts of increasing length, two models.
  // gpt-image-2 expands internally (output ~flat); qwen-image renders the prompt
  // (output rises then saturates). Builds a 2×N image grid from a manifest.
  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function rendererDemo(root) {
    const grid = root.querySelector(".rdemo-grid");
    const promptBox = root.querySelector(".rdemo-prompt");
    const bigGpt = root.querySelector('.rd-big[data-model="gpt"]');
    const bigQwen = root.querySelector('.rd-big[data-model="qwen"]');
    const cell = (text, cls) => { const d = document.createElement("div"); d.className = cls; if (text) d.textContent = text; return d; };

    fetch("assets/renderer_demo/manifest.json").then(r => r.json()).then(mf => {
      const ps = mf.prompts;
      grid.style.gridTemplateColumns = `96px repeat(${ps.length}, 1fr)`;

      // header row: corner + per-level word counts
      grid.appendChild(cell("", "rd-corner"));
      ps.forEach(p => { const h = cell(p.words + " words", "rd-head"); h.dataset.lvl = p.level; grid.appendChild(h); });

      // two model rows
      [["gpt-image-2", "gpt"], ["qwen-image", "qwen"]].forEach(([label, key]) => {
        grid.appendChild(cell(label, "rd-rowlab"));
        ps.forEach(p => {
          const c = document.createElement("div"); c.className = "rd-cell"; c.dataset.lvl = p.level;
          const im = document.createElement("img"); im.loading = "lazy";
          im.src = `assets/renderer_demo/${key}_L${p.level}.jpg`;
          im.alt = `${label}, prompt of ${p.words} words`;
          c.appendChild(im); grid.appendChild(c);
        });
      });

      function setActive(lvl) {
        grid.querySelectorAll("[data-lvl]").forEach(e => e.classList.toggle("active", +e.dataset.lvl === lvl));
        const p = ps.find(x => x.level === lvl);
        if (bigGpt) bigGpt.src = `assets/renderer_demo/gpt_L${lvl}.jpg`;
        if (bigQwen) bigQwen.src = `assets/renderer_demo/qwen_L${lvl}.jpg`;
        promptBox.innerHTML = `<span class="rd-plab">Prompt &middot; ${p.words} words</span> ${escapeHtml(p.prompt)}`;
      }
      grid.querySelectorAll("[data-lvl]").forEach(e => e.addEventListener("click", () => setActive(+e.dataset.lvl)));
      setActive(1);
    }).catch(() => { promptBox.textContent = "(demo images not generated yet)"; });
  }
  reg("renderer-demo", rendererDemo);

  // §4 — ED, the black-box meter. Tuple matcher on the same truck image.
  // ILLUSTRATIVE attribute tuples (representative, not measured) — badged in the
  // figure caption. [tuple, matched/covered?, bbox (normalised 0-1) | null]
  const ED_COLOR = { ok: "#2f6e3a", miss: "#7d7565", no: "#9a3b33" };
  const ED_TUPLES = {
    caption: [
      ["truck — silver", true, [0.02, 0.05, 1.0, 0.93]],
      ["truck — pickup", true, [0.55, 0.04, 1.0, 0.37]],
      ["boxes — cardboard", true, [0.0, 0.27, 0.86, 0.62]],
      ["cylinder — wrapped", true, [0.0, 0.30, 0.88, 0.58]],
      ["cans — Coca-Cola", true, [0.49, 0.23, 0.63, 0.35]],
      ["scene — street", true, [0.0, 0.82, 1.0, 1.0]],
      ["traffic cone — orange", false, null],
    ],
    source: [
      ["truck — silver", true, [0.02, 0.05, 1.0, 0.93]],
      ["truck — pickup", true, [0.55, 0.04, 1.0, 0.37]],
      ["boxes — cardboard", true, [0.0, 0.27, 0.86, 0.62]],
      ["cylinder — wrapped", true, [0.0, 0.30, 0.88, 0.58]],
      ["cans — Coca-Cola", true, [0.49, 0.23, 0.63, 0.35]],
      ["scene — street", true, [0.0, 0.82, 1.0, 1.0]],
      ["boxes — green labels", false, [0.0, 0.31, 0.42, 0.53]],
      ["boxes — long, stacked", false, [0.0, 0.30, 0.88, 0.58]],
      ["boxes — several", false, [0.0, 0.27, 0.86, 0.62]],
      ["scene — daytime", false, [0.0, 0.0, 1.0, 0.20]],
      ["background — palm trees", false, [0.0, 0.0, 0.66, 0.22]],
      ["road — manhole cover", false, [0.60, 0.80, 0.96, 0.99]],
    ],
  };
  function edMatch(root) {
    const capBox = root.querySelector(".edm-caption"), srcBox = root.querySelector(".edm-source");
    const btn = root.querySelector(".edm-reveal");
    const overlay = root.querySelector(".edm-overlay"), flag = root.querySelector(".edm-halluc");
    const pf = root.querySelector(".edm-fill.p"), rf = root.querySelector(".edm-fill.r");
    const pv = root.querySelector(".pv"), rv = root.querySelector(".rv"), edv = root.querySelector(".edv");
    let revealed = false;

    function showBox(bbox, state) {
      if (!bbox) { flag.classList.add("on"); overlay.innerHTML = ""; return; }
      flag.classList.remove("on");
      const [x1, y1, x2, y2] = bbox, col = ED_COLOR[state] || "#2166ac";
      overlay.innerHTML = `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" fill="${col}" fill-opacity="0.16" stroke="${col}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    }
    function clearBox() { overlay.innerHTML = ""; flag.classList.remove("on"); }

    function mkChip(box, d, stateOf) {
      const [t, m, bbox] = d;
      const c = document.createElement("span");
      c.className = "etup neutral"; c.dataset.state = stateOf(m); c.dataset.label = t; c.textContent = t;
      c.addEventListener("mouseenter", () => showBox(bbox, c.dataset.state));
      c.addEventListener("mouseleave", clearBox);
      box.appendChild(c);
      return m;
    }
    let ok = 0, cov = 0;
    ED_TUPLES.caption.forEach(d => { if (mkChip(capBox, d, m => m ? "ok" : "no")) ok++; });
    ED_TUPLES.source.forEach(d => { if (mkChip(srcBox, d, m => m ? "ok" : "miss")) cov++; });
    const P = ok / ED_TUPLES.caption.length, R = cov / ED_TUPLES.source.length, ED = H.f05(P, R);

    function apply() {
      [...capBox.children, ...srcBox.children].forEach(c => {
        c.classList.toggle("neutral", !revealed);
        c.classList.remove("ok", "no", "miss");
        if (revealed) {
          c.classList.add(c.dataset.state);
          c.textContent = (c.dataset.state === "ok" ? "✓ " : c.dataset.state === "no" ? "✗ " : "○ ") + c.dataset.label;
        } else {
          c.textContent = c.dataset.label;
        }
      });
      pf.style.width = (revealed ? P * 100 : 0) + "%";
      rf.style.width = (revealed ? R * 100 : 0) + "%";
      pv.innerHTML = revealed ? `${P.toFixed(2)} <span class="frac">${ok}/${ED_TUPLES.caption.length}</span>` : "—";
      rv.innerHTML = revealed ? `${R.toFixed(2)} <span class="frac">${cov}/${ED_TUPLES.source.length}</span>` : "—";
      edv.textContent = revealed ? ED.toFixed(2) : "—";
      btn.textContent = revealed ? "reset" : "score it →";
      btn.classList.toggle("on", revealed);
    }
    btn.addEventListener("click", () => { revealed = !revealed; apply(); });
    apply();
  }
  reg("ed-match", edMatch);

  // §5 — the text prompt scaling law. Real 15-cell data (gpg_data_points.json):
  // [kind, label, GPG, ED, MSE]. MSE linear in GPG (r=-0.98), power-law in ED (r=-0.97).
  const CELLS = [
    ["dense", "L6", 106.3, 0.759, 0.44523], ["dense", "L8", 110.1, 0.751, 0.44542],
    ["struct", "L5", 111.6, 0.749, 0.44664], ["dense", "L10", 112.5, 0.754, 0.44536],
    ["struct", "L6", 128.3, 0.759, 0.44384], ["struct", "L7", 141.5, 0.772, 0.44275],
    ["spatial", "coarse", 151.0, 0.778, 0.44293], ["spatial", "fine", 152.2, 0.785, 0.44205],
    ["spatial", "finer", 154.4, 0.787, 0.44093], ["struct", "L8", 164.8, 0.793, 0.44074],
    ["abl", "−scene", 168.5, 0.799, 0.44052], ["struct", "L9", 191.8, 0.819, 0.43843],
    ["abl", "−bbox", 204.7, 0.807, 0.43843], ["abl", "−rel", 207.2, 0.808, 0.43706],
    ["struct", "L10", 210.5, 0.833, 0.43699],
  ];
  const SL_FAM = {
    dense:   { color: H.RED,    marker: "square",   name: "Natural language" },
    struct:  { color: H.BLUE,   marker: "circle",   name: "Structured" },
    spatial: { color: H.GREEN,  marker: "triangle", name: "Spatial variants" },
    abl:     { color: H.PURPLE, marker: "diamond",  name: "Field ablations" },
  };
  // Ordered detail series (real data). NL = dense natural language (flat loss),
  // SP = structured (loss falls). Each point: {tok, gpg, ed, mse}.
  const SL_NL = [
    { tok: 786, gpg: 106.3, ed: 0.759, mse: 0.44523 },
    { tok: 956, gpg: 110.1, ed: 0.751, mse: 0.44542 },
    { tok: 1249, gpg: 112.5, ed: 0.754, mse: 0.44536 },
  ];
  const SL_SP = [
    { tok: 294, gpg: 111.6, ed: 0.749, mse: 0.44664 },
    { tok: 369, gpg: 128.3, ed: 0.759, mse: 0.44384 },
    { tok: 439, gpg: 141.5, ed: 0.772, mse: 0.44275 },
    { tok: 566, gpg: 164.8, ed: 0.793, mse: 0.44074 },
    { tok: 780, gpg: 191.8, ed: 0.819, mse: 0.43843 },
    { tok: 957, gpg: 210.5, ed: 0.833, mse: 0.43699 },
  ];
  function slInterp(s, t) {
    const pos = t * (s.length - 1), i = Math.min(s.length - 2, Math.floor(pos)), f = pos - i;
    const a = s[i], b = s[i + 1], k = key => a[key] + (b[key] - a[key]) * f;
    return { tok: k("tok"), gpg: k("gpg"), ed: k("ed"), mse: k("mse") };
  }
  function scalingLaw(root) {
    const lenHost = root.querySelector(".slaw-len");
    const infoHost = root.querySelector(".slaw-info");
    const slider = root.querySelector(".slaw-slider");
    const readout = root.querySelector(".slaw-readout");
    const btns = Array.from(root.querySelectorAll(".slaw-rb"));

    // GPG linear least-squares fit (for the info panel)
    const g = CELLS.map(c => c[2]), y = CELLS.map(c => c[4]), n = CELLS.length;
    const mg = g.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { sxx += (g[i] - mg) ** 2; sxy += (g[i] - mg) * (y[i] - my); }
    const slope = sxy / sxx, intc = my - slope * mg;
    const INFO = {
      gpg: { lo: 100, hi: 222, key: "gpg", ckey: 2, fit: x => intc + slope * x, curve: false,
             eqn: `MSE = ${intc.toFixed(4)} − ${(Math.abs(slope) * 1e5).toFixed(2)}×10⁻⁵·GPG`, r: "−0.98",
             ticks: [100, 140, 180, 220], lab: "information  GPG (nats)" },
      ed: { lo: 0.74, hi: 0.84, key: "ed", ckey: 3, fit: x => 0.4206 * Math.pow(x, -0.2016), curve: true,
            eqn: "MSE = 0.421·ED^(−0.202)", r: "−0.97", ticks: [0.74, 0.78, 0.82], lab: "information  ED" },
    };
    const W = 470, HT = 380, m = { l: 62, r: 14, t: 16, b: 50 }, yLo = 0.4358, yHi = 0.4470;
    let mode = "gpg", lenC = null, infoC = null;
    const Ymse = v => m.t + (yHi - v) / (yHi - yLo) * (HT - m.t - m.b);

    function frame(svg) {
      [0.436, 0.438, 0.440, 0.442, 0.444, 0.446].forEach(v => {
        svg.appendChild(H.el("line", { x1: m.l, y1: Ymse(v), x2: W - m.r, y2: Ymse(v), stroke: H.GRID, "stroke-width": 0.6, opacity: 0.5 }));
        svg.appendChild(H.txt(m.l - 8, Ymse(v) + 4, v.toFixed(3), { "text-anchor": "end", class: "ax-lab" }));
      });
      svg.appendChild(H.el("line", { x1: m.l, y1: HT - m.b, x2: W - m.r, y2: HT - m.b, stroke: H.AXIS, "stroke-width": 1 }));
      svg.appendChild(H.el("line", { x1: m.l, y1: m.t, x2: m.l, y2: HT - m.b, stroke: H.AXIS, "stroke-width": 1 }));
    }
    function yTitle(svg) {
      const yt = H.txt(18, m.t + (HT - m.t - m.b) / 2, "converged MSE", { "text-anchor": "middle", class: "ax-title" });
      yt.setAttribute("transform", `rotate(-90 18 ${m.t + (HT - m.t - m.b) / 2})`); svg.appendChild(yt);
    }
    function mkMarker(svg, col, label) {
      const ring = H.el("circle", { r: 8, fill: "none", stroke: col, "stroke-width": 2.5 });
      const dot = H.el("circle", { r: 2.5, fill: col });
      const lab = H.txt(0, 0, label, { class: "end-lab", "font-size": 12, fill: col });
      svg.appendChild(ring); svg.appendChild(dot); svg.appendChild(lab);
      return { ring, dot, lab };
    }
    function place(mk, x, y) {
      mk.ring.setAttribute("cx", x); mk.ring.setAttribute("cy", y);
      mk.dot.setAttribute("cx", x); mk.dot.setAttribute("cy", y);
      mk.lab.setAttribute("x", x + 11); mk.lab.setAttribute("y", y + 4);
    }

    function buildLen() {
      const tokLo = 250, tokHi = 1300, X = v => m.l + (v - tokLo) / (tokHi - tokLo) * (W - m.l - m.r);
      const svg = H.makeSVG(W, HT); frame(svg);
      [400, 600, 800, 1000, 1200].forEach(t => {
        svg.appendChild(H.el("line", { x1: X(t), y1: HT - m.b, x2: X(t), y2: HT - m.b + 5, stroke: H.AXIS }));
        svg.appendChild(H.txt(X(t), HT - m.b + 18, t, { "text-anchor": "middle", class: "ax-lab" }));
      });
      const series = (s, col) => {
        let d = ""; s.forEach((p, i) => d += (i ? "L" : "M") + X(p.tok) + "," + Ymse(p.mse));
        svg.appendChild(H.el("path", { d, fill: "none", stroke: col, "stroke-width": 1.8, opacity: 0.45 }));
        s.forEach(p => svg.appendChild(H.marker("circle", X(p.tok), Ymse(p.mse), 3, col)));
      };
      series(SL_NL, H.RED); series(SL_SP, H.BLUE);
      svg.appendChild(H.txt(m.l + (W - m.l - m.r) / 2, HT - 8, "caption length (tokens)", { "text-anchor": "middle", class: "ax-title" }));
      yTitle(svg);
      const nlM = mkMarker(svg, H.RED, "NL"), spM = mkMarker(svg, H.BLUE, "SP");
      lenHost.innerHTML = ""; lenHost.appendChild(svg); lenC = { X, nlM, spM };
    }

    function buildInfo() {
      const M = INFO[mode], X = v => m.l + (v - M.lo) / (M.hi - M.lo) * (W - m.l - m.r);
      const svg = H.makeSVG(W, HT); frame(svg);
      M.ticks.forEach(t => {
        svg.appendChild(H.el("line", { x1: X(t), y1: HT - m.b, x2: X(t), y2: HT - m.b + 5, stroke: H.AXIS }));
        svg.appendChild(H.txt(X(t), HT - m.b + 18, mode === "gpg" ? t : t.toFixed(2), { "text-anchor": "middle", class: "ax-lab" }));
      });
      if (M.curve) {
        let d = ""; for (let i = 0; i <= 40; i++) { const x = M.lo + (M.hi - M.lo) * i / 40; d += (i ? "L" : "M") + X(x) + "," + Ymse(M.fit(x)); }
        svg.appendChild(H.el("path", { d, fill: "none", stroke: H.NAVY, "stroke-width": 1.6 }));
      } else {
        svg.appendChild(H.el("line", { x1: X(M.lo), y1: Ymse(M.fit(M.lo)), x2: X(M.hi), y2: Ymse(M.fit(M.hi)), stroke: H.NAVY, "stroke-width": 1.6 }));
      }
      CELLS.forEach(c => {
        const f = SL_FAM[c[0]], faint = (c[0] !== "dense" && c[0] !== "struct");
        const pt = H.marker(f.marker, X(c[M.ckey]), Ymse(c[4]), 4.5, f.color);
        pt.setAttribute("opacity", faint ? 0.3 : 0.85); pt.setAttribute("stroke", "#fff"); pt.setAttribute("stroke-width", 0.7); pt.style.cursor = "pointer";
        pt.addEventListener("mouseenter", e => H.showTip(`<b>${f.name} · ${c[1]}</b><br>GPG = ${c[2]} nats<br>ED = ${c[3]}<br>MSE = ${c[4]}`, e));
        pt.addEventListener("mousemove", H.moveTip);
        pt.addEventListener("mouseleave", H.hideTip);
        svg.appendChild(pt);
      });
      const bx = W - m.r - 198, by = m.t + 2;
      svg.appendChild(H.el("rect", { x: bx, y: by, width: 196, height: 40, rx: 5, fill: "rgba(255,255,255,0.92)", stroke: H.NAVY, "stroke-width": 0.9 }));
      svg.appendChild(H.txt(bx + 98, by + 17, M.eqn, { "text-anchor": "middle", class: "eqn", "font-size": 11, fill: H.NAVY }));
      svg.appendChild(H.txt(bx + 98, by + 32, "r = " + M.r, { "text-anchor": "middle", class: "eqn", "font-size": 11, fill: H.NAVY }));
      svg.appendChild(H.txt(m.l + (W - m.l - m.r) / 2, HT - 8, M.lab, { "text-anchor": "middle", class: "ax-title" }));
      yTitle(svg);
      const nlM = mkMarker(svg, H.RED, "NL"), spM = mkMarker(svg, H.BLUE, "SP");
      infoHost.innerHTML = ""; infoHost.appendChild(svg); infoC = { X, nlM, spM };
    }

    function update() {
      const t = +slider.value / 1000, nl = slInterp(SL_NL, t), sp = slInterp(SL_SP, t);
      place(lenC.nlM, lenC.X(nl.tok), Ymse(nl.mse));
      place(lenC.spM, lenC.X(sp.tok), Ymse(sp.mse));
      const ik = mode === "gpg" ? "gpg" : "ed";
      place(infoC.nlM, infoC.X(nl[ik]), Ymse(nl.mse));
      place(infoC.spM, infoC.X(sp[ik]), Ymse(sp.mse));
      readout.innerHTML = `At this detail: <b class="rd-nl">natural language</b> &mdash; ${Math.round(nl.tok)} tokens, loss stuck at <b>${nl.mse.toFixed(4)}</b>; &nbsp; <b class="rd-sp">structured</b> &mdash; ${Math.round(sp.tok)} tokens, loss at <b>${sp.mse.toFixed(4)}</b>.`;
    }

    btns.forEach(b => b.addEventListener("click", () => {
      mode = b.dataset.mode;
      btns.forEach(x => x.classList.toggle("on", x === b));
      buildInfo(); update();
    }));
    slider.addEventListener("input", update);
    buildLen(); buildInfo(); update();
  }
  reg("scaling-law", scalingLaw);


  // ---- boot ----
  function init() {
    progressBar();
    for (const id in MOUNTS) {
      const root = document.getElementById(id);
      if (root) {
        try { MOUNTS[id](root); }
        catch (e) { console.error("mount failed for #" + id, e); }
      }
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
