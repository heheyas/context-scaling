/* ============================================================
   Web-native SVG charts for the project page.
   Hand-rolled (no deps) in the paper's chart style:
   Palatino serif, #b2182b NL / #2166ac SP family colors,
   minimal y-grid, navy fit line + equation box, hover tooltips.
   ============================================================ */
(function () {
  "use strict";

  const RED = "#b2182b", BLUE = "#2166ac", GREEN = "#2ca02c", PURPLE = "#9467bd";
  const NAVY = "#16324f", INK = "#1a1a1a", AXIS = "#3a342a", GRID = "#cfc7b4";
  const SVGNS = "http://www.w3.org/2000/svg";

  // 15-cell canon: [kind, label, GPG, ED, MSE]
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
  const FAMILY = {
    dense:   { color: RED,    marker: "square",   name: "Natural language" },
    struct:  { color: BLUE,   marker: "circle",   name: "Structured" },
    spatial: { color: GREEN,  marker: "triangle", name: "Spatial variants" },
    abl:     { color: PURPLE, marker: "diamond",  name: "Field ablations" },
  };

  // ---- svg helpers ----
  function el(name, attrs) {
    const n = document.createElementNS(SVGNS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }
  function txt(x, y, s, attrs) {
    const t = el("text", Object.assign({ x, y }, attrs));
    t.textContent = s;
    return t;
  }
  function marker(kind, cx, cy, r, fill) {
    if (kind === "circle") return el("circle", { cx, cy, r, fill });
    if (kind === "square") return el("rect", { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, fill });
    if (kind === "triangle")
      return el("polygon", { points: `${cx},${cy - r * 1.15} ${cx - r * 1.1},${cy + r * 0.8} ${cx + r * 1.1},${cy + r * 0.8}`, fill });
    if (kind === "diamond")
      return el("polygon", { points: `${cx},${cy - r * 1.25} ${cx + r * 1.15},${cy} ${cx},${cy + r * 1.25} ${cx - r * 1.15},${cy}`, fill });
  }

  // shared tooltip
  let tip;
  function ensureTip() {
    if (tip) return tip;
    tip = document.createElement("div");
    tip.className = "chart-tip";
    document.body.appendChild(tip);
    return tip;
  }
  function showTip(html, evt) {
    const t = ensureTip();
    t.innerHTML = html;
    t.style.opacity = "1";
    moveTip(evt);
  }
  function moveTip(evt) {
    if (!tip) return;
    tip.style.left = (evt.clientX + 14) + "px";
    tip.style.top = (evt.clientY - 10 + window.scrollY) + "px";
  }
  function hideTip() { if (tip) tip.style.opacity = "0"; }

  function makeSVG(W, H) {
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg",
      preserveAspectRatio: "xMidYMid meet" });
    return svg;
  }

  // ============================================================
  //  Scaling-law chart: MSE vs GPG
  // ============================================================
  // Dual scaling-law figure: (a) MSE linear in GPG, (b) MSE power-law in ED,
  // both across the same 15 cells, sharing the converged-MSE y-axis (paper Fig.).
  function scalingChart(container) {
    const W = 1040, H = 440;
    const yLo = 0.4358, yHi = 0.4470;
    const pTop = 66, pBot = H - 58;          // shared vertical band for both panels
    const leftAxis = 78, panelGap = 74, rightPad = 20;
    const panelW = (W - leftAxis - rightPad - panelGap) / 2;
    const svg = makeSVG(W, H);
    const Y = v => pTop + (yHi - v) / (yHi - yLo) * (pBot - pTop);

    // shared MSE y-axis title (far left)
    const ytc = (pTop + pBot) / 2;
    const yt = txt(20, ytc, "Converged training MSE", { "text-anchor": "middle", class: "ax-title" });
    yt.setAttribute("transform", `rotate(-90 20 ${ytc})`);
    svg.appendChild(yt);

    // shared family legend (top, centred)
    const lg = [["circle", BLUE, "Structured", 96], ["square", RED, "Natural language", 128],
                ["triangle", GREEN, "Spatial variants", 116], ["diamond", PURPLE, "Field ablations", 112]];
    let lx = W / 2 - lg.reduce((a, b) => a + b[3], 0) / 2;
    lg.forEach(([mk, col, name, w]) => {
      svg.appendChild(marker(mk, lx, 26, 5, col));
      svg.appendChild(txt(lx + 12, 30, name, { class: "ax-lab", fill: INK }));
      lx += w;
    });

    // one panel: cfg = { x0, xLo, xHi, xVal, xticks, xfmt, xtitle, sub, fit, eqn, showY }
    function panel(cfg) {
      const { x0, xLo, xHi, xVal, xticks, xfmt, fit } = cfg;
      const x1 = x0 + panelW;
      const X = g => x0 + (g - xLo) / (xHi - xLo) * (x1 - x0);

      // y gridlines (+ labels only on the left panel)
      [0.436, 0.438, 0.440, 0.442, 0.444, 0.446].forEach(v => {
        svg.appendChild(el("line", { x1: x0, y1: Y(v), x2: x1, y2: Y(v), stroke: GRID, "stroke-width": 0.6, opacity: 0.55 }));
        if (cfg.showY) svg.appendChild(txt(x0 - 9, Y(v) + 4, v.toFixed(3), { "text-anchor": "end", class: "ax-lab" }));
      });
      // axes
      svg.appendChild(el("line", { x1: x0, y1: pBot, x2: x1, y2: pBot, stroke: AXIS, "stroke-width": 1 }));
      svg.appendChild(el("line", { x1: x0, y1: pTop, x2: x0, y2: pBot, stroke: AXIS, "stroke-width": 1 }));
      xticks.forEach(g => {
        svg.appendChild(el("line", { x1: X(g), y1: pBot, x2: X(g), y2: pBot + 5, stroke: AXIS }));
        svg.appendChild(txt(X(g), pBot + 20, xfmt(g), { "text-anchor": "middle", class: "ax-lab" }));
      });

      // fit — straight line (linear) or sampled curve (power law)
      let pts;
      if (fit.type === "linear") {
        pts = [xLo, xHi].map(g => `${X(g)},${Y(fit.a + fit.b * g)}`);
      } else {
        pts = [];
        for (let i = 0; i <= 48; i++) { const g = xLo + (xHi - xLo) * i / 48; pts.push(`${X(g)},${Y(fit.a * Math.pow(g, fit.b))}`); }
      }
      svg.appendChild(el("polyline", { points: pts.join(" "), fill: "none", stroke: NAVY, "stroke-width": 1.8 }));

      // SP-L10 highlight ring (last cell)
      const L10 = CELLS[CELLS.length - 1];
      svg.appendChild(el("circle", { cx: X(xVal(L10)), cy: Y(L10[4]), r: 11, fill: "none", stroke: INK, "stroke-width": 1.4 }));

      // points
      CELLS.forEach(c => {
        const f = FAMILY[c[0]];
        const pt = marker(f.marker, X(xVal(c)), Y(c[4]), 5.5, f.color);
        pt.setAttribute("stroke", "#fff"); pt.setAttribute("stroke-width", 0.8); pt.style.cursor = "pointer";
        pt.addEventListener("mouseenter", e => showTip(
          `<b>${f.name} · ${c[1]}</b><br>GPG = ${c[2]} nats<br>ED = ${c[3]}<br>MSE = ${c[4]}`, e));
        pt.addEventListener("mousemove", moveTip);
        pt.addEventListener("mouseleave", hideTip);
        svg.appendChild(pt);
      });

      // equation box (top-centre of the panel); eqn[0] is a run list, superscripts via tspan
      const bw = 258, bx = (x0 + x1) / 2 - bw / 2, by = pTop + 4;
      svg.appendChild(el("rect", { x: bx, y: by, width: bw, height: 46, rx: 6, fill: "rgba(255,255,255,0.92)", stroke: NAVY, "stroke-width": 0.9 }));
      const et = txt(bx + bw / 2, by + 19, "", { "text-anchor": "middle", class: "eqn", fill: NAVY });
      cfg.eqn[0].forEach(run => {
        const ts = el("tspan", run.sup ? { "baseline-shift": "super", "font-size": "8.5px" } : {});
        ts.textContent = run.t; et.appendChild(ts);
      });
      svg.appendChild(et);
      svg.appendChild(txt(bx + bw / 2, by + 37, cfg.eqn[1], { "text-anchor": "middle", class: "eqn", fill: NAVY }));

      // x-axis title (supports a subscript via cfg.sub)
      const xt = txt((x0 + x1) / 2, H - 12, cfg.xtitle, { "text-anchor": "middle", class: "ax-title" });
      if (cfg.sub) { const s = el("tspan", { "baseline-shift": "sub", "font-size": "10px" }); s.textContent = cfg.sub; xt.appendChild(s); }
      if (cfg.tail) { const t = el("tspan", {}); t.textContent = cfg.tail; xt.appendChild(t); }
      svg.appendChild(xt);
      // panel label
      svg.appendChild(txt(x0 - 2, pTop - 12, cfg.label, { "text-anchor": "start", class: "ax-title", "font-size": 13, fill: INK }));
    }

    // (a) linear in GPG
    panel({
      x0: leftAxis, xLo: 100, xHi: 222, xVal: c => c[2], showY: true,
      xticks: [100, 120, 140, 160, 180, 200, 220], xfmt: g => g,
      xtitle: "Caption information  GPG", sub: "total", tail: "  (nats)",
      fit: { type: "linear", a: 0.4549, b: -8.45e-5 },
      eqn: [[{ t: "MSE = 0.4549 − 8.45×10" }, { t: "−5", sup: true }, { t: "·GPG" }], "r = −0.984"],
      label: "(a) linear in GPG",
    });
    // (b) power law in ED
    panel({
      x0: leftAxis + panelW + panelGap, xLo: 0.74, xHi: 0.84, xVal: c => c[3], showY: false,
      xticks: [0.74, 0.76, 0.78, 0.80, 0.82, 0.84], xfmt: g => g.toFixed(2),
      xtitle: "Attribute recall  ED",
      fit: { type: "power", a: 0.4200, b: -0.2073 },
      eqn: [[{ t: "MSE = 0.4200·ED" }, { t: "−0.2073", sup: true }], "r = −0.971"],
      label: "(b) power law in ED",
    });

    container.appendChild(svg);
  }

  // ============================================================
  //  Prompter-scaling chart: GenEval++ vs prompter size
  // ============================================================
  // Prompter scaling — 1×4 panels reproducing paper Fig. 8. Qwen3.5 prompters
  // 0.8B→397B, no-thinking (dashed, hollow) vs thinking (solid, filled), L10 schema
  // + Qwen-Image diffuser fixed. GenEval++/WISE from data/qwen3.5-scaling-pe; VLM
  // structure & GSB read from Fig. 8.
  const PS_SIZES = ["0.8B", "4B", "9B", "35B", "122B", "397B"];
  const PS_NT = "#9ec0dc";
  const PS_PANELS = [
    { t: "(a) GenEval++", lo: 40, hi: 90, ticks: [40, 50, 60, 70, 80, 90], fmt: v => v,
      nt: [49, 65, 72, 79, 84, 85], th: [46, 67, 73, 80, 85, 87], unit: "%" },
    { t: "(b) WISE", lo: 0.45, hi: 0.9, ticks: [0.5, 0.6, 0.7, 0.8, 0.9], fmt: v => v.toFixed(2),
      nt: [0.52, 0.69, 0.74, 0.75, 0.82, 0.83], th: [0.50, 0.71, 0.76, 0.77, 0.85, 0.86], unit: "" },
    { t: "(c) VLM structure", lo: 3, hi: 7, ticks: [3, 4, 5, 6, 7], fmt: v => v,
      nt: [3.55, 3.9, 4.6, 5.1, 5.5, 5.9], th: [3.3, 4.4, 5.1, 5.7, 6.25, 6.6], unit: "/10" },
    { t: "(d) GSB", lo: 0, hi: 42, ticks: [0, 10, 20, 30, 40], fmt: v => v,
      nt: [5, 9, 15.5, 22, 26.5, 31.5], th: [2, 14, 22, 29, 35, 40], unit: "%" },
  ];
  function prompterChart(container) {
    const W = 1040, H = 300, PW = W / 4;
    const pin = { l: 44, r: 16, t: 48, b: 44 };
    const svg = makeSVG(W, H);

    // shared legend (top centre)
    const lx = W / 2 - 92, ly = 16;
    svg.appendChild(el("line", { x1: lx, y1: ly, x2: lx + 24, y2: ly, stroke: PS_NT, "stroke-width": 1.8, "stroke-dasharray": "5 3" }));
    svg.appendChild(el("rect", { x: lx + 8, y: ly - 3.5, width: 7, height: 7, fill: "#fff", stroke: PS_NT, "stroke-width": 1.4 }));
    svg.appendChild(txt(lx + 31, ly + 4, "no thinking", { class: "ax-lab", fill: INK }));
    svg.appendChild(el("line", { x1: lx + 128, y1: ly, x2: lx + 152, y2: ly, stroke: BLUE, "stroke-width": 2.4 }));
    svg.appendChild(marker("circle", lx + 140, ly, 4, BLUE));
    svg.appendChild(txt(lx + 159, ly + 4, "thinking", { class: "ax-lab", fill: INK }));

    PS_PANELS.forEach((p, pi) => {
      const ox = pi * PW;
      const x0 = ox + pin.l, x1 = ox + PW - pin.r, y0 = pin.t, y1 = H - pin.b;
      const X = i => x0 + i / (PS_SIZES.length - 1) * (x1 - x0);
      const Y = v => y1 - (v - p.lo) / (p.hi - p.lo) * (y1 - y0);

      svg.appendChild(txt((x0 + x1) / 2, y0 - 16, p.t, { "text-anchor": "middle", class: "ax-title", "font-size": 13.5 }));
      p.ticks.forEach(v => {
        svg.appendChild(el("line", { x1: x0, y1: Y(v), x2: x1, y2: Y(v), stroke: GRID, "stroke-width": 0.6, opacity: 0.5 }));
        svg.appendChild(txt(x0 - 6, Y(v) + 3.5, p.fmt(v), { "text-anchor": "end", class: "ax-lab", "font-size": 10 }));
      });
      svg.appendChild(el("line", { x1: x0, y1: y1, x2: x1, y2: y1, stroke: AXIS, "stroke-width": 1 }));
      svg.appendChild(el("line", { x1: x0, y1: y0, x2: x0, y2: y1, stroke: AXIS, "stroke-width": 1 }));
      PS_SIZES.forEach((s, i) => svg.appendChild(txt(X(i), y1 + 13, s.replace("B", ""), { "text-anchor": "middle", class: "ax-lab", "font-size": 9.5 })));
      svg.appendChild(txt((x0 + x1) / 2, H - 6, "prompter size (B)", { "text-anchor": "middle", class: "ax-lab", "font-size": 10, fill: MUTED }));

      const drawLine = (data, on) => {
        const ln = el("polyline", { points: data.map((v, i) => `${X(i)},${Y(v)}`).join(" "), fill: "none",
          stroke: on ? BLUE : PS_NT, "stroke-width": on ? 2.2 : 1.8 });
        if (!on) ln.setAttribute("stroke-dasharray", "5 3");
        svg.appendChild(ln);
        data.forEach((v, i) => {
          const mk = on ? marker("circle", X(i), Y(v), 3.6, BLUE)
                        : el("rect", { x: X(i) - 3, y: Y(v) - 3, width: 6, height: 6, fill: "#fff", stroke: PS_NT, "stroke-width": 1.4 });
          mk.style.cursor = "pointer";
          mk.addEventListener("mouseenter", e => showTip(`<b>${p.t.replace(/^\([a-d]\)\s*/, "")} · ${PS_SIZES[i]}</b><br>${on ? "thinking" : "no thinking"}: ${p.fmt(v)}${p.unit}`, e));
          mk.addEventListener("mousemove", moveTip);
          mk.addEventListener("mouseleave", hideTip);
          svg.appendChild(mk);
        });
      };
      drawLine(p.nt, false);
      drawLine(p.th, true);
    });

    container.appendChild(svg);
  }

  // ============================================================
  //  Reconstruction-probe widget: budget selector + trend chart
  // ============================================================
  const RECON = {
    budgets: [542, 803, 1374], levels: ["L6", "L8", "L10"],
    gt: "assets/recon_gt.jpg",
    nl: { dino: [0.706, 0.739, 0.731], siglip: [0.79, 0.79, 0.79],
          img: ["assets/recon_nl_542.jpg", "assets/recon_nl_803.jpg", "assets/recon_nl_1374.jpg"] },
    sp: { dino: [0.711, 0.828, 0.840], siglip: [0.72, 0.73, 0.84],
          img: ["assets/recon_sp_542.jpg", "assets/recon_sp_803.jpg", "assets/recon_sp_1374.jpg"] },
  };

  function reconTrend(container, cur) {
    container.innerHTML = "";
    const W = 560, H = 220, m = { l: 52, r: 80, t: 16, b: 40 };
    const xs = RECON.budgets, xLo = 480, xHi = 1440, yLo = 0.68, yHi = 0.86;
    const X = v => m.l + (v - xLo) / (xHi - xLo) * (W - m.l - m.r);
    const Y = v => m.t + (yHi - v) / (yHi - yLo) * (H - m.t - m.b);
    const svg = makeSVG(W, H);
    [0.70, 0.74, 0.78, 0.82, 0.86].forEach(v => {
      svg.appendChild(el("line", { x1: m.l, y1: Y(v), x2: W - m.r, y2: Y(v), stroke: GRID, "stroke-width": 0.6, opacity: 0.55 }));
      svg.appendChild(txt(m.l - 8, Y(v) + 4, v.toFixed(2), { "text-anchor": "end", class: "ax-lab" }));
    });
    svg.appendChild(el("line", { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b, stroke: AXIS, "stroke-width": 1 }));
    xs.forEach((b, i) => {
      svg.appendChild(txt(X(b), H - m.b + 18, b, { "text-anchor": "middle", class: "ax-lab" }));
    });
    // current-budget marker
    svg.appendChild(el("line", { x1: X(xs[cur]), y1: m.t, x2: X(xs[cur]), y2: H - m.b,
      stroke: "#b8862a", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.7 }));
    function series(data, color, mk, label) {
      svg.appendChild(el("polyline", { points: xs.map((b, i) => `${X(b)},${Y(data[i])}`).join(" "),
        fill: "none", stroke: color, "stroke-width": 2.2 }));
      xs.forEach((b, i) => {
        const p = marker(mk, X(b), Y(data[i]), i === cur ? 6.5 : 4.5, color);
        p.setAttribute("stroke", "#fff"); p.setAttribute("stroke-width", i === cur ? 1.2 : 0.7);
        svg.appendChild(p);
      });
      svg.appendChild(txt(X(xs[2]) + 10, Y(data[2]) + 4, label, { class: "end-lab", fill: color }));
    }
    series(RECON.sp.dino, BLUE, "circle", "SP");
    series(RECON.nl.dino, RED, "square", "NL");
    svg.appendChild(txt(m.l + (W - m.l - m.r) / 2, H - 6, "token budget", { "text-anchor": "middle", class: "ax-title" }));
    const yt = txt(14, m.t + (H - m.t - m.b) / 2, "DINOv3 similarity", { "text-anchor": "middle", class: "ax-title" });
    yt.setAttribute("transform", `rotate(-90 14 ${m.t + (H - m.t - m.b) / 2})`);
    svg.appendChild(yt);
    container.appendChild(svg);
  }

  function reconWidget(root) {
    let cur = 2; // default L10
    root.querySelector(".recon-gt").src = RECON.gt;
    const nlImg = root.querySelector(".recon-nl"), spImg = root.querySelector(".recon-sp");
    const nlSc = root.querySelector(".recon-nl-score"), spSc = root.querySelector(".recon-sp-score");
    const chart = root.querySelector(".recon-chart");
    const btns = [...root.querySelectorAll(".recon-seg button")];
    function render() {
      nlImg.src = RECON.nl.img[cur]; spImg.src = RECON.sp.img[cur];
      nlSc.innerHTML = `DINOv3 ${RECON.nl.dino[cur].toFixed(2)} &middot; SigLIP2 ${RECON.nl.siglip[cur].toFixed(2)}`;
      spSc.innerHTML = `DINOv3 ${RECON.sp.dino[cur].toFixed(2)} &middot; SigLIP2 ${RECON.sp.siglip[cur].toFixed(2)}`;
      btns.forEach((b, i) => b.classList.toggle("on", i === cur));
      reconTrend(chart, cur);
    }
    btns.forEach((b, i) => b.addEventListener("click", () => { cur = i; render(); }));
    render();
  }

  // ============================================================
  //  GPG visualisation (SYNTHETIC) — image + per-token surprisal bars.
  //  Blue = surprisal given the image; amber = the extra surprisal the
  //  image explains away (= per-token grounding gain). Toggle the image
  //  to animate the amber portion in/out. GPG = Σ amber.
  // ============================================================
  const GPG_VIZ = {
    // [token, surprisal WITH image (nats), grounding gain (nats)]
    tokens: [["A", 0.3, 0], ["silver", 0.5, 2.6], ["pickup", 0.6, 2.1], ["truck", 0.4, 2.9],
      ["transports", 1.1, 0.7], ["cardboard", 0.5, 3.1], ["boxes", 0.6, 2.8], ["a", 0.3, 0],
      ["wrapped", 0.9, 1.4], ["cylinder", 1.0, 1.2], ["and", 0.2, 0], ["Coca-Cola", 1.4, 0.4],
      ["cans", 0.8, 0.7], ["on", 0.2, 0], ["a", 0.3, 0], ["street", 0.7, 1.7]],
  };
  const ease = u => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;

  function gpgViz(root) {
    const toks = GPG_VIZ.tokens;
    const gpg = toks.reduce((a, [, , g]) => a + g, 0);
    const chart = root.querySelector(".gv-chart");
    const head = root.querySelector(".gv-head");
    const btn = root.querySelector(".gv-toggle");
    const imgwrap = root.querySelector(".gv-imgwrap");

    const W = 680, H = 300, m = { l: 38, r: 14, t: 18, b: 84 };
    const n = toks.length, yMax = 4.0, bw = (W - m.l - m.r) / n * 0.6;
    const Y = v => m.t + (1 - v / yMax) * (H - m.t - m.b);
    const base = Y(0);
    const svg = makeSVG(W, H);
    [0, 1, 2, 3, 4].forEach(v => {
      svg.appendChild(el("line", { x1: m.l, y1: Y(v), x2: W - m.r, y2: Y(v), stroke: GRID, "stroke-width": 0.6, opacity: 0.5 }));
      svg.appendChild(txt(m.l - 7, Y(v) + 4, v, { "text-anchor": "end", class: "ax-lab" }));
    });
    const bars = toks.map(([tok, nll, gain], i) => {
      const cx = m.l + (i + 0.5) / n * (W - m.l - m.r);
      const baseRect = el("rect", { x: cx - bw / 2, width: bw, fill: "#2166ac",
        y: Y(nll), height: Math.max(0, base - Y(nll)) });        // surprisal with image
      const gainRect = el("rect", { x: cx - bw / 2, width: bw, fill: "#d8973c", y: Y(nll), height: 0 });
      svg.appendChild(baseRect); svg.appendChild(gainRect);
      const tl = txt(cx + 3, base + 13, tok, { "text-anchor": "end", class: "ax-lab" });
      tl.setAttribute("transform", `rotate(-42 ${cx + 3} ${base + 13})`);
      svg.appendChild(tl);
      return { nll, gain, baseRect, gainRect };
    });
    // legend
    svg.appendChild(el("rect", { x: m.l, y: 2, width: 10, height: 10, fill: "#2166ac" }));
    svg.appendChild(txt(m.l + 15, 11, "surprisal given the image", { class: "ax-lab" }));
    svg.appendChild(el("rect", { x: m.l + 188, y: 2, width: 10, height: 10, fill: "#d8973c" }));
    svg.appendChild(txt(m.l + 203, 11, "explained by the image  (= grounding gain)", { class: "ax-lab" }));
    const yt = txt(11, m.t + (H - m.t - m.b) / 2, "per-token surprisal  (nats)", { "text-anchor": "middle", class: "ax-title" });
    yt.setAttribute("transform", `rotate(-90 11 ${m.t + (H - m.t - m.b) / 2})`);
    svg.appendChild(yt);
    chart.appendChild(svg);

    function paint(t) { // t: 1 = image on (amber collapsed) … 0 = image off (amber full)
      bars.forEach(b => {
        const top = Y(b.nll + b.gain * (1 - t));
        b.gainRect.setAttribute("y", top);
        b.gainRect.setAttribute("height", Math.max(0, Y(b.nll) - top));
      });
      head.innerHTML = `GPG<sub>total</sub> = Σ (grounding gain) = <b>${gpg.toFixed(1)}</b> nats`;
    }
    let on = true, raf = null;
    function animateTo(target) {
      const from = on ? 1 : 0, dur = 520, t0 = performance.now();
      if (raf) cancelAnimationFrame(raf);
      (function step(now) {
        const u = Math.min(1, (now - t0) / dur), v = from + (target - from) * ease(u);
        paint(v);
        if (u < 1) raf = requestAnimationFrame(step);
      })(t0);
      on = target === 1;
      btn.innerHTML = on ? "image <b>on</b> &nbsp;&middot;&nbsp; hide it" : "image <b>off</b> &nbsp;&middot;&nbsp; reveal it";
      btn.classList.toggle("off", !on);
      if (imgwrap) imgwrap.classList.toggle("hidden", !on);
    }
    btn.addEventListener("click", () => animateTo(on ? 0 : 1));
    paint(1);
    btn.innerHTML = "image <b>on</b> &nbsp;&middot;&nbsp; hide it";
  }

  // ============================================================
  //  ED visualisation (SYNTHETIC) — bipartite tuple matching with a
  //  "reveal the image's source set" toggle (mirrors GPG's reveal).
  //  green = matched · red = caption-only (hallucination, hurts precision)
  //  grey = source-only (missed, hurts recall) · ED = F0.5(P, R)
  // ============================================================
  // [tuple, matched/covered?, bbox [x1,y1,x2,y2] normalised 0-1 | null]
  // bboxes keyed to the real portrait truck photo (assets/gpg_truck.jpg)
  const ED_VIZ = {
    caption: [
      ["truck — silver", true, [0.02, 0.05, 1.0, 0.93]],
      ["truck — pickup", true, [0.55, 0.04, 1.0, 0.37]],
      ["boxes — cardboard", true, [0.0, 0.27, 0.86, 0.62]],
      ["cylinder — wrapped", true, [0.0, 0.30, 0.88, 0.58]],
      ["cans — Coca-Cola", true, [0.49, 0.23, 0.63, 0.35]],
      ["scene — street", true, [0.0, 0.82, 1.0, 1.0]],
      ["traffic cone — orange", false, null]],
    source: [
      ["truck — silver", true, [0.02, 0.05, 1.0, 0.93]],
      ["truck — pickup", true, [0.55, 0.04, 1.0, 0.37]],
      ["boxes — cardboard", true, [0.0, 0.27, 0.86, 0.62]],
      ["cylinder — wrapped", true, [0.0, 0.30, 0.88, 0.58]],
      ["cans — Coca-Cola", true, [0.49, 0.23, 0.63, 0.35]],
      ["scene — street", true, [0.0, 0.82, 1.0, 1.0]],
      ["boxes — green labels", false, [0.0, 0.31, 0.42, 0.53]],
      ["boxes — long, stacked", false, [0.0, 0.30, 0.88, 0.58]],
      ["boxes — count (several)", false, [0.0, 0.27, 0.86, 0.62]],
      ["scene — daytime", false, [0.0, 0.0, 1.0, 0.20]],
      ["background — palm trees, building", false, [0.0, 0.0, 0.66, 0.22]],
      ["road — manhole cover", false, [0.60, 0.80, 0.96, 0.99]]],
  };
  const ED_COLOR = { ok: "#2f6e3a", miss: "#7d7565", no: "#9a3b33" };
  function f05(p, r) { return p + r === 0 ? 0 : 1.25 * p * r / (0.25 * p + r); }

  function edViz(root) {
    const capBox = root.querySelector(".em-caption"), srcBox = root.querySelector(".em-source");
    const btn = root.querySelector(".em-reveal");
    const overlay = root.querySelector(".em-overlay"), flag = root.querySelector(".em-halluc");
    const pf = root.querySelector(".bf.p"), rf = root.querySelector(".bf.r");
    const pv = root.querySelector(".pv"), rv = root.querySelector(".rv"), edv = root.querySelector(".edv");
    let revealed = false;

    function showBox(bbox, state) {
      overlay.innerHTML = "";
      if (!bbox) { flag.classList.add("on"); return; }
      flag.classList.remove("on");
      const [x1, y1, x2, y2] = bbox, col = ED_COLOR[state];
      const fill = el("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: col, opacity: 0.14 });
      const line = el("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, fill: "none", stroke: col, "stroke-width": 2 });
      line.setAttribute("vector-effect", "non-scaling-stroke");
      overlay.appendChild(fill); overlay.appendChild(line);
    }
    function clearBox() { overlay.innerHTML = ""; flag.classList.remove("on"); }

    function mkChip(box, [t, m, bbox], stateOf) {
      const c = document.createElement("span");
      c.className = "etup neutral"; c.dataset.state = stateOf(m); c.textContent = t;
      c.addEventListener("mouseenter", () => showBox(bbox, c.dataset.state));
      c.addEventListener("mouseleave", clearBox);
      box.appendChild(c);
      return m;
    }
    let ok = 0, cov = 0;
    ED_VIZ.caption.forEach(d => { if (mkChip(capBox, d, m => m ? "ok" : "no")) ok++; });
    ED_VIZ.source.forEach(d => { if (mkChip(srcBox, d, m => m ? "ok" : "miss")) cov++; });
    const P = ok / ED_VIZ.caption.length, R = cov / ED_VIZ.source.length, ED = f05(P, R);

    function apply() {
      root.classList.toggle("revealed", revealed);
      [...capBox.children, ...srcBox.children].forEach(c => {
        c.classList.toggle("neutral", !revealed);
        if (revealed) {
          c.classList.add(c.dataset.state);
          c.innerHTML = (c.dataset.state === "ok" ? "✓ " : c.dataset.state === "no" ? "✗ " : "○ ") + c.textContent;
        } else {
          c.classList.remove("ok", "no", "miss");
          c.textContent = c.textContent.replace(/^[✓✗○]\s/, "");
        }
      });
      pf.style.width = (revealed ? P * 100 : 0) + "%";
      rf.style.width = (revealed ? R * 100 : 0) + "%";
      pv.innerHTML = revealed ? `<b>${P.toFixed(2)}</b> &nbsp;${ok}/${ED_VIZ.caption.length}` : "";
      rv.innerHTML = revealed ? `<b>${R.toFixed(2)}</b> &nbsp;${cov}/${ED_VIZ.source.length}` : "";
      edv.textContent = revealed ? ED.toFixed(2) : "—";
      btn.innerHTML = revealed ? "hide the source set" : "reveal the image&rsquo;s source set &rarr;";
      btn.classList.toggle("on", revealed);
    }
    btn.addEventListener("click", () => { revealed = !revealed; apply(); });
    apply();
  }

  // ============================================================
  //  Structured-prompt schema viz — real JSON (draw/json_data_example/0.json)
  //  Image + bbox overlay: hover an element to locate it; click for its
  //  schema fields. Shows every conditioning axis as an addressable slot.
  // ============================================================
  const SCHEMA_LABELS = {
    1: "White tufted sofa", 2: "Black coffee table", 3: "Black armchair (left)",
    4: "Black armchair (right)", 5: "Black fireplace", 6: "Textured artwork",
    7: "Paper floor lamp", 8: "Slender vase", 9: "Bulbous vase", 10: "White side table",
    11: "Black cube table", 12: "Mantel decor", 13: "Stacked books", 14: "Dark stacked books",
    15: "Stacked books", 16: "Black bowl", 17: "Black glossy bowl", 18: "Golden bowl",
    19: "Clear glasses", 20: "Golden bowl (frothy)", 21: "Brass wall sconce", 22: "Window (left)",
    23: "Window (centre)", 24: "Window (right)", 25: "Curtain (left)", 26: "Curtain (right)",
    27: "Light-gray rug", 28: "Wooden ceiling", 29: "Wall (left)", 30: "Wall (right)",
  };
  function parseBbox(s) {
    const m = /<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*<\/bbox>/.exec(s || "");
    return m ? [+m[1], +m[2], +m[3], +m[4]] : null;
  }
  function humanize(k) { return k.replace(/_/g, " "); }
  function fieldVal(v) {
    if (v && typeof v === "object") return Object.entries(v).map(([k, x]) => `${humanize(k)} — ${x}`).join("; ");
    return String(v);
  }

  function schemaViz(root) {
    const overlay = root.querySelector(".sv-overlay");
    const cap = root.querySelector(".sv-cap");
    const list = root.querySelector(".sv-list");
    const allBtn = root.querySelector(".sv-all");
    let pinned = null, showAll = false, EL = [];

    function rect(b, color, op, w) {
      const r = el("rect", { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1],
        fill: color, "fill-opacity": op, stroke: color, "stroke-width": w });
      r.setAttribute("vector-effect", "non-scaling-stroke");
      return r;
    }
    function draw(e) {
      overlay.innerHTML = "";
      if (showAll) EL.forEach(x => { if (x.bbox && (!e || x.id !== e.id)) overlay.appendChild(rect(x.bbox, x.grp === "e" ? "#2166ac" : "#8c8473", 0, 0.8)); });
      if (e && e.bbox) overlay.appendChild(rect(e.bbox, e.grp === "e" ? "#2166ac" : "#b8862a", 0.16, 2.2));
    }
    function setCap(e) {
      cap.innerHTML = e
        ? `<b>#${e.id} ${e.label}</b> &nbsp;<span class="dep">depth ${e.depth}</span><br>${e.caption}`
        : `<span class="muted">hover an element to locate it &middot; click for its full fields</span>`;
    }

    fetch("assets/schema.json").then(r => r.json()).then(data => {
      const mk = (e, grp) => ({
        id: e.id, grp, label: SCHEMA_LABELS[e.id] || (e.caption || "").slice(0, 28),
        bbox: parseBbox(e.position), depth: e.depth, caption: e.caption,
        fields: Object.entries(e).filter(([k]) => !["id", "caption", "position", "depth"].includes(k)),
      });
      EL = [...(data.elements || []).map(e => mk(e, "e")),
            ...((data.scene && data.scene.elements) || []).map(e => mk(e, "s"))];

      const g = root.querySelector(".sv-global");
      g.innerHTML = `<span class="gk">intent</span> ${data.intent || ""}` +
        `<div class="gchips"><span><b>style</b> ${data.style || "—"}</span>` +
        `<span><b>atmosphere</b> ${data.atmosphere || "—"}</span>` +
        `<span><b>lighting</b> natural + brass sconce</span></div>`;

      function addGroup(grp, title) {
        const h = document.createElement("p"); h.className = "sv-grouph"; h.textContent = title;
        list.appendChild(h);
        EL.filter(e => e.grp === grp).forEach(e => {
          const item = document.createElement("div");
          item.className = "sv-item " + (grp === "e" ? "elem" : "scene");
          const row = document.createElement("div");
          row.className = "sv-row";
          row.innerHTML = `<span class="idx">${e.id}</span><span class="lab">${e.label}</span><span class="dep">d${e.depth}</span>`;
          const detail = document.createElement("div");
          detail.className = "sv-detail";
          const bb = e.bbox ? e.bbox.join(" ") : "—";
          detail.innerHTML =
            `<p class="sv-d-cap">${e.caption}</p>` +
            `<dl>` +
            e.fields.map(([k, v]) => `<dt>${humanize(k)}</dt><dd>${fieldVal(v)}</dd>`).join("") +
            `<dt>position</dt><dd class="mono">&lt;bbox&gt; ${bb} &middot; depth ${e.depth}</dd>` +
            `</dl>`;
          item.appendChild(row); item.appendChild(detail);

          row.addEventListener("mouseenter", () => { if (!pinned) { draw(e); setCap(e); } });
          row.addEventListener("mouseleave", () => { if (!pinned) { draw(null); setCap(null); } });
          row.addEventListener("click", () => {
            if (pinned === e) { pinned = null; item.classList.remove("open"); draw(null); setCap(null); }
            else {
              if (pinned) list.querySelectorAll(".sv-item.open").forEach(i => i.classList.remove("open"));
              pinned = e; item.classList.add("open"); draw(e); setCap(e);
              item.scrollIntoView({ block: "nearest", behavior: "smooth" });
            }
          });
          list.appendChild(item);
        });
      }
      addGroup("e", "Elements · foreground objects (20)");
      addGroup("s", "Scene · background (10)");
    }).catch(() => { list.innerHTML = '<p class="sv-grouph">could not load schema.json (serve over http)</p>'; });

    allBtn.addEventListener("click", () => {
      showAll = !showAll; allBtn.classList.toggle("on", showAll);
      allBtn.textContent = showAll ? "hide all regions" : "show all 30 regions";
      draw(pinned);
    });
    setCap(null);
  }

  // ============================================================
  //  Per-stage diagrams: SFT / Cold-start / RFT
  // ============================================================
  const SERIF = "'Hanken Grotesk', system-ui, sans-serif";
  const MONO = "'IBM Plex Mono', monospace";
  const SFT_C = "#cf6f66", CS_C = "#b23d34", RFT_C = "#821e1a";
  const PAPER = "#faf6ec", RULE = "#d6cfb9", SOFT = "#8c8473", MUTED = "#5e574a";

  function rrect(x, y, w, h, attrs) {
    return el("rect", Object.assign({ x, y, width: w, height: h, rx: 5, ry: 5 }, attrs));
  }
  // line + filled arrowhead; opts: {w, dash, head}
  function arrow(x1, y1, x2, y2, color, opts) {
    opts = opts || {};
    const g = el("g", {});
    const ln = el("line", { x1, y1, x2, y2, stroke: color, "stroke-width": opts.w || 1.6, "stroke-linecap": "round" });
    if (opts.dash) ln.setAttribute("stroke-dasharray", opts.dash);
    g.appendChild(ln);
    const ang = Math.atan2(y2 - y1, x2 - x1), s = opts.head || 6.5;
    const a1 = ang + Math.PI * 0.83, a2 = ang - Math.PI * 0.83;
    g.appendChild(el("polygon", {
      points: `${x2},${y2} ${x2 + s * Math.cos(a1)},${y2 + s * Math.sin(a1)} ${x2 + s * Math.cos(a2)},${y2 + s * Math.sin(a2)}`,
      fill: color
    }));
    return g;
  }
  // box with stacked centered text lines; each line: {t, size, fill, mono, weight, italic, dy}
  function fbox(x, y, w, h, lines, opts) {
    opts = opts || {};
    const g = el("g", {});
    g.appendChild(rrect(x, y, w, h, {
      fill: opts.fill || PAPER, stroke: opts.stroke || RULE, "stroke-width": opts.sw || 1.2
    }));
    if (opts.accent) g.appendChild(el("rect", { x, y, width: 3.5, height: h, fill: opts.accent, rx: 2 }));
    const lh = opts.lh || 14.5, total = lines.length * lh, cy = y + h / 2 - total / 2 + lh - 4;
    lines.forEach((ln, i) => {
      g.appendChild(txt(x + w / 2, cy + i * lh + (ln.dy || 0), ln.t, {
        "text-anchor": "middle", "font-size": ln.size || opts.size || 12.5,
        fill: ln.fill || opts.tcolor || INK, "font-family": ln.mono ? MONO : SERIF,
        "font-weight": ln.weight || "normal", "font-style": ln.italic ? "italic" : "normal"
      }));
    });
    return g;
  }

  // ---- SFT: real before/after prompt-to-SP comparison (bbox overlay + toggle) ----
  const SBX_PAL = ["#2166ac", "#cf6f66", "#2ca02c", "#b8862a", "#9467bd", "#1c7293",
                   "#b23d34", "#5f8a3a", "#c25b8c", "#7a6f5d", "#3a7ca5", "#a0522d"];
  function sbxParse(s) {
    const m = /(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)/.exec(s || "");
    return m ? [+m[1] / 999, +m[2] / 999, +m[3] / 999, +m[4] / 999] : null;
  }
  const SFT_PANELS = [
    { pair: "Glasses & book", stage: "Before SFT", cls: "before", img: "assets/sft_glasses_before.jpg", json: "assets/sft_glasses_before.json", hl: "3" },
    { pair: "Glasses & book", stage: "After SFT",  cls: "after",  img: "assets/sft_glasses_after.jpg",  json: "assets/sft_glasses_after.json",  hl: "2" },
    { pair: "Man with guitar", stage: "Before SFT", cls: "before", img: "assets/sft_guitar_before.jpg", json: "assets/sft_guitar_before.json", hl: "3" },
    { pair: "Man with guitar", stage: "After SFT",  cls: "after",  img: "assets/sft_guitar_after.jpg",  json: "assets/sft_guitar_after.json",  hl: "2" }
  ];
  function sftCompare(root) {
    const wrap = root.querySelector(".sftc-pair");
    const meta = root.querySelector(".sftc-meta");
    const boxBtn = root.querySelector(".sftc-box");
    let showBoxes = true;
    const defMeta = "Hover a box to read its element caption.";
    const setMeta = h => { meta.innerHTML = h || defMeta; };
    const canvases = [];

    function buildPanel(p) {
      const col = document.createElement("div");
      col.className = "sftc-col";
      col.innerHTML = `<p class="sftc-h"><span class="pp">${p.pair}</span>` +
        `<span class="st ${p.cls}">${p.stage}</span></p><div class="sftc-canvas"></div>`;
      wrap.appendChild(col);
      return col.querySelector(".sftc-canvas");
    }

    function renderPanel(cv, d, p) {
      const r = (d.ratio || "1:1").split(":").map(Number);
      const H = 228, W = Math.round(H * r[0] / r[1]);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "sbx-svg" });
      const im = el("image", { x: 0, y: 0, width: W, height: H, preserveAspectRatio: "xMidYMid slice" });
      im.setAttribute("href", p.img);
      im.setAttributeNS("http://www.w3.org/1999/xlink", "href", p.img);
      svg.appendChild(im);
      const ov = el("g", { class: "sft-ov" });
      const fg = (d.elements || []).map(e => ({ e, layer: "fg" }));
      const bg = ((d.scene || {}).elements || []).map(e => ({ e, layer: "bg" }));
      bg.concat(fg).forEach(({ e, layer }) => {                  // bg behind fg
        const bb = sbxParse(e.position || e.bbox);
        if (!bb) return;
        const idn = parseInt(e.id, 10) || 1;
        const color = SBX_PAL[(idn - 1 + SBX_PAL.length) % SBX_PAL.length];
        const hl = String(e.id) === String(p.hl);
        const x = bb[0] * W, y = bb[1] * H, w = Math.max(2, (bb[2] - bb[0]) * W), h = Math.max(2, (bb[3] - bb[1]) * H);
        const g = el("g", { class: "sft-bx" });
        g.appendChild(el("rect", { x, y, width: w, height: h, fill: color, "fill-opacity": 0, rx: 2, class: "fillr" }));
        g.appendChild(el("rect", { x, y, width: w, height: h, fill: "none", stroke: color,
          "stroke-width": hl ? 3 : (layer === "fg" ? 2 : 1.3), rx: 2 }));
        const lab = "#" + e.id, tw = lab.length * 6 + 5;
        const tx = Math.min(Math.max(x, 1), W - tw - 1), ty = Math.min(Math.max(y, 0.5), H - 13.5);
        const tag = el("g", {});
        tag.appendChild(el("rect", { x: tx, y: ty, width: tw, height: 13, fill: color, rx: 2 }));
        tag.appendChild(txt(tx + 3.5, ty + 9.7, lab, { "font-size": 8.5, fill: "#fff", "font-family": MONO, "font-weight": "600" }));
        g.appendChild(tag);
        const cap = (e.caption || "").replace(/\s+/g, " ").trim();
        g.addEventListener("mouseenter", () => { g.classList.add("hot");
          setMeta(`<b style="color:${color}">#${e.id}</b> <span class="lyr">${layer}</span> &middot; ${cap} <span class="src">${p.pair} &middot; ${p.stage}</span>`); });
        g.addEventListener("mouseleave", () => { g.classList.remove("hot"); setMeta(null); });
        ov.appendChild(g);
      });
      svg.appendChild(ov);
      cv.innerHTML = "";
      cv.appendChild(svg);
      cv.classList.toggle("ov-off", !showBoxes);
    }

    function applyToggle() {
      boxBtn.classList.toggle("on", showBoxes);
      boxBtn.textContent = showBoxes ? "bounding boxes: on" : "bounding boxes: off";
      canvases.forEach(cv => cv.classList.toggle("ov-off", !showBoxes));
    }
    boxBtn.addEventListener("click", () => { showBoxes = !showBoxes; applyToggle(); });

    Promise.all(SFT_PANELS.map(p => fetch(p.json).then(r => r.json()).catch(() => null)))
      .then(ds => {
        SFT_PANELS.forEach((p, i) => {
          const cv = buildPanel(p);
          canvases.push(cv);
          if (ds[i]) renderPanel(cv, ds[i], p);
        });
        setMeta(null);
      });
  }

  // ---- Cold-start: distil derive-from-prompt CoT; no image at inference ----
  function coldstartFig(container) {
    const W = 760, H = 300;
    const svg = makeSVG(W, H);
    const tY = 96, sY = 234, bh = 50;                       // lane centres, box height
    const yT = y => y - bh / 2;
    // teacher: 5 boxes ; student: 4 (student.<think> aligns under teacher.<student>)
    const T = { prompt: [10, 106], img: [128, 88], think: [232, 150], student: [400, 158], json: [574, 140] };
    const S = { prompt: [10, 106], img: [128, 88], think: [400, 158], json: [574, 140] };
    function laneLabel(y, t, c) { svg.appendChild(txt(10, y - 50, t, { "font-size": 11, fill: c, "font-family": MONO, "letter-spacing": "0.04em" })); }
    laneLabel(tY, "TEACHER · TRAINING — SEES THE IMAGE", CS_C);
    laneLabel(sY, "STUDENT πθ · INFERENCE — NO IMAGE", BLUE);
    const promptLines = [{ t: "user prompt", size: 12 }, { t: "“a cozy living room”", size: 9.5, italic: true, fill: MUTED }];
    // teacher lane: prompt → image → <think> → <student> → JSON
    svg.appendChild(fbox(T.prompt[0], yT(tY), T.prompt[1], bh, promptLines));
    svg.appendChild(fbox(T.img[0], yT(tY), T.img[1], bh, [{ t: "◉ reference", size: 12 }, { t: "image", size: 12 }], { stroke: GREEN, fill: "#eef5ee" }));
    svg.appendChild(fbox(T.think[0], yT(tY), T.think[1], bh, [{ t: "<think> … </think>", size: 11.5, weight: "600", fill: CS_C, mono: true }, { t: "image-grounded reasoning", size: 9.5, italic: true, fill: MUTED }], { accent: CS_C, fill: "#fbf1ee" }));
    svg.appendChild(fbox(T.student[0], yT(tY), T.student[1], bh, [{ t: "<student> … </student>", size: 11.5, weight: "600", fill: BLUE, mono: true }, { t: "the same, as a prompt-only derivation", size: 8.5, italic: true, fill: MUTED }], { accent: BLUE, fill: "#eef3f8" }));
    svg.appendChild(fbox(T.json[0], yT(tY), T.json[1], bh, [{ t: "JSON prompt", size: 11.5, weight: "600", mono: true, fill: INK }, { t: "the structured caption", size: 9, italic: true, fill: MUTED }], { accent: INK, fill: PAPER }));
    // student lane: prompt → (no image) → <think> → JSON
    svg.appendChild(fbox(S.prompt[0], yT(sY), S.prompt[1], bh, promptLines));
    svg.appendChild(fbox(S.img[0], yT(sY), S.img[1], bh, [{ t: "no image", size: 12, fill: SOFT }, { t: "at inference", size: 10, fill: SOFT }], { stroke: SOFT, fill: "#f1ede2" }));
    svg.appendChild(el("line", { x1: S.img[0] + 6, y1: yT(sY) + bh - 6, x2: S.img[0] + S.img[1] - 6, y2: yT(sY) + 6, stroke: SOFT, "stroke-width": 1.4 }));
    svg.appendChild(fbox(S.think[0], yT(sY), S.think[1], bh, [{ t: "<think> … </think>", size: 11.5, weight: "600", fill: BLUE, mono: true }, { t: "distilled from <student>", size: 9, italic: true, fill: MUTED }], { accent: BLUE, fill: "#eef3f8" }));
    svg.appendChild(fbox(S.json[0], yT(sY), S.json[1], bh, [{ t: "JSON prompt", size: 11.5, weight: "600", mono: true, fill: INK }, { t: "the student emits it", size: 9, italic: true, fill: MUTED }], { accent: INK, fill: PAPER }));
    // flow arrows
    function flow(y, boxes) {
      for (let i = 0; i < boxes.length - 1; i++)
        svg.appendChild(arrow(boxes[i][0] + boxes[i][1] + 3, y, boxes[i + 1][0] - 3, y, AXIS, { w: 1.5 }));
    }
    flow(tY, [T.prompt, T.img, T.think, T.student, T.json]);
    flow(sY, [S.prompt, S.img, S.think, S.json]);
    // distillation: teacher's <student> block trains the student's <think>
    const dx = T.student[0] + T.student[1] / 2;
    svg.appendChild(arrow(dx, yT(tY) + bh + 2, dx, yT(sY) - 3, BLUE, { w: 1.8, dash: "6 4", head: 7 }));
    svg.appendChild(txt(dx + 10, (tY + sY) / 2 + 4, "distil  <student> → <think>", { "font-size": 11, fill: BLUE, "font-family": SERIF, "font-style": "italic", "text-anchor": "start" }));
    container.appendChild(svg);
  }

  // ---- RFT: on-policy loop, two rewards, two rulers ----
  function rftFig(container) {
    const W = 700, H = 300;
    const svg = makeSVG(W, H);
    const midY = 90, bh = 56;
    const P = [16, 116], REN = [150, 92], VER = [264, 122];
    const PANEL = { x: 414, y: 34, w: 270, h: 150 };
    // prompter (image-free policy) — samples its own rollout
    svg.appendChild(fbox(P[0], midY - bh / 2, P[1], bh, [
      { t: "Prompter πθ", size: 13, weight: "600", fill: RFT_C },
      { t: "image-free policy", size: 10.5, italic: true, fill: MUTED }], { accent: RFT_C, fill: "#f7ecea" }));
    // diffuser renders
    svg.appendChild(fbox(REN[0], midY - 26, REN[1], 52, [
      { t: "diffuser", size: 12.5 }, { t: "render → image", size: 10.5, mono: true, fill: MUTED }], { fill: PAPER }));
    // verifier as a gate
    svg.appendChild(fbox(VER[0], midY - 32, VER[1], 64, [
      { t: "QA verifier", size: 12.5, weight: "600", fill: "#2f6e3a" },
      { t: "gate: accept / reject", size: 10.5, mono: true, fill: MUTED },
      { t: "coverage · structure · aesthetics", size: 9.5, fill: SOFT }], { accent: GREEN, fill: "#eef5ee", lh: 14 }));
    // OPSD panel: teacher (sees image) distils into student (image-free)
    svg.appendChild(rrect(PANEL.x, PANEL.y, PANEL.w, PANEL.h, { fill: "#fbf8f1", stroke: RULE, "stroke-width": 1.2 }));
    svg.appendChild(txt(PANEL.x + PANEL.w / 2, PANEL.y + 18, "OPSD · on-policy self-distillation", { "text-anchor": "middle", "font-size": 11.5, "font-weight": "600", fill: INK, "font-family": MONO }));
    const rw = PANEL.w - 24, rx = PANEL.x + 12;
    svg.appendChild(fbox(rx, PANEL.y + 30, rw, 38, [{ t: "teacher π★ — sees the reference image", size: 11, fill: BLUE }], { accent: BLUE, fill: "#eef3f8" }));
    svg.appendChild(fbox(rx, PANEL.y + 74, rw, 38, [{ t: "student πθ — image-free", size: 11, fill: RFT_C }], { accent: RFT_C, fill: "#f7ecea" }));
    svg.appendChild(txt(PANEL.x + PANEL.w / 2, PANEL.y + PANEL.h - 9, "match next-token distributions along the rollout", { "text-anchor": "middle", "font-size": 10, "font-style": "italic", fill: MUTED, "font-family": SERIF }));
    // arrows
    svg.appendChild(arrow(P[0] + P[1] + 3, midY, REN[0] - 3, midY, AXIS, { w: 1.6 }));
    svg.appendChild(txt((P[0] + P[1] + REN[0]) / 2, midY - 7, "rollout", { "text-anchor": "middle", "font-size": 10, fill: SOFT, "font-family": MONO }));
    svg.appendChild(arrow(REN[0] + REN[1] + 3, midY, VER[0] - 3, midY, AXIS, { w: 1.6 }));
    svg.appendChild(arrow(VER[0] + VER[1] + 3, midY, PANEL.x - 3, midY, AXIS, { w: 1.6 }));
    svg.appendChild(txt((VER[0] + VER[1] + PANEL.x) / 2, midY - 7, "accepted", { "text-anchor": "middle", "font-size": 10, fill: "#2f6e3a", "font-family": MONO }));
    // feedback loop: panel -> down -> left -> up into prompter (only the student updates)
    const fy = H - 24, pcx = P[0] + P[1] / 2, panelcx = PANEL.x + PANEL.w / 2;
    svg.appendChild(el("path", {
      d: `M${panelcx},${PANEL.y + PANEL.h} L${panelcx},${fy} L${pcx},${fy} L${pcx},${midY + bh / 2 + 2}`,
      fill: "none", stroke: RFT_C, "stroke-width": 1.8, "stroke-dasharray": "6 4"
    }));
    svg.appendChild(arrow(pcx, midY + bh / 2 + 14, pcx, midY + bh / 2 + 2, RFT_C, { w: 1.8, head: 7 }));
    svg.appendChild(txt((panelcx + pcx) / 2, fy - 7, "gradient update — the verifier gates, OPSD teaches; only the student is trained", { "text-anchor": "middle", "font-size": 11, fill: RFT_C, "font-family": SERIF, "font-style": "italic" }));
    container.appendChild(svg);
  }

  // ---- init on DOM ready ----
  // ============================================================
  //  Figure 9 — three-stage prompter-training pipeline (SFT · Cold-start · RFT).
  //  Native reproduction; replaces the separate cold-start & RFT diagrams.
  // ============================================================
  function trainPipeline(container) {
    const W = 1000, HT = 300, svg = makeSVG(W, HT);
    const MAG = "#a5318f", GRN = "#2f8f4e", GRY = "#5e574a";  // paper palette: LLM/SP magenta, CoT green, i/o grey
    const IH = 23;  // tab height
    const GAP = 8;  // breathing room between a box and its input/output tabs
    const SPRING = "transform 0.55s cubic-bezier(0.34, 1.56, 0.64, 1)";  // overshoot ease → spring-back

    // Make one <g> draggable; it springs back to its home position when released.
    function makeDraggable(g) {
      let startX = 0, startY = 0, scale = 1, active = false;
      function onMove(e) {
        if (!active) return;
        g.style.transform = `translate(${(e.clientX - startX) * scale}px, ${(e.clientY - startY) * scale}px)`;
      }
      function onUp() {
        active = false;
        g.style.transition = SPRING;
        g.style.transform = "translate(0px, 0px)";
        g.style.cursor = "grab";
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      }
      g.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
        scale = r.width ? vb.width / r.width : 1;
        startX = e.clientX; startY = e.clientY; active = true;
        g.style.transition = "none";
        g.style.cursor = "grabbing";
        svg.appendChild(g);  // bring the grabbed block to the front
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
      });
    }

    // rounded labelled box, transparent fill, coloured outline — each is its OWN draggable <g>.
    // opts: {dash, lh, sw}
    function rbox(x, y, w, h, label, color, opts) {
      opts = opts || {};
      const g = el("g", { class: "pl-unit" });
      g.style.cursor = "grab"; g.style.touchAction = "none"; g.style.transition = SPRING;
      const r = rrect(x, y, w, h, { fill: "none", stroke: color, "stroke-width": opts.sw || 1.4 });
      if (opts.dash) r.setAttribute("stroke-dasharray", "4 3");
      g.appendChild(r);
      const lines = Array.isArray(label) ? label : [{ t: label, weight: "600" }];
      const lh = opts.lh || 12.5, cy = y + h / 2 - lines.length * lh / 2 + lh - 3.5;
      lines.forEach((ln, i) => g.appendChild(txt(x + w / 2, cy + i * lh, ln.t, {
        "text-anchor": "middle", "font-size": ln.size || 11.5, fill: ln.fill || color,
        "font-family": SERIF, "font-weight": ln.weight || "normal", "font-style": ln.italic ? "italic" : "normal" })));
      svg.appendChild(g);
      makeDraggable(g);
      return g;
    }
    // input tab below the box (bottom-left); output tabs above the box (top-right), with a small gap
    function tabIn(bx, by, bh, label, w) { rbox(bx + 8, by + bh + GAP, w, IH, [{ t: label, size: label.length > 8 ? 9 : 11 }], GRY); }
    function tabsOut(bx, by, bw, outs) {
      let ox = bx + bw - 8;
      outs.slice().reverse().forEach(o => { ox -= o.w; rbox(ox, by - IH - GAP, o.w, IH, [{ t: o.t, size: 10.5, italic: o.it }], o.c, { dash: o.dash }); ox -= 6; });
    }
    const ptitle = (cx, t, sub) => {
      svg.appendChild(txt(cx, 25, t, { "text-anchor": "middle", "font-family": SERIF, "font-weight": "600", "font-size": 14.5, fill: INK }));
      svg.appendChild(txt(cx, HT - 12, sub, { "text-anchor": "middle", "font-family": SERIF, "font-style": "italic", "font-size": 11.5, fill: MUTED }));
    };
    [180, 372].forEach(x => svg.appendChild(el("line", { x1: x, y1: 36, x2: x, y2: HT - 30, stroke: RULE, "stroke-width": 1, "stroke-dasharray": "2 4" })));

    // Static connector arrows live under the boxes so a lifted box passes over them.
    // (a) SFT
    rbox(43, 198, 104, 50, "LLM prompter", MAG, { lh: 13 });
    tabIn(43, 198, 50, "Prompt", 62);
    tabsOut(43, 198, 104, [{ t: "SP", w: 42, c: MAG }]);
    ptitle(95, "(a) SFT", "learn the target SP distribution");

    // (b) Cold-start
    rbox(224, 198, 104, 50, "LLM prompter", MAG, { lh: 13 });
    tabIn(224, 198, 50, "Prompt", 62);
    tabsOut(224, 198, 104, [{ t: "CoT", w: 42, c: GRN }, { t: "SP", w: 38, c: MAG }]);
    ptitle(276, "(b) Cold-start", "bootstrap image-free derivation");

    // (c) RFT — student (left) distilled from a frozen image-conditioned teacher (right)
    // connector arrows first (static), boxes on top
    svg.appendChild(arrow(550, 166, 550, 156, AXIS, { w: 1.4 }));  // SP tab → diffuser
    svg.appendChild(arrow(546, 132, 546, 120, AXIS, { w: 1.4 }));  // diffuser → Image
    svg.appendChild(arrow(582, 107, 600, 107, AXIS, { w: 1.4 }));  // Image → verifier
    // student
    rbox(444, 198, 130, 48, [{ t: "LLM prompter", weight: "600" }, { t: "(student)", size: 9.5, italic: true, fill: MUTED }], MAG, { lh: 14 });
    tabIn(444, 198, 48, "Prompt", 66);
    tabsOut(444, 198, 130, [{ t: "CoT", w: 40, c: GRN }, { t: "SP", w: 36, c: MAG }]);
    // render path from the student SP up to Image, then the verifier gate
    rbox(506, 132, 78, 22, [{ t: "diffuser", size: 10.5, fill: GRY }], GRY);
    rbox(510, 96, 72, 22, [{ t: "Image", size: 10.5, fill: GRY }], GRY);
    rbox(600, 94, 84, 30, [{ t: "verifier", size: 10.5, weight: "600" }, { t: "gate", size: 9, italic: true }], MAG, { dash: true, lh: 12.5 });
    // teacher (frozen, sees the reference image)
    rbox(762, 198, 140, 48, [{ t: "LLM teacher", weight: "600", fill: BLUE }, { t: "(frozen)", size: 9.5, italic: true, fill: MUTED }], BLUE, { lh: 14 });
    tabIn(762, 198, 48, "Prompt + ref. image", 126);
    tabsOut(762, 198, 140, [{ t: "thinking", w: 54, c: GRN, dash: true, it: true }, { t: "CoT", w: 38, c: GRN, dash: true }, { t: "SP", w: 32, c: MAG, dash: true }]);
    // OPSD: distil the teacher's targets into the image-free student
    svg.appendChild(arrow(762, 220, 578, 220, MAG, { w: 1.8, dash: "6 4", head: 7 }));
    svg.appendChild(txt(670, 213, "OPSD · distil", { "text-anchor": "middle", "font-size": 11, "font-family": SERIF, "font-style": "italic", fill: MAG }));
    ptitle(673, "(c) RFT", "improve on self-generated trajectories");

    // discoverability hint
    svg.appendChild(txt(W - 6, 15, "drag a box — it springs back", {
      "text-anchor": "end", "font-size": 10, "font-family": SERIF, "font-style": "italic", fill: MUTED, opacity: 0.72 }));

    container.appendChild(svg);
  }

  // ============================================================
  //  Zero-shot SP-editing showcase (paper Fig. 1 bottom) + bbox overlay
  //  Data from scripts/plot_sp_edit_showcase.py (bboxes in 0–1000 units).
  // ============================================================
  const SPEDIT = {
    A: {
      label: "Case A · desk", base: "assets/sp_edit/A_base.jpg",
      edits: [
        { k: "Move", sub: "swap books", img: "assets/sp_edit/A_move.jpg", field: "elements[0,2].position",
          old: "300 470 655 580", new: "700 350 950 700",
          boxes: [[[300, 470, 655, 580], [700, 350, 950, 700]], [[320, 260, 695, 480], [50, 350, 280, 700]]] },
        { k: "Attribute", sub: "pens", img: "assets/sp_edit/A_attr.jpg", field: "elements[6].material",
          old: "blue ballpoint pen", new: "brass fountain pen", boxes: null },
        { k: "Scene", sub: "rainy neon", img: "assets/sp_edit/A_scene.jpg", field: "scene.setting",
          old: "evening study desk", new: "rainy night cafe, neon", boxes: null },
        { k: "Style", sub: "cyberpunk", img: "assets/sp_edit/A_style.jpg", field: "style",
          old: "warm photorealistic", new: "cyberpunk neon-noir", boxes: null },
      ],
    },
    B: {
      label: "Case B · café", base: "assets/sp_edit/B_base.jpg",
      edits: [
        { k: "Move", sub: "raise pendants", img: "assets/sp_edit/B_move.jpg", field: "elements[10,11].position",
          old: "563 0 674 141", new: "200 0 350 200",
          boxes: [[[563, 0, 674, 141], [200, 0, 350, 200]], [[680, 0, 791, 216], [650, 0, 800, 200]]] },
        { k: "Attribute", sub: "roses", img: "assets/sp_edit/B_attr.jpg", field: "elements[3].material",
          old: "dried grass bundle", new: "fresh red roses", boxes: null },
        { k: "Scene", sub: "snowy cabin", img: "assets/sp_edit/B_scene.jpg", field: "scene.setting",
          old: "late-night cafe", new: "snowy winter cabin", boxes: null },
        { k: "Style", sub: "ink-wash", img: "assets/sp_edit/B_style.jpg", field: "style",
          old: "cinematic photoreal", new: "Chinese ink-wash", boxes: null },
      ],
    },
  };
  function spEditWidget(root) {
    const DGRN = "#1a7f37", DRED = "#cf222e";
    let curCase = "A", curEdit = 0;
    const mk = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };

    root.innerHTML = "";
    // top bar — case toggle only (descriptive title removed)
    const top = mk("div", "spe-top");
    const cases = mk("div", "spe-cases");
    const caseBtns = {};
    Object.keys(SPEDIT).forEach(k => {
      const b = mk("button", "spe-case"); b.textContent = SPEDIT[k].label;
      b.addEventListener("click", () => { curCase = k; curEdit = 0; render(); });
      cases.appendChild(b); caseBtns[k] = b;
    });
    top.appendChild(cases);

    // stage: baseline → edited
    const stage = mk("div", "spe-stage");
    const lside = mk("div", "spe-side");
    const llab = mk("div", "spe-lab"); llab.textContent = "Baseline — full SP";
    const lwrap = mk("div", "spe-imgwrap");
    const lImg = mk("img", "spe-img"); lImg.alt = "baseline generation";
    lwrap.appendChild(lImg); lside.appendChild(llab); lside.appendChild(lwrap);
    const arr = mk("div", "spe-arrow"); arr.innerHTML = "&rarr;";
    const rside = mk("div", "spe-side");
    const rlab = mk("div", "spe-lab edit");
    const rwrap = mk("div", "spe-imgwrap");
    const rImg = mk("img", "spe-img"); rImg.alt = "edited generation";
    const overlay = el("svg", { class: "spe-overlay", viewBox: "0 0 1000 1000", preserveAspectRatio: "none" });
    rwrap.appendChild(rImg); rwrap.appendChild(overlay);
    const diff = mk("div", "spe-diff");
    rside.appendChild(rlab); rside.appendChild(rwrap); rside.appendChild(diff);
    stage.appendChild(lside); stage.appendChild(arr); stage.appendChild(rside);

    // edit chips + note
    const chips = mk("div", "spe-chips");
    const chipEls = [];
    for (let i = 0; i < 4; i++) {
      const c = mk("button", "spe-chip");
      c.addEventListener("click", () => { curEdit = i; render(); });
      chips.appendChild(c); chipEls.push(c);
    }
    const note = mk("p", "spe-note");
    root.appendChild(top); root.appendChild(stage); root.appendChild(chips); root.appendChild(note);

    function render() {
      const cd = SPEDIT[curCase], ed = cd.edits[curEdit];
      Object.keys(caseBtns).forEach(k => caseBtns[k].classList.toggle("on", k === curCase));
      lImg.src = cd.base;
      rImg.src = ed.img;
      rlab.innerHTML = `${ed.k} <span>· ${ed.sub}</span>`;
      cd.edits.forEach((e, i) => {
        chipEls[i].innerHTML = `${e.k}${e.boxes ? ' <span class="bb">bbox</span>' : ''}`;
        chipEls[i].classList.toggle("on", i === curEdit);
      });
      diff.innerHTML =
        `<span class="f">${ed.field}</span><span class="d rem">&minus; ${ed.old}</span><span class="d add">+ ${ed.new}</span>`;
      while (overlay.firstChild) overlay.removeChild(overlay.firstChild);
      if (ed.boxes) {
        overlay.style.display = "block";
        ed.boxes.forEach(([o, n]) => {
          overlay.appendChild(el("rect", { x: o[0], y: o[1], width: o[2] - o[0], height: o[3] - o[1],
            fill: "none", stroke: DRED, "stroke-width": 4, "stroke-dasharray": "11 8" }));
          overlay.appendChild(el("rect", { x: n[0], y: n[1], width: n[2] - n[0], height: n[3] - n[1],
            fill: "none", stroke: DGRN, "stroke-width": 5 }));
          const oc = [(o[0] + o[2]) / 2, (o[1] + o[3]) / 2], nc = [(n[0] + n[2]) / 2, (n[1] + n[3]) / 2];
          const a = arrow(oc[0], oc[1], nc[0], nc[1], DGRN, { w: 3, head: 26 });
          a.setAttribute("opacity", "0.9");
          overlay.appendChild(a);
        });
        note.innerHTML = '<span class="lg rem">dashed red</span> = old bbox &middot; ' +
          '<span class="lg add">green</span> = new bbox · one field changes; the rest of the composition is preserved.';
      } else {
        overlay.style.display = "none";
        note.innerHTML = 'One structured-prompt field changes; the diffuser re-renders exactly that aspect and preserves the rest.';
      }
    }
    render();
  }

  function init() {
    const s = document.getElementById("chart-scaling");
    if (s) scalingChart(s);
    const p = document.getElementById("chart-prompter");
    if (p) prompterChart(p);
    const r = document.getElementById("recon");
    if (r) reconWidget(r);
    const g = document.getElementById("gpg-viz");
    if (g) gpgViz(g);
    const e = document.getElementById("ed-viz");
    if (e) edViz(e);
    const sv = document.getElementById("schema-viz");
    if (sv) schemaViz(sv);
    const fsft = document.getElementById("sft-compare");
    if (fsft) sftCompare(fsft);
    const fp = document.getElementById("fig-pipeline");
    if (fp) trainPipeline(fp);
    const spe = document.getElementById("sp-edit");
    if (spe) spEditWidget(spe);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
