/* ============================================================
   Shared hand-rolled SVG helpers (no deps).
   Extracted from charts.js so the blog explainer (blog.js) can
   reuse the exact same drawing primitives and tooltip behaviour
   as the project page, in the typed-paper aesthetic.
   Exposed as window.SVGH. Pure helpers only — no page-specific
   data or widgets live here.
   ============================================================ */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";

  // palette (matches styles.css :root and charts.js)
  const RED = "#b2182b", BLUE = "#2166ac", GREEN = "#2ca02c", PURPLE = "#9467bd";
  const NAVY = "#16324f", INK = "#1a1a1a", AXIS = "#3a342a", GRID = "#cfc7b4";
  const EMERALD = "#4a7a4f", AMBER = "#b8862a";
  const PAPER = "#faf6ec", RULE = "#d6cfb9", SOFT = "#8c8473", MUTED = "#5e574a";
  const SERIF = "'EB Garamond', Georgia, serif";
  const MONO = "'JetBrains Mono', monospace";

  // ---- element builders ----
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
  function makeSVG(W, H) {
    return el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", preserveAspectRatio: "xMidYMid meet" });
  }
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

  // ---- shared tooltip (single instance shared across all widgets) ----
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

  // ---- parsing / math utilities ----
  function parseBbox(s) {
    const m = /<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*<\/bbox>/.exec(s || "");
    return m ? [+m[1], +m[2], +m[3], +m[4]] : null;
  }
  function sbxParse(s) {
    const m = /(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)/.exec(s || "");
    return m ? [+m[1] / 999, +m[2] / 999, +m[3] / 999, +m[4] / 999] : null;
  }
  function humanize(k) { return k.replace(/_/g, " "); }
  function fieldVal(v) {
    if (v && typeof v === "object") return Object.entries(v).map(([k, x]) => `${humanize(k)} — ${x}`).join("; ");
    return String(v);
  }
  function f05(p, r) { return p + r === 0 ? 0 : 1.25 * p * r / (0.25 * p + r); }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function lerp(a, b, u) { return a + (b - a) * u; }
  function easeInOut(u) { return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; }

  // requestAnimationFrame tween helper (Date.now-free): step(0..1) → done()
  function tween(ms, step, done) {
    let start = null;
    function frame(now) {
      if (start === null) start = now;
      const u = clamp((now - start) / ms, 0, 1);
      step(easeInOut(u), u);
      if (u < 1) requestAnimationFrame(frame);
      else if (done) done();
    }
    requestAnimationFrame(frame);
  }

  window.SVGH = {
    SVGNS, RED, BLUE, GREEN, PURPLE, NAVY, INK, AXIS, GRID, EMERALD, AMBER,
    PAPER, RULE, SOFT, MUTED, SERIF, MONO,
    el, txt, marker, makeSVG, rrect, arrow, fbox,
    ensureTip, showTip, moveTip, hideTip,
    parseBbox, sbxParse, humanize, fieldVal, f05, clamp, lerp, easeInOut, tween,
  };
})();
