// Small dependency-free canvas charts: line/step plots with log or linear axes, a crosshair
// tooltip, a moving cursor, and a tornado (horizontal bar) chart. Colors come from CSS tokens.

export function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

function withAlpha(color, a) {
  if (color.startsWith('#')) {
    const n = color.length === 4
      ? color.slice(1).split('').map((c) => parseInt(c + c, 16))
      : [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
    return `rgba(${n[0]},${n[1]},${n[2]},${a})`;
  }
  return color;
}
export { withAlpha };

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
export function pow10Label(k) {
  if (k >= 0 && k <= 3) return String(10 ** k);
  if (k === -1) return '0.1';
  if (k === -2) return '0.01';
  return '10' + String(k).split('').map((c) => SUP[c]).join('');
}

function niceStep(span, n) {
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const r = raw / mag;
  return (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
}

export function setupCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = height ?? canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export class Plot {
  constructor(container, opts = {}) {
    this.el = container;
    this.opts = {
      xLog: false, yLog: false, xLabel: '', yLabel: '', height: 230,
      pad: { l: 58, r: 14, t: 14, b: 42 }, xFmt: (v) => String(v), yFmt: (v) => String(v),
      xTickFmt: null, yTickFmt: null, xTicks: null, yTicks: null, ...opts,
    };
    this.canvas = document.createElement('canvas');
    this.canvas.style.height = `${this.opts.height}px`;
    this.canvas.setAttribute('role', 'img');
    if (opts.ariaLabel) this.canvas.setAttribute('aria-label', opts.ariaLabel);
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    container.classList.add('plot');
    container.append(this.canvas, this.tip);
    this.data = { series: [], markers: [], vlines: [], hlines: [], bands: [], xDomain: [0, 1], yDomain: [0, 1] };
    this.cursorX = null;
    this.hoverPx = null;
    new ResizeObserver(() => this.draw()).observe(container);
    this.canvas.addEventListener('pointermove', (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hoverPx = [e.clientX - r.left, e.clientY - r.top];
      this.draw();
    });
    this.canvas.addEventListener('pointerleave', () => { this.hoverPx = null; this.tip.hidden = true; this.draw(); });
  }

  setData(d) { Object.assign(this.data, d); this.draw(); }
  setCursor(x) { this.cursorX = x; this.draw(); }

  scales(w, h) {
    const { pad, xLog, yLog } = this.opts;
    const [x0, x1] = this.data.xDomain, [y0, y1] = this.data.yDomain;
    const fx = xLog ? (v) => Math.log10(v) : (v) => v;
    const fy = yLog ? (v) => Math.log10(v) : (v) => v;
    const X = (v) => pad.l + ((fx(v) - fx(x0)) / (fx(x1) - fx(x0))) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - ((fy(v) - fy(y0)) / (fy(y1) - fy(y0))) * (h - pad.t - pad.b);
    const Xinv = (px) => {
      const f = fx(x0) + ((px - pad.l) / (w - pad.l - pad.r)) * (fx(x1) - fx(x0));
      return xLog ? 10 ** f : f;
    };
    return { X, Y, Xinv };
  }

  ticks(axis) {
    const o = this.opts;
    const [a, b] = axis === 'x' ? this.data.xDomain : this.data.yDomain;
    const custom = axis === 'x' ? o.xTicks : o.yTicks;
    if (custom) return (typeof custom === 'function' ? custom(a, b) : custom).filter(([v]) => v >= a * 0.999999 && v <= b * 1.000001);
    const log = axis === 'x' ? o.xLog : o.yLog;
    const fmt = (axis === 'x' ? o.xTickFmt : o.yTickFmt) || null;
    const out = [];
    if (log) {
      const k0 = Math.ceil(Math.log10(a) - 1e-9), k1 = Math.floor(Math.log10(b) + 1e-9);
      const every = Math.max(1, Math.ceil((k1 - k0 + 1) / 7));
      for (let k = k0; k <= k1; k++) if ((k - k0) % every === 0) out.push([10 ** k, fmt ? fmt(10 ** k) : pow10Label(k)]);
    } else {
      const st = niceStep(b - a, axis === 'x' ? 6 : 5);
      for (let v = Math.ceil(a / st) * st; v <= b + st * 1e-9; v += st) {
        const vv = Math.abs(v) < st * 1e-9 ? 0 : v;
        out.push([vv, fmt ? fmt(vv) : String(+vv.toPrecision(6))]);
      }
    }
    return out;
  }

  draw() {
    const { ctx, w, h } = setupCanvas(this.canvas, this.opts.height);
    const o = this.opts, d = this.data, pad = o.pad;
    const ink = cssVar('--ink'), ink2 = cssVar('--ink-2'), muted = cssVar('--muted');
    const grid = cssVar('--grid'), axis = cssVar('--axis'), surface = cssVar('--surface');
    ctx.clearRect(0, 0, w, h);
    if (w < 50) return;
    const { X, Y, Xinv } = this.scales(w, h);
    const [x0, x1] = d.xDomain, [y0, y1] = d.yDomain;
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';

    // bands
    for (const bnd of d.bands) {
      ctx.fillStyle = withAlpha(cssVar(bnd.color || '--grid'), bnd.alpha ?? 0.5);
      const xa = X(Math.max(bnd.x0, x0)), xb = X(Math.min(bnd.x1, x1));
      ctx.fillRect(xa, pad.t, xb - xa, h - pad.t - pad.b);
      if (bnd.label) { ctx.fillStyle = ink2; ctx.textAlign = 'left'; ctx.fillText(bnd.label, xa + 4, pad.t + 12); }
    }
    // grid + ticks
    ctx.lineWidth = 1;
    ctx.textBaseline = 'middle';
    for (const [v, lab] of this.ticks('y')) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = ink2; ctx.textAlign = 'right'; ctx.fillText(lab, pad.l - 6, y);
    }
    ctx.textBaseline = 'top';
    let lastRight = -Infinity;
    for (const [v, lab] of this.ticks('x')) {
      const x = Math.round(X(v)) + 0.5;
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      const tw = ctx.measureText(lab).width;
      if (x - tw / 2 > lastRight + 6) {
        ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.fillText(lab, x, h - pad.b + 6);
        lastRight = x + tw / 2;
      }
    }
    // axes
    ctx.strokeStyle = axis;
    ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r, h - pad.b + 0.5); ctx.stroke();
    // axis labels
    ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(o.xLabel, pad.l + (w - pad.l - pad.r) / 2, h - 6);
    ctx.save(); ctx.translate(13, pad.t + (h - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(o.yLabel, 0, 0); ctx.restore();

    // clip plot area
    ctx.save();
    ctx.beginPath(); ctx.rect(pad.l, pad.t - 2, w - pad.l - pad.r, h - pad.t - pad.b + 4); ctx.clip();
    for (const hl of d.hlines) {
      ctx.strokeStyle = cssVar(hl.color || '--ink'); ctx.lineWidth = 1;
      const y = Math.round(Y(hl.y)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      if (hl.label) { ctx.fillStyle = ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(hl.label, pad.l + 4, y - 3); }
    }
    for (const vl of d.vlines) {
      if (vl.x < x0 || vl.x > x1) continue;
      ctx.strokeStyle = cssVar(vl.color || '--axis'); ctx.lineWidth = 1;
      const x = Math.round(X(vl.x)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      if (vl.label) { ctx.fillStyle = ink2; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(vl.label, x + 4, pad.t + 2); }
    }
    for (const s of d.series) this.drawSeries(ctx, s, X, Y);
    ctx.restore();

    // markers (dots with a surface ring)
    for (const m of d.markers) {
      if (!(m.x >= x0 && m.x <= x1 && m.y >= y0 && m.y <= y1)) continue;
      dot(ctx, X(m.x), Y(m.y), cssVar(m.color), surface, m.r || 4.5);
      if (m.label) {
        ctx.fillStyle = ink2; ctx.textBaseline = 'middle';
        ctx.textAlign = m.align || 'left';
        ctx.fillText(m.label, X(m.x) + (m.align === 'right' ? -9 : 9), Y(m.y) + (m.dy || 0));
      }
    }
    // cursor
    if (this.cursorX != null && this.cursorX >= x0 && this.cursorX <= x1) {
      const x = Math.round(X(this.cursorX)) + 0.5;
      ctx.strokeStyle = ink; ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.globalAlpha = 1;
      for (const s of d.series) {
        if (s.kind === 'step' || s.noCursor) continue;
        const v = interp(s, this.cursorX);
        if (v != null && v >= y0 && v <= y1) dot(ctx, x, Y(v), cssVar(s.color), surface, 4.5);
      }
    }
    // hover crosshair + tooltip
    if (this.hoverPx && this.hoverPx[0] >= pad.l && this.hoverPx[0] <= w - pad.r) {
      const xv = Xinv(this.hoverPx[0]);
      const x = Math.round(this.hoverPx[0]) + 0.5;
      ctx.strokeStyle = muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      const rows = [];
      for (const s of d.series) {
        if (s.noTip) continue;
        const v = s.kind === 'step' ? stepValue(s, xv) : interp(s, xv);
        if (v == null) continue;
        if (s.kind !== 'step' && v >= y0 && v <= y1) dot(ctx, x, Y(v), cssVar(s.color), surface, 4);
        rows.push(`<div class="tip-row"><span class="key" style="background:${cssVar(s.color)}"></span>${s.name}<b>${(s.fmt || o.yFmt)(v)}</b></div>`);
      }
      this.tip.innerHTML = `<div class="tip-head">${o.xFmt(xv)}</div>${rows.join('')}`;
      this.tip.hidden = false;
      const tw = this.tip.offsetWidth;
      const left = this.hoverPx[0] + 14 + tw > w ? this.hoverPx[0] - 14 - tw : this.hoverPx[0] + 14;
      this.tip.style.left = `${left}px`;
      this.tip.style.top = `${Math.max(0, this.hoverPx[1] - 20)}px`;
    } else if (this.hoverPx) {
      this.tip.hidden = true;
    }
  }

  drawSeries(ctx, s, X, Y) {
    const col = cssVar(s.color);
    const { xLog, yLog } = this.opts;
    const ok = (xv, yv) => Number.isFinite(xv) && Number.isFinite(yv) && (!xLog || xv > 0) && (!yLog || yv > 0);
    if (s.kind === 'step') {
      // histogram: s.x are bin edges (n+1), s.y bin heights (n)
      ctx.beginPath();
      ctx.moveTo(X(s.x[0]), Y(0));
      for (let i = 0; i < s.y.length; i++) { ctx.lineTo(X(s.x[i]), Y(s.y[i])); ctx.lineTo(X(s.x[i + 1]), Y(s.y[i])); }
      ctx.lineTo(X(s.x[s.x.length - 1]), Y(0));
      ctx.fillStyle = withAlpha(col, 0.12); ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
      return;
    }
    ctx.strokeStyle = col; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      if (!ok(s.x[i], s.y[i])) { pen = false; continue; }
      const px = X(s.x[i]), py = Y(s.y[i]);
      if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

export function dot(ctx, x, y, color, ring, r = 4.5) {
  ctx.beginPath(); ctx.arc(x, y, r + 2, 0, 2 * Math.PI); ctx.fillStyle = ring; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
}

function interp(s, xv) {
  const xs = s.x, ys = s.y, n = xs.length;
  if (!n || xv < xs[0] || xv > xs[n - 1]) return null;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= xv) lo = m; else hi = m; }
  if (xs[hi] === xs[lo]) return ys[lo];
  return ys[lo] + ((ys[hi] - ys[lo]) * (xv - xs[lo])) / (xs[hi] - xs[lo]);
}

function stepValue(s, xv) {
  const e = s.x;
  if (xv < e[0] || xv >= e[e.length - 1]) return null;
  let lo = 0, hi = e.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (e[m] <= xv) lo = m; else hi = m; }
  return s.y[lo];
}

/** Horizontal one-at-a-time sensitivity bars around a centre value. */
export class Tornado {
  constructor(container, opts = {}) {
    this.el = container;
    this.opts = { rowH: 34, xLabel: '', xFmt: (v) => v.toFixed(1), tickFmt: (v) => String(+v.toPrecision(6)), ...opts };
    this.canvas = document.createElement('canvas');
    this.canvas.setAttribute('role', 'img');
    if (opts.ariaLabel) this.canvas.setAttribute('aria-label', opts.ariaLabel);
    container.classList.add('plot');
    container.append(this.canvas);
    this.rows = [];
    new ResizeObserver(() => this.draw()).observe(container);
  }

  setRows(rows) { this.rows = rows; this.draw(); }

  draw() {
    const o = this.opts, rows = this.rows;
    const h = 34 + rows.length * o.rowH + 44;
    this.canvas.style.height = `${h}px`;
    const { ctx, w } = setupCanvas(this.canvas, h);
    ctx.clearRect(0, 0, w, h);
    if (!rows.length || w < 50) return;
    const ink = cssVar('--ink'), ink2 = cssVar('--ink-2'), grid = cssVar('--grid'), axis = cssVar('--axis');
    const lo = cssVar('--ramp-lo'), hi = cssVar('--ramp-hi');
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    const labelW = Math.max(...rows.map((r) => ctx.measureText(r.label).width));
    const padL = labelW + 14, padR = 8, inner = Math.min(78, (w - padL) * 0.2);
    const y0 = rows[0].y0;
    let span = 0;
    for (const r of rows) span = Math.max(span, Math.abs(r.yLo - y0), Math.abs(r.yHi - y0));
    span = span * 1.02 || 1;
    const X = (v) => padL + inner + ((v - (y0 - span)) / (2 * span)) * (w - padL - padR - 2 * inner);
    // grid
    const st = niceStep(2 * span, 5);
    ctx.textBaseline = 'top'; ctx.textAlign = 'center';
    const bottom = 30 + rows.length * o.rowH;
    let lastR = -Infinity;
    for (let v = Math.ceil((y0 - span) / st) * st; v <= y0 + span; v += st) {
      const x = Math.round(X(v)) + 0.5;
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(x, 24); ctx.lineTo(x, bottom); ctx.stroke();
      const lab = o.tickFmt(v), tw = ctx.measureText(lab).width;
      if (x - tw / 2 > lastR + 6) { ctx.fillStyle = ink2; ctx.fillText(lab, x, bottom + 6); lastR = x + tw / 2; }
    }
    ctx.strokeStyle = axis; ctx.beginPath(); ctx.moveTo(padL + inner, bottom + 0.5); ctx.lineTo(w - padR - inner, bottom + 0.5); ctx.stroke();
    ctx.fillStyle = ink2; ctx.textBaseline = 'alphabetic';
    ctx.fillText(o.xLabel, padL + (w - padL - padR) / 2, h - 6);
    rows.forEach((r, i) => {
      const yc = 30 + i * o.rowH + o.rowH / 2;
      ctx.fillStyle = ink; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(r.label, padL - 10, yc);
      for (const [yv, col, txt] of [[r.yLo, lo, r.loText], [r.yHi, hi, r.hiText]]) {
        const xa = X(Math.min(y0, yv)), xb = X(Math.max(y0, yv));
        if (xb - xa < 0.5) continue;
        ctx.fillStyle = col;
        roundRect(ctx, xa, yc - 9, xb - xa, 18, 3);
        ctx.fill();
        ctx.fillStyle = ink2;
        const right = yv > y0;
        ctx.textAlign = right ? 'left' : 'right';
        if (txt) ctx.fillText(txt, right ? xb + 4 : xa - 4, yc);
      }
      if (Math.abs(r.yHi - r.yLo) < span * 0.01) {
        ctx.fillStyle = ink2; ctx.textAlign = 'left'; ctx.fillText('≈ 0', X(y0) + 6, yc);
      }
    });
    const xc = Math.round(X(y0)) + 0.5;
    ctx.strokeStyle = ink; ctx.beginPath(); ctx.moveTo(xc, 22); ctx.lineTo(xc, bottom); ctx.stroke();
    ctx.fillStyle = ink2; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(`centre ${o.xFmt(y0)}`, xc, 16);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}
