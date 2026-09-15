/* Minimal canvas plotting for the dashboard.
 *
 * Written rather than imported: the whole page must work offline from a
 * `physim serve` on a laptop with no network, and a charting library would
 * be a CDN dependency for two plot types. Supports linear and log axes,
 * line/point/step series, vertical error bars, reference markers, and a
 * crosshair readout.
 */

const FONT = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif';
const MONO = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

function niceTicks(lo, hi, target = 6) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

function decadeTicks(lo, hi) {
  const major = [];
  const minor = [];
  const start = Math.floor(Math.log10(lo));
  const stop = Math.ceil(Math.log10(hi));
  for (let e = start; e <= stop; e++) {
    const base = Math.pow(10, e);
    if (base >= lo * 0.999 && base <= hi * 1.001) major.push(base);
    for (let m = 2; m <= 9; m++) {
      const v = base * m;
      if (v >= lo && v <= hi) minor.push(v);
    }
  }
  return { major, minor };
}

function fmtTick(v, scale) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (scale === 'log' || a >= 1e5 || a < 1e-3) {
    const e = Math.round(Math.log10(a));
    if (scale === 'log' && Math.abs(a - Math.pow(10, e)) < a * 1e-6) {
      return e === 0 ? '1' : `1e${e}`;
    }
    return v.toExponential(0);
  }
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(4)));
}

export class Plot {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = Object.assign(
      { xLabel: '', yLabel: '', xScale: 'linear', yScale: 'linear',
        series: [], markers: [], pad: { l: 64, r: 16, t: 14, b: 42 } },
      opts
    );
    this.hover = null;
    canvas.addEventListener('mousemove', (e) => this._onMove(e));
    canvas.addEventListener('mouseleave', () => { this.hover = null; this.draw(); });
  }

  update(opts) {
    Object.assign(this.opts, opts);
    this.draw();
  }

  _visible() {
    return this.opts.series.filter((s) => s.x && s.x.length && s.visible !== false);
  }

  _domains() {
    const xs = [];
    const ys = [];
    for (const s of this._visible()) {
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i];
        const yv = s.y[i];
        if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
        if (this.opts.xScale === 'log' && xv <= 0) continue;
        if (this.opts.yScale === 'log' && yv <= 0) continue;
        xs.push(xv);
        ys.push(yv);
        if (s.ylow) ys.push(s.ylow[i]);
        if (s.yhigh) ys.push(s.yhigh[i]);
      }
    }
    if (!xs.length) return null;
    let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    if (this.opts.xDomain) [x0, x1] = this.opts.xDomain;
    if (this.opts.yDomain) [y0, y1] = this.opts.yDomain;

    if (this.opts.yScale === 'log') {
      y0 = Math.pow(10, Math.floor(Math.log10(Math.max(y0, 1e-300))));
      y1 = Math.pow(10, Math.ceil(Math.log10(Math.max(y1, y0 * 10))));
    } else if (y1 - y0 < 1e-15) {
      const c = (y0 + y1) / 2 || 1;
      y0 = c - Math.abs(c) * 0.1 - 1e-9;
      y1 = c + Math.abs(c) * 0.1 + 1e-9;
    } else {
      const m = (y1 - y0) * 0.06;
      y0 -= m;
      y1 += m;
    }
    if (x1 - x0 < 1e-15) { x0 -= 1; x1 += 1; }
    return { x0, x1, y0, y1 };
  }

  _project(d, rect) {
    const { xScale, yScale } = this.opts;
    const fx = (v) => {
      const t = xScale === 'log'
        ? (Math.log10(v) - Math.log10(d.x0)) / (Math.log10(d.x1) - Math.log10(d.x0))
        : (v - d.x0) / (d.x1 - d.x0);
      return rect.l + t * rect.w;
    };
    const fy = (v) => {
      const t = yScale === 'log'
        ? (Math.log10(v) - Math.log10(d.y0)) / (Math.log10(d.y1) - Math.log10(d.y0))
        : (v - d.y0) / (d.y1 - d.y0);
      return rect.t + (1 - t) * rect.h;
    };
    return { fx, fy };
  }

  draw() {
    const { canvas, ctx, opts } = this;
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth;
    const chh = canvas.clientHeight;
    if (canvas.width !== cw * dpr || canvas.height !== chh * dpr) {
      canvas.width = cw * dpr;
      canvas.height = chh * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, chh);

    const css = getComputedStyle(document.documentElement);
    const inkC = css.getPropertyValue('--ink').trim() || '#12171c';
    const mutedC = css.getPropertyValue('--muted').trim() || '#5c6b78';
    const ruleC = css.getPropertyValue('--rule').trim() || '#c9d3db';
    const gridC = css.getPropertyValue('--grid').trim() || '#e3e9ee';

    const p = opts.pad;
    const rect = { l: p.l, t: p.t, w: cw - p.l - p.r, h: chh - p.t - p.b };
    if (rect.w <= 10 || rect.h <= 10) return;

    const d = this._domains();
    if (!d) {
      ctx.fillStyle = mutedC;
      ctx.font = FONT;
      ctx.textAlign = 'center';
      ctx.fillText(opts.empty || 'No data yet', cw / 2, chh / 2);
      return;
    }
    const { fx, fy } = this._project(d, rect);
    this._last = { d, rect, fx, fy };

    // grid
    ctx.save();
    ctx.lineWidth = 1;
    const xt = opts.xScale === 'log' ? decadeTicks(d.x0, d.x1).major : niceTicks(d.x0, d.x1, 7);
    let ytMajor;
    let ytMinor = [];
    if (opts.yScale === 'log') {
      const t = decadeTicks(d.y0, d.y1);
      ytMajor = t.major;
      ytMinor = t.minor;
    } else {
      ytMajor = niceTicks(d.y0, d.y1, 6);
    }
    ctx.strokeStyle = gridC;
    ctx.beginPath();
    for (const v of xt) { const x = Math.round(fx(v)) + 0.5; ctx.moveTo(x, rect.t); ctx.lineTo(x, rect.t + rect.h); }
    for (const v of ytMajor) { const y = Math.round(fy(v)) + 0.5; ctx.moveTo(rect.l, y); ctx.lineTo(rect.l + rect.w, y); }
    ctx.stroke();
    if (ytMinor.length) {
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      for (const v of ytMinor) { const y = Math.round(fy(v)) + 0.5; ctx.moveTo(rect.l, y); ctx.lineTo(rect.l + rect.w, y); }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    // reference markers (vertical lines with a label)
    for (const m of opts.markers || []) {
      if (!Number.isFinite(m.x)) continue;
      const x = fx(m.x);
      if (x < rect.l || x > rect.l + rect.w) continue;
      ctx.save();
      ctx.strokeStyle = m.color || mutedC;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, rect.t);
      ctx.lineTo(x, rect.t + rect.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = m.color || mutedC;
      ctx.font = FONT;
      ctx.textAlign = 'left';
      ctx.fillText(m.label || '', x + 5, rect.t + 12);
      ctx.restore();
    }

    // series
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.l, rect.t, rect.w, rect.h);
    ctx.clip();
    for (const s of this._visible()) {
      const kind = s.kind || 'line';
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = s.width || 1.75;
      ctx.setLineDash(s.dash || []);
      if (kind === 'line') {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < s.x.length; i++) {
          const xv = s.x[i];
          const yv = s.y[i];
          if (!Number.isFinite(xv) || !Number.isFinite(yv)) { started = false; continue; }
          if (opts.yScale === 'log' && yv <= 0) { started = false; continue; }
          const X = fx(xv);
          const Y = fy(yv);
          if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      } else {
        const r = s.radius || 3;
        for (let i = 0; i < s.x.length; i++) {
          const xv = s.x[i];
          const yv = s.y[i];
          if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
          if (opts.yScale === 'log' && yv <= 0) continue;
          const X = fx(xv);
          const Y = fy(yv);
          if (s.ylow && s.yhigh && s.ylow[i] > 0) {
            ctx.beginPath();
            ctx.moveTo(X, fy(s.ylow[i]));
            ctx.lineTo(X, fy(s.yhigh[i]));
            ctx.lineWidth = 1;
            ctx.stroke();
          }
          if (s.shape === 'cross') {
            // Drawn with a light halo so reference points stay legible on
            // top of a dense scatter cloud.
            ctx.save();
            ctx.lineWidth = 3.5;
            ctx.strokeStyle = 'rgba(255,255,255,0.85)';
            ctx.beginPath();
            ctx.moveTo(X - r, Y); ctx.lineTo(X + r, Y);
            ctx.moveTo(X, Y - r); ctx.lineTo(X, Y + r);
            ctx.stroke();
            ctx.lineWidth = 1.6;
            ctx.strokeStyle = s.color;
            ctx.stroke();
            ctx.restore();
          } else {
            ctx.beginPath();
            ctx.arc(X, Y, r, 0, Math.PI * 2);
            if (s.hollow) { ctx.lineWidth = 1.5; ctx.stroke(); } else ctx.fill();
          }
        }
      }
      ctx.setLineDash([]);
    }
    ctx.restore();

    // frame + ticks
    ctx.strokeStyle = ruleC;
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(rect.l) + 0.5, Math.round(rect.t) + 0.5, rect.w, rect.h);

    ctx.fillStyle = mutedC;
    ctx.font = MONO;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const v of xt) {
      const x = fx(v);
      if (x < rect.l - 1 || x > rect.l + rect.w + 1) continue;
      ctx.fillText(fmtTick(v, opts.xScale), x, rect.t + rect.h + 7);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const v of ytMajor) {
      const y = fy(v);
      if (y < rect.t - 1 || y > rect.t + rect.h + 1) continue;
      ctx.fillText(fmtTick(v, opts.yScale), rect.l - 8, y);
    }

    ctx.fillStyle = inkC;
    ctx.font = FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    if (opts.xLabel) ctx.fillText(opts.xLabel, rect.l + rect.w / 2, chh - 6);
    if (opts.yLabel) {
      ctx.save();
      ctx.translate(13, rect.t + rect.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(opts.yLabel, 0, 0);
      ctx.restore();
    }

    this._drawLegend(rect, inkC);
    if (this.hover) this._drawHover(rect, inkC, ruleC);
  }

  _drawLegend(rect, inkC) {
    const items = this._visible().filter((s) => s.name);
    if (!items.length) return;
    const ctx = this.ctx;
    ctx.font = FONT;
    const pad = 8;
    const lh = 17;
    const w = Math.max(...items.map((s) => ctx.measureText(s.name).width)) + 34;
    const h = items.length * lh + pad;
    const where = this.opts.legend || 'top-right';
    const x = where.includes('left') ? rect.l + 10 : rect.l + rect.w - w - 10;
    const y = where.includes('bottom') ? rect.t + rect.h - h - 10 : rect.t + 10;
    ctx.save();
    ctx.globalAlpha = 0.93;
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--panel').trim() || '#fff';
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--rule').trim() || '#c9d3db';
    ctx.strokeRect(Math.round(x) + 0.5, Math.round(y) + 0.5, w, h);
    items.forEach((s, i) => {
      const cy = y + pad / 2 + lh * i + lh / 2;
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = 2;
      if ((s.kind || 'line') === 'line') {
        ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        ctx.moveTo(x + 8, cy);
        ctx.lineTo(x + 24, cy);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (s.shape === 'cross') {
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(x + 11, cy); ctx.lineTo(x + 21, cy);
        ctx.moveTo(x + 16, cy - 5); ctx.lineTo(x + 16, cy + 5);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(x + 16, cy, 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = inkC;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.name, x + 30, cy);
    });
    ctx.restore();
  }

  _onMove(e) {
    if (!this._last) return;
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    const { rect } = this._last;
    if (mx < rect.l || mx > rect.l + rect.w || my < rect.t || my > rect.t + rect.h) {
      if (this.hover) { this.hover = null; this.draw(); }
      return;
    }
    this.hover = { mx, my };
    this.draw();
  }

  _drawHover(rect, inkC, ruleC) {
    const { fx, fy } = this._last;
    const { mx } = this.hover;
    const ctx = this.ctx;
    let best = null;
    for (const s of this._visible()) {
      for (let i = 0; i < s.x.length; i++) {
        const X = fx(s.x[i]);
        const dist = Math.abs(X - mx);
        if (Number.isFinite(X) && (!best || dist < best.dist)) {
          best = { dist, s, i, X, Y: fy(s.y[i]) };
        }
      }
    }
    if (!best || best.dist > 40) return;
    ctx.save();
    ctx.strokeStyle = ruleC;
    ctx.beginPath();
    ctx.moveTo(Math.round(best.X) + 0.5, rect.t);
    ctx.lineTo(Math.round(best.X) + 0.5, rect.t + rect.h);
    ctx.stroke();
    ctx.fillStyle = best.s.color;
    ctx.beginPath();
    ctx.arc(best.X, best.Y, 4, 0, Math.PI * 2);
    ctx.fill();

    const fmt = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)
      ? v.toExponential(3) : Number(v.toPrecision(5)).toString());
    const text = `${fmt(best.s.x[best.i])}, ${fmt(best.s.y[best.i])}`;
    ctx.font = MONO;
    const w = ctx.measureText(text).width + 14;
    let bx = best.X + 10;
    if (bx + w > rect.l + rect.w) bx = best.X - w - 10;
    const by = Math.min(Math.max(best.Y - 26, rect.t + 2), rect.t + rect.h - 24);
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--panel').trim() || '#fff';
    ctx.globalAlpha = 0.96;
    ctx.fillRect(bx, by, w, 22);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = ruleC;
    ctx.strokeRect(Math.round(bx) + 0.5, Math.round(by) + 0.5, w, 22);
    ctx.fillStyle = inkC;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + 7, by + 11);
    ctx.restore();
  }
}

export function scatterPlot(canvas, opts) {
  return new Plot(canvas, opts);
}
