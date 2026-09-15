/* Dashboard controller: form state in, plots and readouts out. */

import { Plot } from '/static/plot.js';

const css = getComputedStyle(document.documentElement);
const COLOR = {
  measured: css.getPropertyValue('--measured').trim(),
  reference: css.getPropertyValue('--reference').trim(),
  third: css.getPropertyValue('--third').trim(),
  fourth: css.getPropertyValue('--fourth').trim(),
  muted: css.getPropertyValue('--muted').trim(),
};
const SERIES_COLORS = [COLOR.measured, COLOR.reference, COLOR.third, COLOR.fourth, '#6b21a8'];

const $ = (id) => document.getElementById(id);
const plots = {};
let meta = null;

/* ------------------------------------------------------------------ status */
let busy = 0;
function setStatus(text, state) {
  const el = $('status');
  el.className = 'status' + (state ? ' ' + state : '');
  $('statusText').textContent = text;
}

async function call(path, body) {
  busy++;
  setStatus('computing', 'busy');
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* not json */ }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    const out = await res.json();
    if (--busy === 0) setStatus('ready', '');
    return out;
  } catch (err) {
    busy = Math.max(0, busy - 1);
    setStatus(err.message, 'error');
    throw err;
  }
}

/* ---------------------------------------------------------------- helpers */
const fmt = (v, digits = 4) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '\u2014';
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(digits - 1);
  return Number(v.toPrecision(digits)).toString();
};

function readout(el, cells) {
  el.innerHTML = cells
    .map((c) => `<div class="cell"><span class="k">${c.k}</span><span class="v ${c.cls || ''}">${c.v}</span></div>`)
    .join('');
}

function table(el, columns, rows) {
  const head = `<thead><tr>${columns.map((c) => `<th>${c.h}</th>`).join('')}</tr></thead>`;
  const body = rows.map((r) => `<tr>${columns
    .map((c) => {
      const cell = c.f(r);
      const value = typeof cell === 'object' ? cell.v : cell;
      const cls = typeof cell === 'object' ? cell.cls : '';
      return `<td class="${cls || ''}">${value}</td>`;
    })
    .join('')}</tr>`).join('');
  el.innerHTML = head + `<tbody>${body}</tbody>`;
}

function plot(id, opts) {
  if (!plots[id]) {
    plots[id] = new Plot($(id), opts);
    plots[id].draw();   // the constructor only stores options
  } else {
    plots[id].update(opts);
  }
  return plots[id];
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/* -------------------------------------------------------------------- BER */
async function runBER() {
  const btn = $('berRun');
  btn.disabled = true;
  try {
    const data = await call('/api/phy/ber', {
      scheme: $('berScheme').value,
      channel: $('berChannel').value,
      ebn0_start: +$('berStart').value,
      ebn0_stop: +$('berStop').value,
      ebn0_step: +$('berStep').value,
      target_errors: +$('berErrors').value,
      max_symbols: +$('berMax').value,
    });
    const pts = data.sweep.points;
    plot('berCanvas', {
      xLabel: 'Eb/N0 (dB)',
      yLabel: 'bit error rate',
      yScale: 'log',
      legend: 'bottom-left',
      series: [
        { name: 'closed form', color: COLOR.reference,
          x: data.theory_curve.ebn0_db, y: data.theory_curve.ber },
        { name: 'simulated, 95% CI', color: COLOR.measured, kind: 'points',
          x: pts.map((p) => p.ebn0_db), y: pts.map((p) => p.ber),
          ylow: pts.map((p) => p.ci_low), yhigh: pts.map((p) => p.ci_high) },
      ],
    });

    const worst = data.sweep.worst_relative_deviation;
    const inCI = pts.filter((p) => p.theory_within_ci).length;
    const bits = pts.reduce((a, p) => a + p.n_bits, 0);
    const secs = pts.reduce((a, p) => a + p.elapsed_s, 0);
    readout($('berReadout'), [
      { k: 'Worst gap to theory', v: Number.isFinite(worst) ? (100 * worst).toFixed(1) + ' %' : '\u2014',
        cls: worst < 0.2 ? 'pass' : '' },
      { k: 'Theory inside 95% CI', v: `${inCI} of ${pts.length}`, cls: inCI === pts.length ? 'pass' : 'fail' },
      { k: 'Implementation loss', v: fmt(data.sweep.implementation_loss_db, 3) + ' dB' },
      { k: 'Bits simulated', v: bits.toLocaleString() },
      { k: 'Bits per second', v: Math.round(bits / Math.max(secs, 1e-9)).toLocaleString() },
      { k: 'Seed', v: data.sweep.seed },
    ]);

    table($('berTable'), [
      { h: 'Eb/N0 dB', f: (p) => p.ebn0_db.toFixed(1) },
      { h: 'simulated', f: (p) => (p.n_bit_errors === 0
        ? '< ' + fmt(p.ci_high, 2) : fmt(p.ber, 3)) },
      { h: 'closed form', f: (p) => fmt(p.theory_ber, 3) },
      { h: 'deviation', f: (p) => (p.n_bit_errors === 0 || !Number.isFinite(p.rel_deviation)
        ? 'no errors seen'
        : (p.rel_deviation >= 0 ? '+' : '') + (100 * p.rel_deviation).toFixed(1) + ' %') },
      { h: 'bit errors', f: (p) => p.n_bit_errors.toLocaleString() },
      { h: 'bits', f: (p) => p.n_bits.toLocaleString() },
      { h: 'theory in CI', f: (p) => ({ v: p.theory_within_ci ? 'yes' : 'no',
        cls: p.theory_within_ci ? 'pass' : 'fail' }) },
    ], pts);

    const m = data.modulation;
    $('berNote').textContent =
      `${m.bits_per_symbol} bits per symbol, minimum distance ${fmt(m.min_distance, 3)} at unit average energy.`;
  } catch (e) { /* status bar already shows it */ }
  btn.disabled = false;
}

/* ------------------------------------------------------------------ solve */
async function runSolve() {
  const btn = $('solveRun');
  btn.disabled = true;
  try {
    const data = await call('/api/ode/solve', {
      problem: $('solveProblem').value,
      solver: $('solveSolver').value,
      steps: +$('solveSteps').value,
      rtol: +$('solveRtol').value,
      atol: +$('solveRtol').value * 1e-3,
    });

    const labels = data.problem.labels;
    const series = [];
    data.y.forEach((comp, i) => {
      if (i >= 4) return;
      series.push({ name: labels[i] + ' computed', color: SERIES_COLORS[i % SERIES_COLORS.length],
        x: data.t, y: comp });
    });
    if (data.exact) {
      data.exact.forEach((comp, i) => {
        if (i >= 4) return;
        series.push({ name: labels[i] + ' exact', color: SERIES_COLORS[i % SERIES_COLORS.length],
          x: data.t, y: comp, dash: [5, 4], width: 1.1 });
      });
    }
    plot('solveCanvas', { xLabel: 'time', yLabel: 'state', series });

    if (data.rel_error) {
      const peak = Math.max(...data.rel_error.flat().filter((v) => v > 0), 1e-16);
      const floor = Math.pow(10, Math.floor(Math.log10(peak)) - 8);
      plot('errCanvas', {
        xLabel: 'time', yLabel: 'relative error', yScale: 'log',
        yDomain: [floor, peak],
        series: data.rel_error.slice(0, 4).map((comp, i) => ({
          name: labels[i], color: SERIES_COLORS[i % SERIES_COLORS.length],
          x: data.t, y: comp.map((v) => (v > floor ? v : NaN)), width: 1.2,
        })),
      });
    } else {
      plot('errCanvas', { series: [], empty: 'No closed-form solution for this problem' });
    }

    const a = data.accuracy;
    const cells = [
      { k: 'Steps', v: data.stats.n_steps.toLocaleString() },
      { k: 'Rejected', v: data.stats.n_rejected.toLocaleString() },
      { k: 'Right-hand-side evals', v: data.stats.n_rhs.toLocaleString() },
      { k: 'LU factorisations', v: data.stats.n_lu.toLocaleString() },
      { k: 'Wall clock', v: (data.stats.elapsed_s * 1e3).toFixed(1) + ' ms' },
    ];
    if (a) {
      cells.unshift({ k: 'Max relative error', v: fmt(a.max_rel_error_pct, 3) + ' %',
        cls: a.passes_001pct ? 'pass' : 'fail' });
      cells.push({ k: 'Within 0.01 %', v: a.passes_001pct ? 'yes' : 'no',
        cls: a.passes_001pct ? 'pass' : 'fail' });
    }
    if (data.conservation_drift !== null && data.conservation_drift !== undefined) {
      cells.push({ k: 'Invariant drift', v: fmt(data.conservation_drift, 3) });
    }
    readout($('solveReadout'), cells);

    const s = data.solver;
    $('solveNote').textContent =
      `${s.name}: order ${s.order}, ${s.stability}.`
      + (data.success ? '' : ' Integration failed: ' + data.message);
    $('solveSteps').disabled = s.adaptive;
    $('solveRtol').disabled = !s.adaptive;
  } catch (e) { /* status bar */ }
  btn.disabled = false;
}

/* ------------------------------------------------------------------ order */
async function runOrder() {
  const btn = $('orderRun');
  btn.disabled = true;
  const chosen = [...document.querySelectorAll('#orderMethods input:checked')].map((i) => i.value);
  if (!chosen.length) { setStatus('pick at least one method', 'error'); btn.disabled = false; return; }
  try {
    const data = await call('/api/ode/convergence', {
      problem: $('orderProblem').value,
      solvers: chosen,
      counts: [50, 100, 200, 400, 800, 1600],
    });
    plot('orderCanvas', {
      xLabel: 'step size h', yLabel: 'relative error',
      xScale: 'log', yScale: 'log', legend: 'top-left',
      series: data.studies.map((s, i) => ({
        name: `${s.method} (p = ${s.observed_order.toFixed(2)})`,
        color: SERIES_COLORS[i % SERIES_COLORS.length],
        x: s.step_sizes, y: s.errors, kind: 'line',
      })).concat(data.studies.map((s, i) => ({
        color: SERIES_COLORS[i % SERIES_COLORS.length],
        x: s.step_sizes, y: s.errors, kind: 'points', radius: 3,
      }))),
    });

    const worst = Math.max(...data.studies.map((s) => Math.abs(s.order_error_pct)).filter(Number.isFinite));
    readout($('orderReadout'), [
      { k: 'Methods measured', v: data.studies.length },
      { k: 'Largest deviation from theory', v: fmt(worst, 3) + ' %',
        cls: worst < 10 ? 'pass' : 'fail' },
      { k: 'Problem', v: data.problem.name },
    ]);
    table($('orderTable'), [
      { h: 'method', f: (s) => s.method },
      { h: 'theoretical p', f: (s) => s.theoretical_order },
      { h: 'measured p', f: (s) => s.observed_order.toFixed(3) },
      { h: 'deviation', f: (s) => ({ v: s.order_error_pct.toFixed(1) + ' %',
        cls: s.order_error_pct < 10 ? 'pass' : 'fail' }) },
      { h: 'points fitted', f: (s) => s.n_fitted },
      { h: 'finest error', f: (s) => fmt(s.errors[s.errors.length - 1], 3) },
      { h: 'rhs evals', f: (s) => s.rhs_evals[s.rhs_evals.length - 1].toLocaleString() },
    ], data.studies);
  } catch (e) { /* status bar */ }
  btn.disabled = false;
}

/* ---------------------------------------------------------- constellation */
async function runIQ() {
  try {
    const data = await call('/api/phy/constellation', {
      scheme: $('iqScheme').value,
      channel: $('iqChannel').value,
      ebn0_db: +$('iqSnr').value,
      n_symbols: +$('iqCount').value,
    });
    const lim = Math.max(2, ...data.received.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])])) * 1.02;
    plot('iqCanvas', {
      xLabel: 'in phase', yLabel: 'quadrature',
      xDomain: [-lim, lim], yDomain: [-lim, lim],
      series: [
        { name: 'received', color: COLOR.measured, kind: 'points', radius: 1.3,
          x: data.received.map((p) => p[0]), y: data.received.map((p) => p[1]) },
        { name: 'transmitted', color: COLOR.reference, kind: 'points', radius: 6,
          shape: 'cross', x: data.ideal.map((p) => p[0]), y: data.ideal.map((p) => p[1]) },
      ],
    });
    readout($('iqReadout'), [
      { k: 'Eb/N0', v: data.ebn0_db.toFixed(1) + ' dB' },
      { k: 'Error vector magnitude', v: data.evm_pct.toFixed(2) + ' %' },
      { k: 'SNR implied by EVM', v: data.evm_snr_db.toFixed(2) + ' dB' },
      { k: 'Noise variance N0', v: fmt(data.noise_var, 3) },
      { k: 'Symbols', v: data.received.length.toLocaleString() },
    ]);
  } catch (e) { /* status bar */ }
}

/* ------------------------------------------------------------- attenuation */
async function runPath() {
  try {
    const freq = +$('pathFreq').value;
    const data = await call('/api/phy/pathloss', {
      frequency_hz: freq,
      d_min_m: 1, d_max_m: 5000, n_points: 400,
      exponent: +$('pathExp').value,
      h_tx_m: +$('pathHt').value, h_rx_m: +$('pathHr').value,
    });
    plot('pathCanvas', {
      xLabel: 'distance (m)', yLabel: 'path loss (dB)',
      xScale: 'log', legend: 'top-left',
      markers: [{ x: data.breakpoint_m, label: 'two-ray breakpoint', color: COLOR.muted }],
      series: [
        { name: 'free space', color: COLOR.reference, x: data.distance_m, y: data.free_space_db, dash: [6, 4] },
        { name: 'log distance', color: COLOR.measured, x: data.distance_m, y: data.log_distance_db },
        { name: 'two ray', color: COLOR.third, x: data.distance_m, y: data.two_ray_db, width: 1.2 },
      ],
    });
    await runBudget();
  } catch (e) { /* status bar */ }
}

async function runBudget() {
  try {
    const data = await call('/api/phy/link_budget', {
      distance_m: +$('budDist').value,
      frequency_hz: +$('pathFreq').value,
      scheme: $('budScheme').value,
      target_ber: +$('budBer').value,
      tx_power_dbm: +$('budPow').value,
      bandwidth_hz: +$('budBw').value * 1e6,
      noise_figure_db: +$('budNf').value,
      model: 'log_distance',
      exponent: +$('pathExp').value,
    });
    readout($('budReadout'), [
      { k: 'Path loss', v: data.path_loss_db.toFixed(1) + ' dB' },
      { k: 'Received power', v: data.rx_power_dbm.toFixed(1) + ' dBm' },
      { k: 'Noise floor', v: data.noise_power_dbm.toFixed(1) + ' dBm' },
      { k: 'SNR', v: data.snr_db.toFixed(1) + ' dB' },
      { k: 'Required SNR', v: data.required_snr_db.toFixed(1) + ' dB' },
      { k: 'Margin', v: (data.margin_db >= 0 ? '+' : '') + data.margin_db.toFixed(1) + ' dB',
        cls: data.closes ? 'pass' : 'fail' },
      { k: 'Raw throughput', v: data.throughput_mbps.toFixed(0) + ' Mb/s' },
    ]);
  } catch (e) { /* status bar */ }
}

/* ------------------------------------------------------------------- init */
function fillSelect(el, items, selected) {
  el.innerHTML = items
    .map((i) => `<option value="${i.value}"${i.value === selected ? ' selected' : ''}>${i.label}</option>`)
    .join('');
}

async function init() {
  meta = await (await fetch('/api/meta')).json();
  $('verText').textContent = 'physim ' + meta.version;

  const problemOptions = meta.problems.map((p) => ({
    value: p.name,
    label: p.name.replace(/_/g, ' ') + (p.stiff ? '  (stiff)' : '') + (p.has_exact ? '' : '  (no closed form)'),
  }));
  fillSelect($('solveProblem'), problemOptions, 'damped_oscillator');
  fillSelect($('orderProblem'),
    problemOptions.filter((p) => meta.problems.find((m) => m.name === p.value).has_exact),
    'damped_oscillator');
  fillSelect($('solveSolver'), meta.solvers.map((s) => ({
    value: s.name,
    label: s.name.replace(/_/g, ' ') + '  (order ' + s.order + (s.adaptive ? ', adaptive' : '') + ')',
  })), 'rk45');

  const fixed = meta.solvers.filter((s) => !s.adaptive);
  $('orderMethods').innerHTML = '<legend>Methods</legend>' + fixed.map((s) => `
    <label><input type="checkbox" value="${s.name}"
      ${['heun', 'rk4', 'radau_iia5'].includes(s.name) ? 'checked' : ''}>
      ${s.name.replace(/_/g, ' ')} <span class="hint">p = ${s.order}</span></label>`).join('');

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
      tab.classList.add('active');
      $('view-' + tab.dataset.view).classList.add('active');
      const first = { solve: runSolve, order: runOrder, iq: runIQ, path: runPath }[tab.dataset.view];
      if (first && !tab.dataset.loaded) { tab.dataset.loaded = '1'; first(); }
      Object.values(plots).forEach((p) => p.draw());
    });
  });

  $('berRun').addEventListener('click', runBER);
  $('solveRun').addEventListener('click', runSolve);
  $('orderRun').addEventListener('click', runOrder);
  $('solveProblem').addEventListener('change', runSolve);
  $('solveSolver').addEventListener('change', runSolve);

  const iqDebounced = debounce(runIQ, 180);
  ['iqScheme', 'iqChannel', 'iqCount'].forEach((id) => $(id).addEventListener('change', runIQ));
  $('iqSnr').addEventListener('input', () => { $('iqSnrOut').textContent = (+$('iqSnr').value).toFixed(1); iqDebounced(); });

  const pathDebounced = debounce(runPath, 180);
  ['pathFreq', 'pathHt', 'pathHr'].forEach((id) => $(id).addEventListener('change', runPath));
  $('pathExp').addEventListener('input', () => { $('pathExpOut').textContent = (+$('pathExp').value).toFixed(1); pathDebounced(); });
  ['budDist', 'budPow', 'budBw', 'budNf', 'budScheme', 'budBer'].forEach((id) =>
    $(id).addEventListener('change', runBudget));

  window.addEventListener('resize', debounce(() => Object.values(plots).forEach((p) => p.draw()), 120));

  document.querySelector('.tab[data-view="ber"]').dataset.loaded = '1';
  runBER();
}

init().catch((e) => setStatus('cannot reach the server: ' + e.message, 'error'));
