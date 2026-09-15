# Architecture

## Layout

```
src/physim/
  core/        types.py     SolverResult, SolverStats
               linalg.py    finite-difference Jacobians, simplified Newton, LU
  solvers/     base.py      the ABC, fixed and adaptive drivers, PI controller
               explicit.py  Euler, Heun, RK4, RK23, RK45
               implicit.py  backward Euler, trapezoidal, TR-BDF2, Radau IIA
  problems/    base.py      ODEProblem
               library.py   eleven problems and their analytical solutions
  validation/  metrics.py   amplitude-normalised error, accuracy reports
               convergence.py  refinement studies, work-precision curves
               report.py    the full harness, JSON output
  phy/         modulation.py  constellations, Gray coding, slicers, LLRs
               channel.py     AWGN, fading, path loss, link budget
               theory.py      closed-form BER and SER
               link.py        Monte Carlo sweeps, Wilson intervals
               pulse.py       RRC/RC shaping, eye diagrams, PAPR
               impairments.py phase noise, CFO, IQ imbalance, quantisation
  experiments/ run_all.py   regenerates every figure and results file
  cli.py
api/           main.py      FastAPI endpoints
               static/      index.html, app.js, plot.js, style.css
```

The dependency direction is one-way: `core` knows nothing of `solvers`,
`solvers` know nothing of `validation`, and `phy` is independent of all of
them. The API and CLI are the only places that compose across subpackages.

## Solver design

`ODESolver` is an abstract base holding tolerances and the step-size
controller. A concrete method supplies a Butcher tableau and a `_step`
implementation; the base class owns everything else, so adding a method means
writing coefficients rather than another integration loop.

```python
class Ralston(ExplicitRK):
    name = "ralston"
    order = 2
    A = np.array([[0, 0], [2/3, 0]])
    B = np.array([1/4, 3/4])
    C = np.array([0, 2/3])
```

Register it in `solvers/__init__.py` and it appears in the CLI, the API, the
dashboard's method list and the convergence harness without further work.

Two drivers serve every method. `_drive_fixed` walks a uniform grid, clips the
last step to land exactly on `tf`, and aborts on divergence or a failed Newton
iteration. `_drive_adaptive` runs the PI controller, rejecting and retrying
steps; a Newton failure there is a rejection rather than an abort, because
shrinking the step usually restores convergence.

Every run returns a `SolverResult` carrying the trajectory, a `SolverStats`
work record (steps, rejections, right-hand-side evaluations, Jacobians, LU
factorisations, Newton iterations, wall clock) and an explicit success flag
with a message. Work counters are part of the result, not an afterthought:
comparing methods on accuracy alone is meaningless without the cost that
bought it.

## Problem design

`ODEProblem` is a dataclass bundling the right-hand side, initial condition,
time span, optional analytical Jacobian, optional exact solution, optional
conserved quantity, component labels and a parameter dictionary.

The parameter dictionary is the introspection surface. The transmission line
publishes `far_end_index`, `dc_far_end_v`, `explicit_h_limit` and
`explicit_steps_required`, so a test or a user can ask the problem where its
far end is and what step an explicit method would need, instead of hardcoding
index 39 and a magic constant.

Problems with no closed form are first-class; `has_exact` is false and the
harness skips accuracy scoring rather than inventing a reference.

## HTTP contract

All endpoints are POST with JSON bodies, except `/api/meta` and
`/api/health`.

| Endpoint | Purpose |
|---|---|
| `/api/ode/solve` | integrate one problem, return trajectory, stats, accuracy |
| `/api/ode/convergence` | refinement study over several methods |
| `/api/phy/ber` | Monte Carlo sweep plus the matching theory curve |
| `/api/phy/constellation` | received scatter with EVM |
| `/api/phy/pulse` | shaping demo, eye, ISI, PAPR, bandwidth |
| `/api/phy/pathloss` | three path-loss models against distance |
| `/api/phy/link_budget` | margin and required SNR |
| `/api/meta` | everything the dashboard needs to populate its controls |

Three conventions:

**Optional fields are null, never absent.** `/api/ode/solve` always returns
`accuracy`, `exact`, `rel_error` and `conservation_drift`, set to null when
the problem does not support them, so a client can distinguish "not
applicable" from "the field went missing".

**Unknown names are 404, invalid combinations are 422.** Asking for a solver
that does not exist is a different error from asking for a convergence study
on an adaptive method, and they get different codes.

**Limits are enforced server-side.** Step counts, symbol counts and sweep
lengths are capped so a browser cannot request an unbounded computation.
Failed integrations return 200 with `success: false`: a diverging explicit
method on a stiff problem is a result, and one the dashboard displays.

## Dashboard

Plain ES modules, no build step, no framework, no CDN. `plot.js` is a small
canvas plotting library supporting linear and log axes, error bars, markers,
legends and device-pixel-ratio scaling; `app.js` wires the controls to the
endpoints.

Everything is computed on request. The five views share one status indicator
and one error path, so a server-side failure surfaces as a message rather
than an empty chart.

Colours come from CSS custom properties, with the measured series in blue and
the reference in red consistently across every view and every generated
figure, so a curve means the same thing in the browser as it does in the
documentation.

## Extending

**A new solver.** Subclass `ExplicitRK` or `ImplicitSolver`, supply the
tableau, register it. Add its order conditions to `tests/test_solvers.py`;
the parametrised tests pick up the new name automatically.

**A new problem.** Write a factory returning an `ODEProblem` and register it
in `PROBLEMS`. If it has an analytical solution, pass `exact` and it joins
the accuracy harness; if it has an invariant, pass `conserved` and drift is
tracked.

**A new constellation.** Add a factory producing a `Modulation` with unit
mean energy and a Gray-coded labelling. The parametrised tests in
`tests/test_modulation.py` will check energy normalisation, round-trip
fidelity and single-bit neighbour labelling without further code.

**A new channel.** Extend `_apply_channel` in `phy/link.py` and add the name
to `CHANNELS`. Supply a closed form in `theory.py` if one exists; if not, the
sweep still runs and simply has nothing to be scored against.
