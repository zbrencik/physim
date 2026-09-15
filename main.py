"""HTTP API and dashboard server.

Thin by design: every endpoint validates its input, calls into ``physim``,
and serialises the result. No numerics live here, so the API and the CLI
cannot disagree about what the engine computes.

Run with ``physim serve`` or ``uvicorn api.main:app --reload``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from physim import __version__
from physim.phy.channel import (
    breakpoint_distance_m,
    free_space_path_loss_db,
    link_budget,
    log_distance_path_loss_db,
    two_ray_path_loss_db,
)
from physim.phy.link import ber_sweep, constellation_samples
from physim.phy.modulation import get_modulation, list_modulations
from physim.phy.pulse import shaped_link_demo
from physim.phy.theory import required_ebn0_db, theoretical_ber
from physim.problems import get_problem, list_problems
from physim.solvers import get_solver, list_solvers
from physim.validation.convergence import convergence_study
from physim.validation.metrics import conservation_drift, score

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="physim",
    version=__version__,
    description="Numerical analysis and physical-layer simulation engine",
)

# Ceilings that keep a single request from monopolising the process.
MAX_STEPS = 200_000
MAX_SYMBOLS = 2_000_000
MAX_SWEEP_POINTS = 40


# --------------------------------------------------------------- ODE endpoints
class SolveRequest(BaseModel):
    problem: str = "damped_oscillator"
    solver: str = "rk45"
    steps: int = Field(2000, ge=4, le=MAX_STEPS)
    rtol: float = Field(1e-8, gt=0, le=1e-1)
    atol: float = Field(1e-11, gt=0, le=1e-1)
    t_end: float | None = None
    max_points: int = Field(1500, ge=50, le=20_000)


@app.post("/api/ode/solve")
def api_solve(req: SolveRequest) -> dict[str, Any]:
    """Integrate a problem and, where possible, score it against the exact solution."""
    try:
        problem = get_problem(req.problem)
        solver = get_solver(req.solver, rtol=req.rtol, atol=req.atol)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    t_span = problem.t_span
    if req.t_end is not None:
        if req.t_end <= t_span[0]:
            raise HTTPException(status_code=422,
                                detail="t_end must be greater than t0")
        t_span = (t_span[0], req.t_end)

    kwargs = {} if solver.adaptive else {"n_steps": req.steps}
    result = solver.solve(problem, t_span=t_span, **kwargs)
    payload = result.as_dict(max_points=req.max_points)
    payload["problem"] = problem.meta()
    payload["solver"] = solver.meta()

    # These keys are always present, null when the problem does not support
    # them, so a client can tell "not applicable" from "field went missing".
    payload["exact"] = None
    payload["accuracy"] = None
    payload["rel_error"] = None
    payload["conservation_drift"] = conservation_drift(result, problem)

    if problem.has_exact:
        t_dec = np.array(payload["t"])
        exact = problem.exact_at(t_dec)
        payload["exact"] = exact.T.tolist()
        rep = score(result, problem)
        payload["accuracy"] = rep.as_dict()
        payload["accuracy"]["passes_001pct"] = rep.passes(0.01)
        # Error trace on the decimated grid, for the residual plot.
        y_dec = np.array(payload["y"]).T
        scale = np.max(np.abs(exact), axis=0)
        scale[scale == 0] = 1.0
        payload["rel_error"] = (np.abs(y_dec - exact) / scale).T.tolist()
    return payload


class ConvergenceRequest(BaseModel):
    problem: str = "damped_oscillator"
    solvers: list[str] = ["heun", "rk4", "radau_iia5"]
    counts: list[int] = [50, 100, 200, 400, 800]
    metric: Literal["max", "final"] = "max"


@app.post("/api/ode/convergence")
def api_convergence(req: ConvergenceRequest) -> dict[str, Any]:
    """Measure the empirical order of accuracy of one or more methods."""
    if any(c < 4 or c > MAX_STEPS for c in req.counts):
        raise HTTPException(status_code=422,
                            detail=f"step counts must lie in [4, {MAX_STEPS}]")
    try:
        problem = get_problem(req.problem)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    if not problem.has_exact:
        raise HTTPException(
            status_code=422,
            detail=f"'{req.problem}' has no closed-form solution to measure against",
        )

    studies = []
    for name in req.solvers[:6]:
        try:
            study = convergence_study(name, problem,
                                      step_counts=tuple(req.counts),
                                      metric=req.metric)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        studies.append(study.as_dict())
    return {"problem": problem.meta(), "studies": studies}


# --------------------------------------------------------------- PHY endpoints
class BERRequest(BaseModel):
    scheme: str = "qpsk"
    channel: Literal["awgn", "rayleigh", "rician"] = "awgn"
    ebn0_start: float = Field(0.0, ge=-20, le=60)
    ebn0_stop: float = Field(12.0, ge=-20, le=60)
    ebn0_step: float = Field(2.0, gt=0, le=10)
    target_errors: int = Field(200, ge=10, le=5000)
    max_symbols: int = Field(200_000, ge=1000, le=MAX_SYMBOLS)
    seed: int = 12345
    rician_k_db: float = 6.0


@app.post("/api/phy/ber")
def api_ber(req: BERRequest) -> dict[str, Any]:
    """Run a Monte Carlo BER sweep and return it alongside the closed form."""
    if req.ebn0_stop < req.ebn0_start:
        raise HTTPException(status_code=422,
                            detail="ebn0_stop must not be below ebn0_start")
    grid = np.arange(req.ebn0_start, req.ebn0_stop + 1e-9, req.ebn0_step)
    if grid.size > MAX_SWEEP_POINTS:
        raise HTTPException(
            status_code=422,
            detail=f"sweep has {grid.size} points, limit is {MAX_SWEEP_POINTS}",
        )
    try:
        mod = get_modulation(req.scheme)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    kwargs = {"rician_k_db": req.rician_k_db} if req.channel == "rician" else {}
    sweep = ber_sweep(req.scheme, grid, channel=req.channel, seed=req.seed,
                      target_errors=req.target_errors,
                      max_symbols=req.max_symbols, **kwargs)

    fine = np.linspace(grid[0], grid[-1], 200)
    return {
        "sweep": sweep.as_dict(),
        "modulation": mod.meta(),
        "theory_curve": {
            "ebn0_db": fine.tolist(),
            "ber": np.atleast_1d(
                theoretical_ber(fine, req.scheme, req.channel)
            ).tolist(),
        },
    }


class ConstellationRequest(BaseModel):
    scheme: str = "qam16"
    ebn0_db: float = Field(15.0, ge=-10, le=60)
    n_symbols: int = Field(2000, ge=100, le=20_000)
    channel: Literal["awgn", "rayleigh", "rician"] = "awgn"
    seed: int = 7


@app.post("/api/phy/constellation")
def api_constellation(req: ConstellationRequest) -> dict[str, Any]:
    try:
        data = constellation_samples(req.scheme, req.ebn0_db, req.n_symbols,
                                     req.channel, seed=req.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    return data


class PulseRequest(BaseModel):
    scheme: str = "qpsk"
    beta: float = Field(0.35, ge=0.0, le=1.0)
    sps: int = Field(8, ge=2, le=32)
    n_symbols: int = Field(300, ge=50, le=5000)
    ebn0_db: float = Field(20.0, ge=-10, le=60)
    span_symbols: int = Field(10, ge=2, le=32)


@app.post("/api/phy/pulse")
def api_pulse(req: PulseRequest) -> dict[str, Any]:
    return shaped_link_demo(req.scheme, req.beta, req.sps, req.n_symbols,
                            req.ebn0_db, req.span_symbols)


class PathLossRequest(BaseModel):
    frequency_hz: float = Field(2.4e9, gt=1e6, le=1e12)
    d_min_m: float = Field(1.0, gt=0)
    d_max_m: float = Field(5000.0, gt=0)
    n_points: int = Field(300, ge=10, le=2000)
    exponent: float = Field(3.0, ge=1.5, le=6.0)
    h_tx_m: float = Field(10.0, gt=0)
    h_rx_m: float = Field(1.5, gt=0)


@app.post("/api/phy/pathloss")
def api_pathloss(req: PathLossRequest) -> dict[str, Any]:
    """Compare the three attenuation models over a distance sweep."""
    if req.d_max_m <= req.d_min_m:
        raise HTTPException(status_code=422, detail="d_max must exceed d_min")
    d = np.logspace(np.log10(req.d_min_m), np.log10(req.d_max_m), req.n_points)
    return {
        "distance_m": d.tolist(),
        "free_space_db": free_space_path_loss_db(d, req.frequency_hz).tolist(),
        "log_distance_db": log_distance_path_loss_db(
            d, req.frequency_hz, exponent=req.exponent).tolist(),
        "two_ray_db": two_ray_path_loss_db(
            d, req.frequency_hz, req.h_tx_m, req.h_rx_m).tolist(),
        "breakpoint_m": float(breakpoint_distance_m(req.frequency_hz,
                                                    req.h_tx_m, req.h_rx_m)),
        "frequency_hz": req.frequency_hz,
    }


class LinkBudgetRequest(BaseModel):
    distance_m: float = Field(1000.0, gt=0)
    frequency_hz: float = Field(2.4e9, gt=1e6)
    scheme: str = "qam16"
    target_ber: float = Field(1e-6, gt=0, lt=0.5)
    tx_power_dbm: float = 20.0
    bandwidth_hz: float = Field(20e6, gt=0)
    noise_figure_db: float = 6.0
    model: Literal["free_space", "log_distance", "two_ray"] = "log_distance"
    exponent: float = Field(3.0, ge=1.5, le=6.0)


@app.post("/api/phy/link_budget")
def api_link_budget(req: LinkBudgetRequest) -> dict[str, Any]:
    try:
        mod = get_modulation(req.scheme)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    required_ebn0 = float(required_ebn0_db(req.target_ber, mod.M, mod.family))
    required_snr = float(required_ebn0 + 10 * np.log10(mod.bits_per_symbol))
    budget = link_budget(
        req.distance_m, req.frequency_hz, tx_power_dbm=req.tx_power_dbm,
        bandwidth_hz=req.bandwidth_hz, noise_figure_db=req.noise_figure_db,
        required_snr_db=required_snr, model=req.model, exponent=req.exponent,
    )
    out = budget.as_dict()
    out.update({
        "path_loss_db": budget.path_loss_db,
        "required_ebn0_db": required_ebn0,
        "required_snr_db": required_snr,
        "scheme": req.scheme,
        "spectral_efficiency_bps_hz": mod.bits_per_symbol,
        "throughput_mbps": float(mod.bits_per_symbol * req.bandwidth_hz / 1e6),
    })
    return out


# -------------------------------------------------------------- meta and pages
@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    return {
        "version": __version__,
        "solvers": list_solvers(),
        "problems": list_problems(),
        "modulations": list_modulations(),
        "channels": ["awgn", "rayleigh", "rician"],
    }


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
