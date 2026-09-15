"""Regenerate every figure and results file in one pass.

Run with ``python -m physim.experiments.run_all`` or ``make experiments``.
Everything written here is reproducible from a fixed seed, so a rerun on the
same commit produces the same numbers; the environment block in
``results/validation.json`` records what produced them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from physim.phy.link import ber_sweep, constellation_samples
from physim.phy.pulse import shaped_link_demo
from physim.phy.theory import theoretical_ber
from physim.problems import get_problem
from physim.solvers import get_solver
from physim.validation import convergence_study, work_precision
from physim.validation.report import run_all

FIGURES = Path("figures")
RESULTS = Path("results")

# Consistent, colour-blind-safe series colours across every figure.
MEASURED = "#1550c8"
REFERENCE = "#a6192e"
THIRD = "#0f7357"
FOURTH = "#8a5a00"
PALETTE = [MEASURED, REFERENCE, THIRD, FOURTH, "#6b21a8"]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8)


def figure_convergence(plt) -> dict:
    """Fitted order of accuracy for every fixed-step method."""
    problem = get_problem("damped_oscillator")
    methods = ["forward_euler", "heun", "rk4", "backward_euler",
               "trapezoidal", "radau_iia5"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    out = []
    for i, name in enumerate(methods):
        study = convergence_study(name, problem,
                                  step_counts=(50, 100, 200, 400, 800, 1600))
        colour = PALETTE[i % len(PALETTE)]
        ax.loglog(study.step_sizes, study.errors, "o-", color=colour, ms=3.5,
                  lw=1.3, label=f"{name} (p = {study.observed_order:.2f})")
        out.append(study.as_dict())
    _style(ax, "Order of accuracy, damped oscillator",
           "step size h", "max relative error")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.png", dpi=160)
    plt.close(fig)
    return {"problem": problem.name, "studies": out}


def figure_work_precision(plt) -> dict:
    """Accuracy bought per right-hand-side evaluation."""
    problem = get_problem("damped_oscillator")
    curves = work_precision(
        ["heun", "rk4", "radau_iia5", "rk45", "trbdf2"], problem,
        step_counts=(50, 100, 200, 400, 800, 1600),
        tolerances=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, (name, points) in enumerate(curves.items()):
        ax.loglog([p["n_rhs"] for p in points], [p["error"] for p in points],
                  "o-", color=PALETTE[i % len(PALETTE)], ms=3.5, lw=1.3,
                  label=name)
    _style(ax, "Work against precision, damped oscillator",
           "right-hand-side evaluations", "max relative error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "work_precision.png", dpi=160)
    plt.close(fig)
    return curves


def figure_stiff(plt) -> dict:
    """The stability argument for implicit methods, in one picture."""
    problem = get_problem("prothero_robinson")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    t_ref = np.linspace(*problem.t_span, 2000)
    ax.plot(t_ref, problem.exact_at(t_ref)[:, 0], color="#12171c", lw=2.0,
            label="exact", zorder=1)

    summary = []
    for i, name in enumerate(["forward_euler", "backward_euler",
                              "trapezoidal", "radau_iia5"]):
        result = get_solver(name).solve(problem, n_steps=200)
        colour = PALETTE[i % len(PALETTE)]
        mark = "" if result.success else " (diverged)"
        ax.plot(result.t, result.y[:, 0], color=colour, lw=1.2, ls="--",
                label=name + mark)
        exact = problem.exact_at(result.t)
        scale = max(float(np.max(np.abs(exact))), 1e-300)
        err = float(np.max(np.abs(result.y - exact)) / scale)
        summary.append({"method": name, "success": bool(result.success),
                        "max_rel_error_pct": 100 * err if result.success else None,
                        "message": result.message})

    ax.set_ylim(-1.6, 1.6)  # the diverging trace would otherwise own the axis
    _style(ax, "Same problem, same 200 steps: stiff stability",
           "time", "y")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "stiff_stability.png", dpi=160)
    plt.close(fig)
    return {"problem": problem.name, "n_steps": 200, "methods": summary}


def figure_transmission_line(plt) -> dict:
    """A 40-state RLGC ladder against its matrix exponential."""
    problem = get_problem("transmission_line")
    result = get_solver("radau_iia5", rtol=1e-10, atol=1e-14).solve(
        problem, n_steps=3000)
    exact = problem.exact_at(result.t)
    far = problem.params["far_end_index"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True,
                                   height_ratios=[2, 1])
    ns = result.t * 1e9
    ax1.plot(ns, exact[:, far], color=REFERENCE, lw=2.0, label="matrix exponential")
    ax1.plot(ns, result.y[:, far], color=MEASURED, lw=1.1, ls="--",
             label="Radau IIA, order 5")
    ax1.axhline(problem.params["dc_far_end_v"], color="#5c6b78", lw=0.8, ls=":",
                label="DC divider, 50/55")
    _style(ax1, "Step response at the far end of a 10 m line", "", "volts")
    ax1.legend(fontsize=8, loc="lower right")

    scale = float(np.max(np.abs(exact)))
    ax2.semilogy(ns, np.abs(result.y[:, far] - exact[:, far]) / scale,
                 color=THIRD, lw=1.0)
    _style(ax2, "", "time (ns)", "relative error")
    fig.tight_layout()
    fig.savefig(FIGURES / "transmission_line.png", dpi=160)
    plt.close(fig)

    err = float(np.max(np.abs(result.y - exact)) / scale)
    return {
        "max_rel_error_pct": 100 * err,
        "n_states": problem.dim,
        "explicit_steps_required": problem.params["explicit_steps_required"],
        "steps_used": result.stats.n_steps,
        "settled_v": float(result.y[-1, far]),
        "dc_divider_v": problem.params["dc_far_end_v"],
    }


def figure_ber(plt) -> dict:
    """Monte Carlo against closed form for four constellations."""
    schemes = ["bpsk", "qpsk", "qam16", "qam64"]
    ranges = {"bpsk": (0, 10), "qpsk": (0, 10), "qam16": (4, 16),
              "qam64": (8, 20)}
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    out = []
    for i, scheme in enumerate(schemes):
        lo, hi = ranges[scheme]
        sweep = ber_sweep(scheme, np.arange(lo, hi + 1, 2), seed=12345,
                          target_errors=500, max_symbols=4_000_000)
        colour = PALETTE[i % len(PALETTE)]
        smooth = np.linspace(lo, hi, 200)
        ax.semilogy(smooth, theoretical_ber(smooth, scheme), color=colour,
                    lw=1.2, alpha=0.8)
        points = [p for p in sweep.points if p.n_bit_errors > 0]
        ax.errorbar([p.ebn0_db for p in points], [p.ber for p in points],
                    yerr=[[p.ber - p.ci_low for p in points],
                          [p.ci_high - p.ber for p in points]],
                    fmt="o", color=colour, ms=4, lw=1, capsize=2, label=scheme)
        out.append(sweep.as_dict())
    ax.set_ylim(1e-7, 1)
    _style(ax, "Bit error rate: simulated points, closed-form curves",
           "Eb/N0 (dB)", "bit error rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ber_curves.png", dpi=160)
    plt.close(fig)
    return {"sweeps": out}


def figure_constellations(plt) -> dict:
    """What the noise actually does to each constellation."""
    cases = [("qpsk", 8.0), ("qam16", 14.0), ("qam64", 20.0), ("qam256", 26.0)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    out = []
    for ax, (scheme, ebn0) in zip(axes, cases, strict=False):
        data = constellation_samples(scheme, ebn0, n_symbols=4000, seed=7)
        rx = np.array(data["received"])
        ideal = np.array(data["ideal"])
        ax.scatter(rx[:, 0], rx[:, 1], s=1.0, color=MEASURED, alpha=0.35,
                   linewidths=0)
        ax.scatter(ideal[:, 0], ideal[:, 1], s=26, marker="+",
                   color=REFERENCE, linewidths=1.1)
        ax.set_aspect("equal")
        ax.set_title(f"{scheme}, {ebn0:g} dB\nEVM {data['evm_pct']:.1f} %",
                     fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)
        out.append({k: data[k] for k in
                    ("scheme", "ebn0_db", "evm_pct", "evm_snr_db", "esn0_db")})
    fig.tight_layout()
    fig.savefig(FIGURES / "constellations.png", dpi=160)
    plt.close(fig)
    return {"cases": out}


def figure_pulse_shaping(plt) -> dict:
    """Eye diagram and the spectral cost of the rolloff factor."""
    demo = shaped_link_demo("qpsk", beta=0.35, sps=8, n_symbols=4000,
                            ebn0_db=18.0, seed=3, return_waveforms=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    eye = demo["eye"]
    for trace in eye["traces"]:
        ax1.plot(eye["t"], trace, color=MEASURED, lw=0.5, alpha=0.3)
    _style(ax1, f"Eye diagram, rolloff 0.35, {demo['ebn0_db']:g} dB",
           "symbol periods", "amplitude")

    for i, beta in enumerate((0.15, 0.35, 0.75)):
        d = shaped_link_demo("qpsk", beta=beta, sps=8, n_symbols=4000,
                             ebn0_db=None, seed=3, return_waveforms=True)
        w = d["tx_waveform"]
        spec = np.abs(np.fft.fftshift(np.fft.fft(w, 8192))) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(8192, 1 / (8 * 1e6))) / 1e6
        ax2.plot(freqs, 10 * np.log10(spec / spec.max()),
                 color=PALETTE[i], lw=1.0, label=f"beta = {beta}")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-70, 5)
    _style(ax2, "Transmit spectrum against rolloff", "MHz", "dB")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "pulse_shaping.png", dpi=160)
    plt.close(fig)
    return {k: demo[k] for k in
            ("beta", "isi_residual", "papr_tx_db", "evm_pct",
             "occupied_bw_hz", "nyquist_bw_hz", "bit_errors")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="short Monte Carlo runs, for a smoke test")
    parser.add_argument("--no-figures", action="store_true",
                        help="results only, no matplotlib dependency")
    args = parser.parse_args(argv)

    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    payload: dict = {}

    if not args.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; run with --no-figures "
                  "or `pip install physim[plots]`")
            return 1

        for name, fn in [
            ("convergence", figure_convergence),
            ("work_precision", figure_work_precision),
            ("stiff", figure_stiff),
            ("transmission_line", figure_transmission_line),
            ("ber", figure_ber),
            ("constellations", figure_constellations),
            ("pulse_shaping", figure_pulse_shaping),
        ]:
            t0 = time.perf_counter()
            payload[name] = fn(plt)
            print(f"  {name:20s} {time.perf_counter() - t0:6.1f} s")

    print("  validation suite ...")
    report = run_all(quick=args.quick)
    report.write(RESULTS / "validation.json")
    payload["validation_summary"] = report.as_dict()["summary"]

    (RESULTS / "experiments.json").write_text(json.dumps(payload, indent=2,
                                                         default=float))
    print(f"\nwrote {RESULTS}/experiments.json and {RESULTS}/validation.json "
          f"in {time.perf_counter() - started:.1f} s")
    print(f"accuracy pass rate {100 * report.accuracy_pass_rate:.0f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
