# Validation

What is measured, how, and what each number is allowed to mean.

Run `make validate` to regenerate `results/validation.json`. Every figure in
this document comes from that file.

## The principle

A numerical result is only as good as what it was compared against. Comparing
a solver to a finer run of itself measures self-consistency, not correctness:
a method with a sign error converges beautifully to the wrong answer. So
every accuracy claim here is against an independent reference:

| Claim | Reference |
|---|---|
| Solver accuracy | analytical solution of the same problem |
| Order of accuracy | the method's theoretical order |
| Tableau correctness | the algebraic order conditions |
| Stiff behaviour | stability theory's prediction of divergence |
| Bit error rate | closed-form expression for the same constellation |
| Channel statistics | the distribution's own moments |
| Conservation | an exact invariant of the system |

## 1. Accuracy against closed forms

Thirty-two problem-method pairs. For each, the numerical trajectory is
compared with the analytical solution evaluated on the same grid.

**The metric.** Error is normalised by the amplitude of the reference, not
pointwise:

```
scale = max over t of |y_exact(t)|,  per component
error = max over t of |y_num(t) - y_exact(t)| / scale
```

A pointwise relative error is unusable on any oscillatory solution: it goes
to infinity at every zero crossing, and the resulting figure says more about
where the grid points fell than about the method. Amplitude normalisation
keeps the number meaningful and comparable across problems.

**The result.** All 32 pass the 0.01 % threshold. The worst is Radau IIA on
`stiff_linear` at 0.0045 %, which is the fast transient at `lambda = -1000`
being resolved by a fixed step rather than any deficiency in the method.

**What this number does not mean.** It is a statement about discretisation
error on problems with known solutions, evaluated at the tolerances listed in
the report. It is not a bound for arbitrary problems, and it says nothing
about the Monte Carlo results in section 5.

## 2. Order of accuracy

For a method of order `p`, halving the step should divide the error by `2^p`.
Fitting `log(error)` against `log(h)` over several refinements recovers `p` as
the slope.

Two things spoil that fit, and both are handled explicitly.

**The round-off floor.** Once the discretisation error reaches machine
precision, further refinement buys nothing and the curve flattens. Points at
or below `1e-13` carry no information about the order and are dropped before
fitting. Radau IIA reaches this floor within the default step range, which is
why its study fits fewer points than Heun's.

**The pre-asymptotic regime.** At coarse steps the higher-order terms have
not yet become negligible and the local slope is not `p`. Points above a
relative error of 0.1 are excluded on the same grounds.

If no asymptotic range survives, the study reports `n_fitted = 0` and a NaN
slope rather than fitting a line through whatever is left. Forward Euler on
Prothero-Robinson at coarse steps does exactly this: every step size tried is
unstable, so there is no order to measure, and saying so is the correct
output.

Measured against theory: Forward Euler 1.029, Backward Euler 0.974, Heun
2.050, Trapezoidal 1.986, RK4 4.054, Radau IIA 5.006. Worst deviation 2.9 %.

## 3. Tableau verification

The coefficients are checked against their defining algebraic identities, not
against a printed table. For Radau IIA the quadrature conditions `B(1..5)` and
the collocation conditions `C(1..3)` hold to 1e-12; for Dormand-Prince the
order conditions hold through fifth order for `b` and through fourth for the
embedded `b_hat`; the FSAL row is verified to equal the weight vector.

This catches transcription errors that no accuracy test would localise.

## 4. Stiff behaviour

Stability theory predicts that forward Euler diverges for `h*|lambda| > 2` and
that A-stable methods do not. The suite runs that experiment.

Prothero-Robinson, `lambda = -1e4`, 200 steps, so `h*|lambda| = 50`:

| Method | Outcome | Max relative error |
|---|---|---|
| RK4 | diverged at t = 0.025, step 5 | — |
| Backward Euler | completed | 2.5e-5 % |
| Trapezoidal | completed | 4.0e-8 % |
| TR-BDF2 | completed | 2.0e-8 % |
| Radau IIA | completed | 1.4e-11 % |

The accuracy ordering follows the method orders, as it should.

**Detecting divergence.** Waiting for a floating-point overflow is too weak a
test. An amplification factor of 99 per step reaches 1e199 in a hundred steps
and is still a finite number, so an `isfinite` check passes and the solver
reports success on an answer wrong by two hundred orders of magnitude. The
drivers therefore declare divergence when the solution exceeds `1e12` times
the initial scale, which catches the instability near where it starts. The
ceiling is configurable for problems whose solutions genuinely grow.

**Conservation.** Robertson kinetics has an exact invariant, the total
concentration. Over nine decades of time, adaptive TR-BDF2 holds it to
3.6e-15. Fixed-step methods cannot: at 4000 uniform steps the initial
transient, a million times faster than the final approach, is unresolved and
the drift reaches 1.2. This is reported as a property of the step strategy,
not hidden.

## 5. Bit error rate

Four constellations, 26 Eb/N0 points, 17.8 million bits, seeded.

**Effort follows errors, not bits.** Each point runs until it has collected a
target number of bit errors or exhausts its symbol budget, so precision is
roughly constant down the curve instead of collapsing at high SNR.

**Deviations are sampling noise.** A run that stops at 500 errors has a
relative standard error of `1/sqrt(500)`, about 4.5 %. The worst measured
deviation is 8.3 % on a BPSK point that stopped at 105 errors, where the
standard error is 9.8 %. These are the numbers a correct simulator produces;
a simulator reporting 0.1 % agreement at these counts would be reporting a
bug.

**The measure that matters is horizontal.** Implementation loss, the dB shift
needed to lay the measured curve onto the theoretical one, is insensitive to
per-point scatter because it uses the whole curve. It stays within 0.015 dB
for all four schemes, positive for the low-order schemes and slightly negative
for the high-order ones, which is consistent with noise rather than with a
systematic receiver penalty.

**Interval coverage.** Theory falls inside the 95 % Wilson interval at 24 of
26 points. Both misses are understood and neither indicates a defect:

- 64-QAM at 14 dB reads 5.5 % low on 1221 errors, a 1.9-sigma fluctuation.
  Over 26 points, one or two such excursions are expected.
- 64-QAM at 20 dB collected one error where 0.1 was expected. No interval
  built on a single count can absorb that, and the honest reading is that the
  point is an upper bound rather than a measurement.

**What this number does not mean.** It is not comparable with the 0.01 % of
section 1. There the reference is exact and the only error is discretisation;
here the reference is exact but the measurement is a finite sample, and
driving the sampling error to 0.01 % would require roughly 1e8 errors per
point.

## 6. Physical-layer models

Checked against their own definitions rather than against each other:

- **Noise convention.** Complex AWGN carries `N0/2` per quadrature, measured
  over 400,000 samples.
- **Fading.** Rayleigh envelope mean `sqrt(pi)/2` at unit power; Rician `K`
  recovered from the ratio of specular to scattered power; Jakes
  autocorrelation matched against `J0(2*pi*f_d*tau)` at several lags.
- **Path loss.** Friis at 2.4 GHz and 1 m gives 40.05 dB; doubling distance
  costs 6.02 dB in free space; the log-distance model reduces exactly to free
  space at exponent 2; the two-ray model steepens to 40 dB per decade past the
  breakpoint at `4*pi*h_t*h_r/lambda`.
- **Noise floor.** `kTB` at 290 K gives -173.98 dBm/Hz.
- **Required SNR.** 16-QAM at 1e-6 needs 14.4 dB Eb/N0, against a textbook
  14.5 dB.
- **Quantisation.** Measured SQNR matches `6.02B + 1.76` dB for a full-scale
  sine, and `6.02B` for a uniform input; the 1.76 dB is a crest-factor term,
  not a property of the converter, and the two cases are tested separately so
  the constant is not treated as universal.
- **Phase noise.** Wiener increment variance `2*pi*linewidth/fs` measured to
  1 %, with the linear growth of variance checked as a ratio.
- **Pulse shaping.** Raised-cosine taps vanish at every non-zero symbol
  instant; the truncated RRC pair leaves 0.0058 residual ISI at rolloff 0.35
  over 10 symbols.

## 7. Reproducibility

Every stochastic routine takes an explicit seed, and each sweep records the
seed that produced it. Two runs of the same sweep agree bit for bit; the test
suite asserts it. `results/validation.json` carries the interpreter, NumPy and
SciPy versions and the platform, so a number that later fails to reproduce can
be traced to an environment change rather than argued about.

CI runs the suite on Python 3.10, 3.11 and 3.12, and uploads the validation
JSON as an artefact on every run.

## 8. Known limits

- Fixed-step implicit methods fail on van der Pol at mu = 100: the switch is
  too sharp for a fixed step's Newton iteration at any step size tried. The
  solver reports the failure; use TR-BDF2.
- The exact QAM formula is meaningless below 1e-20 BER (section 3 of
  THEORY.md).
- Fading is modelled as flat: no delay spread, no frequency-selective
  channel, and therefore no equaliser beyond the single-tap case.
- The receiver assumes perfect synchronisation and perfect channel knowledge.
  Impairments can be injected but are not part of the default BER chain, so
  the measured curves are an ideal-receiver bound.
- Convergence studies require a closed-form reference; problems without one
  are rejected rather than silently compared against a fine-grid surrogate.
