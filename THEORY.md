# Theory

## 1. Runge-Kutta methods

An s-stage method advances `y' = f(t, y)` by

```
Y_i = y_n + h * sum_j a_ij * f(t_n + c_i*h, Y_j)
y_{n+1} = y_n + h * sum_i b_i * f(t_n + c_i*h, Y_i)
```

The coefficients `(A, b, c)` are the Butcher tableau. When `A` is strictly
lower triangular each stage is an explicit formula; otherwise the stages are
coupled and must be solved.

### Order conditions

A method has order `p` when its Taylor expansion matches the true solution's
through `h^p`. Rather than trusting a transcribed tableau, the tests check the
algebraic conditions directly. The quadrature conditions

```
sum_i b_i * c_i^(k-1) = 1/k        for k = 1..p
```

are necessary for order `p`, and for collocation methods such as Radau IIA the
stage conditions

```
sum_j a_ij * c_j^(k-1) = c_i^k / k
```

pin the internal stages. A single mistyped digit breaks one of these
identities at the twelfth decimal place, which is a far sharper test than
noticing that an answer looks slightly wrong three modules downstream. See
`tests/test_solvers.py`.

### Embedded error estimation

An embedded pair carries a second weight vector `b_hat` of one order lower.
The difference between the two results estimates the local error at the cost
of no extra stages:

```
err = h * sum_i (b_i - b_hat_i) * f(t_n + c_i*h, Y_i)
```

Dormand-Prince 5(4) additionally satisfies `a_{7j} = b_j`, so the final stage
of one step is the first stage of the next. That is the FSAL property, and it
makes a nominally seven-stage method cost six evaluations per accepted step.

### Step-size control

Given a scaled error norm

```
scale_i = atol + rtol * max(|y_i|, |y_new_i|)
E = sqrt(mean((err_i / scale_i)^2))
```

the step is accepted when `E <= 1`. The next step follows a PI controller

```
h_new = h * safety * E^(-alpha) * E_prev^(beta)
```

rather than the textbook `E^(-1/(p+1))`. The integral term damps the
oscillation in step size that a pure proportional rule produces near a
stability boundary, which matters most on exactly the mildly stiff problems
where an explicit method is still viable.

## 2. Stability, and what stiffness actually is

Applying a method to `y' = lambda*y` gives `y_{n+1} = R(h*lambda) * y_n`. The
stability function `R(z)` decides whether errors grow.

| Method | R(z) | Region |
|---|---|---|
| Forward Euler | `1 + z` | `|1+z| <= 1`, a disc of radius 1 |
| Backward Euler | `1/(1-z)` | everything outside a disc: A-stable, L-stable |
| Trapezoidal | `(1+z/2)/(1-z/2)` | the whole left half-plane: A-stable |
| Radau IIA (5) | Pade (3,2) | A-stable and L-stable |

**A-stability** means the left half-plane is stable: no step size restriction
from decaying modes. **L-stability** adds `R(z) -> 0` as `z -> -inf`, so a very
fast mode is annihilated in a single step rather than merely kept bounded.

The distinction is visible, not academic. The trapezoidal rule is A-stable but
not L-stable: as `h*lambda -> -inf` its amplification factor tends to `-1`, so
a stiff transient alternates in sign instead of vanishing. Backward Euler and
Radau IIA damp it immediately. `tests/test_stiff.py` counts the sign changes in
the first few steps and asserts the ordering.

### Two different definitions of stiff

The usual measure is the ratio of the largest to smallest eigenvalue modulus.
It works for Prothero-Robinson (1e4) and Robertson (1e9), and the library
reports it as `stiffness_ratio`.

It fails for the transmission line, whose 40 eigenvalues all have similar
magnitude: the ratio is 17.8, which by that test is not stiff at all. Yet an
explicit method still needs about 115 steps per transit time purely to stay
inside its stability disc, long before accuracy asks for anything. That is the
operational definition of stiffness, *the step size is dictated by stability
rather than by accuracy*, and it is the test the library's `stiff` flag uses
for this problem. Both numbers are reported so neither is mistaken for the
other.

### The implicit cost

Each implicit stage requires solving `Y = y_n + h*a_ii*f(t, Y)`. A simplified
Newton iteration uses

```
(I - h*a_ii*J) * dY = -(Y - y_n - h*a_ii*f(t, Y))
```

with `J` held fixed across iterations so one LU factorisation serves them all.
For the 3-stage Radau the stages couple, and the iteration matrix becomes
`I_3d - h*(A kron J)`, a `3d x 3d` system.

Convergence is monitored by the contraction rate. If the iteration fails, a
fixed-step method has no recourse, and the solver reports failure rather than
returning the last iterate. This is not hypothetical: van der Pol at mu = 100
makes fixed-step Radau fail at the switch near t = 81, at every step size
tried. An earlier version accepted the diverged iterate and returned a
trajectory pinned at 3748 with `success = True`. The adaptive drivers instead
treat a Newton failure as a step rejection, which shrinks `h` and usually
recovers; TR-BDF2 clears the same problem with 485 rejections.

## 3. Digital transmission

### Constellations

Every constellation is scaled to unit average symbol energy, so `Eb/N0`
comparisons across schemes are like for like. With `k = log2(M)` bits per
symbol and complex AWGN of total variance `N0`,

```
N0 = Es / (k * 10^(EbN0_dB/10)),   Es = 1
```

and each quadrature carries `N0/2`. A factor of two here is the most common
bug in a link simulator, so it is pinned by measurement in
`tests/test_channel.py` rather than by assertion.

Labelling is Gray coded, so nearest neighbours differ in exactly one bit and
a symbol error usually costs one bit rather than `k`. The tests verify this
over the actual constellation geometry for all six schemes.

Demodulation uses analytic slicers: for square QAM the decision is
independent per dimension, which is `O(n)` instead of the `O(n*M)` of a
nearest-neighbour search, and the tests confirm the two agree exactly even
deep in the noise.

### Closed-form bit error rates

For BPSK and Gray-coded QPSK,

```
BER = Q(sqrt(2*Eb/N0))
```

Both, identically: QPSK is two orthogonal BPSK channels, so it buys twice the
rate at the same energy per bit. This is why the two curves coincide in the
BER figure.

For square M-QAM the library computes an **exact** result rather than the
usual nearest-neighbour approximation. A square constellation is two
independent PAM constellations, one per quadrature, so the bit error rate is
obtained by summing the probability of every PAM level transition weighted by
the Hamming distance between the Gray labels involved:

```
BER = (1/k) * sum over (i,j) of P(j received | i sent) * d_H(g_i, g_j)
```

with the transition probabilities built from differences of Q-functions at the
decision boundaries. The familiar approximation

```
BER ~ (4/k) * (1 - 1/sqrt(M)) * Q(sqrt(3*k/(M-1) * Eb/N0))
```

keeps only the nearest-neighbour term. The two agree to within 1e-5 across the
whole operating range and diverge below it, where the noise reaches past the
adjacent level: at 4 dB on 64-QAM the approximation is already 0.4 % optimistic.
Having both makes that gap measurable instead of assumed.

**A precision floor.** Below about 1e-20 the exact sum loses significance to
cancellation between nearly equal Q-function terms, and the two formulas
separate by exactly a factor of two. Nothing operates there (1e-20 is one
error per three million years at 10 Gb/s), but the limit is recorded as
`QAM_EXACT_PRECISION_FLOOR` so no meaning is read into those digits.

### Fading

Rayleigh coefficients are complex Gaussian with unit mean power, modelling a
scattered path with no line of sight. Rician adds a specular component with
ratio `K`. Both are normalised so the average gain stays at 1 and the only
change is the distribution, not the mean.

Independent draws per symbol describe a channel that changes infinitely fast.
Real fading has memory, and its duration is what decides whether interleaving
helps, so the library also provides Jakes sum-of-sinusoids fading with the
correct autocorrelation

```
R(tau) = J0(2*pi*f_d*tau)
```

checked against the Bessel function directly in the tests.

Averaging the AWGN result over the Rayleigh distribution gives a closed form
that decays as `1/SNR` rather than exponentially, a diversity order of one.
The simulator reproduces both the shape and the slope.

### Confidence intervals

A measured BER is a binomial proportion. The normal approximation fails badly
at small error counts, and collapses to zero width at zero errors, claiming
certainty from a run that observed nothing. The library uses the Wilson score
interval, which stays bounded in `[0, 1]` and gives a meaningful upper limit
when no errors were seen. Every reported point carries one.

### Pulse shaping

A root-raised-cosine filter at the transmitter and its matched pair at the
receiver cascade to a raised cosine, which is zero at every non-zero symbol
instant: no intersymbol interference at the sampling points. Truncating the
filter to a finite span breaks that exactly, and the residual is measured
rather than assumed: 0.0058 at rolloff 0.35 over a 10-symbol span, falling
below 0.002 at 20 symbols.

The rolloff trades bandwidth against peak-to-average power. A lower rolloff
occupies less spectrum and produces sharper peaks, which is what forces
amplifier backoff. Both sides are reported.
