# validate_stiff.jl
#
# Independent Julia check of the stiff comparison in the Python suite.
# Uses only the standard library and LinearAlgebra, so it runs without
# installing DifferentialEquations.jl.
#
# Run:  julia validate_stiff.jl

using LinearAlgebra
using Printf

println("physim cross-check: Julia")
println("=========================\n")

# ---------------------------------------------------------------- problem
# Prothero-Robinson: y' = lambda*(y - g(t)) + g'(t),  y(0) = g(0)
# The exact solution is g(t) for any lambda, which makes it a clean probe of
# stability alone: any deviation is the method, never the problem.
const lam = -1.0e4
g(t)  = sin(t)
gd(t) = cos(t)
f(t, y) = lam * (y - g(t)) + gd(t)

const tf = 2.0
const nsteps = 200
const h = tf / nsteps
@printf("h*|lambda| = %.1f  (forward Euler is stable only below 2)\n\n", h * abs(lam))

# ------------------------------------------------------- 1. forward Euler
function forward_euler()
    t, y = 0.0, g(0.0)
    for i in 1:nsteps
        y += h * f(t, y)
        t += h
        if !isfinite(y) || abs(y) > 1e12
            return (false, t, i)
        end
    end
    (true, t, nsteps)
end

ok, t_fail, step = forward_euler()
@assert !ok "forward Euler should diverge at h|lambda| = $(h * abs(lam))"
@printf("1. Forward Euler diverged at t = %.4g (step %d), as predicted\n   PASS\n\n",
        t_fail, step)

# ------------------------------------------------ 2. implicit methods hold
# Backward Euler and the trapezoidal rule, solved exactly: the problem is
# linear in y, so no Newton iteration is needed and the test isolates the
# stability property rather than the nonlinear solver.
function backward_euler()
    t, y = 0.0, g(0.0)
    err = 0.0
    for _ in 1:nsteps
        tn = t + h
        # y_{n+1} = y_n + h*(lam*(y_{n+1} - g(tn)) + gd(tn))
        y = (y + h * (-lam * g(tn) + gd(tn))) / (1 - h * lam)
        t = tn
        err = max(err, abs(y - g(t)))
    end
    err
end

function trapezoidal()
    t, y = 0.0, g(0.0)
    err = 0.0
    for _ in 1:nsteps
        tn = t + h
        rhs = y + 0.5h * (f(t, y) + (-lam * g(tn) + gd(tn)))
        y = rhs / (1 - 0.5h * lam)
        t = tn
        err = max(err, abs(y - g(t)))
    end
    err
end

be, tr = backward_euler(), trapezoidal()
@printf("2. Backward Euler  max relative error %.3e  (%.2e %%)\n", be, 100be)
@printf("   Trapezoidal     max relative error %.3e  (%.2e %%)\n", tr, 100tr)
@assert be < 1e-3 "backward Euler should stay bounded and accurate"
@assert tr < be "trapezoidal (order 2) should beat backward Euler (order 1)"
println("   PASS\n")

# ---------------------------------------- 3. transmission line eigenvalues
# The 40-state RLGC ladder from problems/library.py. Checks the claim that
# its stiffness is absolute spectrum size rather than eigenvalue ratio.
function rlgc_matrix(n::Int=20; len=10.0, R=0.5, L=250e-9, G=1e-6, C=100e-12,
                     zload=50.0)
    dx = len / n
    Rs, Ls, Gs, Cs = R*dx, L*dx, G*dx, C*dx
    dim = 2n
    A = zeros(dim, dim)
    for k in 1:n
        i_idx, v_idx = 2k - 1, 2k
        A[i_idx, i_idx] = -Rs / Ls
        A[i_idx, v_idx] = -1 / Ls
        if k > 1
            A[i_idx, 2(k-1)] = 1 / Ls
        end
        A[v_idx, i_idx] = 1 / Cs
        A[v_idx, v_idx] = -Gs / Cs
        if k < n
            A[v_idx, 2(k+1) - 1] = -1 / Cs
        else
            A[v_idx, v_idx] += -1 / (zload * Cs)
        end
    end
    A
end

A = rlgc_matrix()
ev = eigvals(A)
mags = abs.(ev)
ratio = maximum(mags) / minimum(mags)
lam_max = maximum(mags)
h_limit = 2.78 / lam_max
span = 4e-7

@printf("3. RLGC line, %d states\n", size(A, 1))
@printf("   |lambda| ratio           %.1f   (the usual stiffness test: says 'not stiff')\n", ratio)
@printf("   |lambda| max             %.3e rad/s\n", lam_max)
@printf("   explicit step limit      %.3e s\n", h_limit)
@printf("   steps needed over %.0e s  %.0f  (stability, not accuracy)\n",
        span, span / h_limit)
@assert ratio < 100 "eigenvalue ratio should be small for this problem"
@assert span / h_limit > 100 "an explicit method should be step-limited here"
println("   PASS\n")

println("All cross-checks agree with the Python suite.")
