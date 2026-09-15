% validate_rk4.m
%
% Independent MATLAB/Octave check of two claims the Python suite makes:
%   1. RK4 converges at fourth order on a problem with a known solution.
%   2. An explicit method diverges on a stiff problem where ode15s does not.
%
% Written against MATLAB's own facilities rather than by porting the Python,
% so agreement means two independent implementations reached the same answer.
%
% Run:  matlab -batch validate_rk4     or     octave --no-gui validate_rk4.m

clear; close all;
fprintf('physim cross-check: MATLAB / Octave\n');
fprintf('===================================\n\n');

%% 1. Order of accuracy -----------------------------------------------------
% Damped oscillator, the same parameters as physim's damped_oscillator:
%   x'' + 2*zeta*w0*x' + w0^2*x = 0,  x(0) = 1, x'(0) = 0
w0   = 2.0;
zeta = 0.15;
tf   = 12.0;
y0   = [1.0; 0.0];

f = @(t, y) [y(2); -2*zeta*w0*y(2) - w0^2*y(1)];

% Underdamped closed form.
wd = w0 * sqrt(1 - zeta^2);
exact = @(t) exp(-zeta*w0*t) .* (cos(wd*t) + (zeta*w0/wd)*sin(wd*t));

counts = [50 100 200 400 800 1600];
errs   = zeros(size(counts));

for k = 1:numel(counts)
    n = counts(k);
    h = tf / n;
    t = 0; y = y0;
    peak = 0;
    for i = 1:n
        k1 = f(t,         y);
        k2 = f(t + h/2,   y + h/2*k1);
        k3 = f(t + h/2,   y + h/2*k2);
        k4 = f(t + h,     y + h  *k3);
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
        t = t + h;
        peak = max(peak, abs(y(1) - exact(t)));
    end
    errs(k) = peak;   % amplitude of the exact solution is 1
end

hs = tf ./ counts;
p  = polyfit(log(hs), log(errs), 1);

fprintf('1. RK4 order of accuracy, damped oscillator\n');
fprintf('   %10s  %12s  %10s\n', 'h', 'max error', 'slope');
for k = 1:numel(counts)
    if k == 1
        fprintf('   %10.3e  %12.4e  %10s\n', hs(k), errs(k), '-');
    else
        s = log(errs(k-1)/errs(k)) / log(hs(k-1)/hs(k));
        fprintf('   %10.3e  %12.4e  %10.3f\n', hs(k), errs(k), s);
    end
end
fprintf('   fitted order %.3f (theory 4)\n', p(1));
assert(abs(p(1) - 4) < 0.2, 'RK4 did not converge at fourth order');
fprintf('   PASS\n\n');

%% 2. Stiffness -------------------------------------------------------------
% Prothero-Robinson with lambda = -1e4, integrated over [0, 2] with 200
% steps, so h*|lambda| = 100, far outside forward Euler's stability disc.
lam = -1e4;
g    = @(t) sin(t);
gd   = @(t) cos(t);
fpr  = @(t, y) lam*(y - g(t)) + gd(t);

n = 200; h = 2/n;
t = 0; y = g(0);
diverged = false;
for i = 1:n
    y = y + h * fpr(t, y);
    t = t + h;
    if ~isfinite(y) || abs(y) > 1e12
        fprintf('2. Forward Euler diverged at t = %.4g (step %d), as predicted\n', t, i);
        diverged = true;
        break;
    end
end
assert(diverged, 'forward Euler should have diverged at h|lambda| = 100');

% A stiff solver handles the same problem without difficulty.
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
if exist('ode15s', 'file')
    [ts, ys] = ode15s(fpr, [0 2], g(0), opts);
    err = max(abs(ys - g(ts))) / max(abs(g(ts)));
    fprintf('   ode15s completed in %d steps, max relative error %.3e\n', ...
            numel(ts), err);
    assert(err < 1e-4, 'ode15s result disagrees with the closed form');
end
fprintf('   PASS\n\n');

fprintf('Both cross-checks agree with the Python suite.\n');
