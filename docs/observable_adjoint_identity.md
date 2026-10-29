# Observable-specific adjoint error control: the mathematical target

**Status:** Derivation of a known goal-oriented error identity, NOT a new theorem or demonstrated machine-learning improvement.

## Why our original weighted PINN failed

Our first toy network weighted an amplitude residual, but judged the nonlinear excited population `|G(T)|²` and also global trajectory RMSE. A loss intended for a terminal amplitude is not guaranteed to optimize either of those metrics, and its four-seed performance was worse on global RMSE.

For the real fermionic DQME/HEOM, we should target a physical observable **directly**.

## Exact finite-dimensional identity

Fix an approximation of the bath and a finite hierarchy with a time-independent linear generator `L` acting on a stacked vector of all ADO matrix entries:

```
  d y(t) / dt = L y(t),
  O(t*) = Re[c* y(t*)].
```

The vector `c` is zero on all nonzero ADO tiers and represents the reduced-density-matrix functional `Tr(n_up rho_0)` on tier zero.

For a differentiable neural approximation `y_hat`, set

```
  e(t) = y_hat(t) - y(t)
  r(t) = d y_hat(t)/dt - L y_hat(t).
```

Then `e'(t) = L e(t) + r(t)`, so variation of constants gives **exactly**

```
  O_hat(t*) - O(t*) =
    Re[c* exp(L t*) e(0)] +
    Re[integral_0^{t*} z_t*(s)* r(s) ds],
  z_t*(s) = exp(L* (t* - s)) c.
```

The first term vanishes only if the neural initial state reproduces the complete initial ADO hierarchy, **not merely its reduced density matrix**. The notation `*` denotes conjugate transpose.

For a time-dependent Liouvillian (e.g., a voltage quench), the matrix exponential is replaced by its two-time propagator, with the corresponding backward adjoint.

## Research proposal and caveats

An observable-specific residual objective might be designed from `z_t*(s)`, either using tier weights, a small family of adjoint snapshots, or an integrated dual-weighted residual. It is **not** sufficient to weigh residual norms arbitrarily: cancellations in the signed integral and errors in unseen observables matter.

A correctly weighted network cannot eliminate error from:
- omitted hierarchy tiers;
- finite Padé expansion;
- time integrator truncation;
- mismatched fermionic coupling and lead-phase conventions;
- a wrongly initialized correlated state.

Those are *modelling/reference errors*, not neural equation-residual errors.

## Mandatory baselines

1. Correct source-order HEOM at two or more hierarchy depths and finer Padé order, with physicality, trace and integrator checks.
2. Uniform-residual neural solver with time-domain decomposition (already established in Cao et al.).
3. Standard adaptive residual sampling, equivalent model capacity and wall-clock budget.
4. Goal-oriented and sensitivity-weighted methods from established numerical analysis (not an invented principle).
5. Comparison with HEOM and other non-neural solvers, including the cost of adjoint sensitivities.

**Go/no-go:** Retain an error-aware neural model only if it improves a prespecified physical observable at matched *total* compute cost, with valid uncertainty/error accounting under source-matched strong-memory conditions.
