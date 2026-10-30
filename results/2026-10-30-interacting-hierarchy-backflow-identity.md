# Interacting HEOM error-backflow decomposition: exact finite-hierarchy identity

**Date:** October 30, 2026  
**Workflow:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40983410447  
**Verified Nk3 result; source-order Nk7 outcome is in the same run.**

We start with the **same high-depth correlated ADO steady state**, project its retained tiers to the shallower hierarchy, and evolve both under the same post-quench reservoir chemical potential. This avoids comparing states prepared at separately truncated equilibria.

Let A_hi and A_lo denote their sparse Liouvillians, P the embedding of retained ADOs, y_hi(0) the original full high-depth steady hierarchy, and y_lo(0)=P* y_hi(0). The embedded low-depth approximation is yhat=P y_lo and its residual against the high-depth dynamics is

\[
r(t)=P A_\mathrm{lo} y_\mathrm{lo}(t)-A_\mathrm{hi}P y_\mathrm{lo}(t).
\]

The initial mismatch is e(0)=P y_lo(0)-y_hi(0), which is nonzero on omitted correlated ADO tiers. For a final linear observable c* y(T), define z(t)=exp[A_hi* (T-t)]c. Then

\[
\operatorname{Re}(c^\dagger[\widehat y(T)-y_\mathrm{hi}(T)])=
\operatorname{Re}(z(0)^\dagger e(0))+
\operatorname{Re}\int_0^T z(t)^\dagger r(t)\,dt.
\]

This is an established dual-weighted residual identity; our contribution here is careful numerical verification on an interacting non-Markovian open quantum model.

## Verified low-temperature Nk3, depth2 versus depth3

Six-picosecond evolution, U=1meV spinful SIAM, one Lorentzian lead per spin, fully correlated higher-depth initial HEOM ADO hierarchy.

| Diagnostic | Value |
|---|---:|
| True terminal low-depth minus high-depth occupation | **-0.001470363638** |
| Initial omitted-tier state contribution | -0.000303614013 |
| Dynamical omitted-tier backflow contribution | -0.001166749659 |
| Sum (full adjoint reconstruction) | -0.001470363673 |
| Reconstruction discrepancy due to numerical quadrature | **3.44e-11** |
| Fraction of residual norm living on omitted hierarchy indices | **1.000000** |

Thus ~20.6% of the signed error is attributable to omitted initial ADOs and ~79.4% to dynamical influence of omitted tiers, for this specific observable, model, and setup. These percentages are not universal and both contributions have the same sign here.

## Compressed adjoint error on the actual hierarchy boundary residual

| Rank | Estimated total terminal error | Absolute error vs direct high-minus-low | Conservative discrete compression bound |
|---:|---:|---:|---:|
| 2 | -0.001528722822 | 5.84e-5 | 6.20e-2 |
| 4 | -0.001481172826 | 1.08e-5 | 1.03e-2 |
| 8 | **-0.001470323409** | **4.02e-8** | 1.53e-4 |
| 16 | -0.001470363670 | 3.13e-11 | 8.96e-9 |
| 32 | -0.001470363673 | 3.48e-11 | 3.01e-9 |

The bound covers only the sampled quadrature **compression** (including the omitted-tier initial value) and is not a physical HEOM convergence bound. Rank16 has a stringent small bound in this particular test; rank8 has a small *observed* error but a conservative bound that allows much larger error.

**Crucial limitation:** the full high-depth forward and backward trajectories were computed to verify this decomposition, so this is NOT yet a cheap predictive a posteriori estimator and NOT a neural solver. A publishable method would need to produce an accurate approximate dual much more cheaply and generalize to unseen parameters/residuals without a high-depth solution.
