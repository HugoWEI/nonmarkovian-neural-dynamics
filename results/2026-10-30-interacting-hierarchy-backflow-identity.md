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


## Verified source-order low-temperature Nk7, depth2 vs depth3

The same controlled nested-hierarchy test passed at the **original low-T source Padé order Nk=7**. The high-depth (5489-ADO) correlated initial hierarchy was projected onto the 529-ADO low-depth subspace, and both were evolved under identical post-quench parameters for 6ps.

| Diagnostic | Measured |
|---|---:|
| True terminal depth2 minus depth3 occupation | **-0.00139135666038** |
| Omitted initial correlated ADO contribution | **-0.00028025638320** |
| Omitted-tier dynamical backflow contribution | **-0.00111110031555** |
| Reconstructed total (Simpson adjoint) | -0.00139135669875 |
| Absolute quadrature reconstruction mismatch | **3.84e-11** |
| Fraction of high-depth residual norm on omitted tier | 1.000000 |
| High-depth correlated steady-state preparation | ~53.64 CPU seconds |
| Full forward/backward dual | ~3.85 CPU seconds |

The initial-state term accounts for ~20.1% and the dynamical boundary-residual term for ~79.9% of the signed terminal error **in this specific configuration**; this is not a general universal separation.

The low-rank dual estimates for the same **genuine hierarchy-boundary residual** were:

| POD rank | Estimated terminal error | Error versus true terminal difference | C-S compression bound versus exact discrete estimate |
|---:|---:|---:|---:|
| 2 | -0.001436491709 | 4.51e-5 | 5.90 |
| 4 | -0.001402619796 | 1.13e-5 | 8.79e-1 |
| **8** | **-0.001391392722** | **3.61e-8** | **1.80e-2** |
| **16** | **-0.001391356682** | **2.13e-11** | **1.35e-6** |
| 32 | -0.001391356696 | 3.51e-11 | 4.74e-7 |

**Interpretation:** The rank-8 method is empirically very accurate here, but its conservative compression bound is *much larger* than the desired correction: it cannot certify reliable error detection for arbitrary residual directions. Rank16 provides a sampled-quadrature compression bound of ~1.35e-6, smaller than the measured ~1.39e-3 hierarchy correction. Neither ranks nor bounds certify missing Padé/hierarchy terms relative to the physical continuum, and **neither is presently a cheap end-to-end predictor because the full high-depth adjoint was computed first**.

### Next methodological direction: approximate omitted-tier dynamics, not just compress a precomputed full adjoint

With retained state \(x\) and omitted top-tier state \(u\), partition the HEOM generator as

\[
L_\mathrm{hi}=\begin{bmatrix}A & B\\ C & D\end{bmatrix},
\quad x'=Ax+Bu,\quad u'=Cx+Du.
\]

Eliminating \(u\) gives the **standard memory-kernel identity**:

\[
x'(t)=Ax(t)+B e^{Dt}u(0)+
\int_0^t B e^{D(t-s)} Cx(s)\,ds.
\]

The first term is an initial correlated-state slip; the memory integral is the omitted-tier feedback. This is not a new theoretical derivation. It suggests a falsifiable *cheap approximation*: use a low-depth forward model, approximate the top-tier response \(u\) without a high-depth steady-state solve, then propagate its effect on final occupation using a low-depth adjoint. Any claimed benefit requires independent comparison with true higher-depth dynamics and honest end-to-end CPU costs.
