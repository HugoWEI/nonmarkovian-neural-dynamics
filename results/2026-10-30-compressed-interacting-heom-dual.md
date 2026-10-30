# Low-rank compressed adjoint for interacting non-Markovian HEOM

**Date:** October 30, 2026  
**Verified GitHub Actions:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40982886311  
**Model:** Spinful interacting Anderson U=1meV, 1 Lorentzian lead per spin, low T/Gamma=0.3, finite-temperature correlated ADO steady state prepared at mu=0, then 6ps mu=0.05meV quench. QuTiP sparse HEOM depth3 at Nk3 and Nk7. This is NOT Cao et al.'s original two-lead stored initial state.

## Numerical procedure

1. Construct finite-hierarchy sparse Liouvillian L using QuTiP.
2. Compute full forward y(t) and backward observable adjoint z(t) at 81 Simpson-grid points, without diagonalizing any dense many-body propagator.
3. Form the weighted adjoint snapshot matrix Z_w = diag(sqrt(w_i)) z(t_i), with positive Simpson quadrature weights w_i.
4. Compute its singular-vector (proper orthogonal decomposition/POD) rank-r approximation. This is a classical **established** technique, not a novel ML algorithm.
5. Reuse the compressed dual to approximate \partial_p O = Re sum_i w_i z_i^dagger (dL/dp)y_i.
6. Check a conservative Cauchy-Schwarz bound on the **sampled finite-hierarchy quadrature**:
   
   |g - g_r| <= ||Z_w-Z_{w,r}||_F ||diag(sqrt(w_i))[(dL/dp)y_i]||_F.
   
   The inequality makes no claim about errors in HEOM hierarchy, Padé approximation, correlated initial state, or continuous-time quadrature.

## Verified observed errors against the full discrete adjoint

| Low T model | rank | chemical potential abs error | U-derivative abs error | relative L2 dual error |
|---|---:|---:|---:|---:|
| Nk3 depth3 | 2 | 5.4734e-2 | (see artifact) | 8.18e-2 |
| Nk3 depth3 | 4 | 6.8140e-4 | (see artifact) | 1.36e-2 |
| Nk3 depth3 | 8 | **8.1472e-8** | **2.3935e-8** | 2.02e-4 |
| Nk3 depth3 | 16 | 1.65e-14 | 2.02e-13 | 1.16e-8 |
| Nk7 depth3 | 2 | 6.3927e-2 | (see artifact) | 8.21e-2 |
| Nk7 depth3 | 4 | 4.0744e-3 | (see artifact) | 1.23e-2 |
| Nk7 depth3 | 8 | **1.0173e-7** | **3.3894e-8** | 2.50e-4 |
| Nk7 depth3 | 16 | 2.75e-13 | 6.27e-13 | 1.85e-8 |

For source-order Nk7 depth3, the full 81×87824 complex adjoint snapshot requires **7,113,744 complex numbers**, whereas the rank8 POD stores **703,240 complex numbers**, a ~10.1x storage reduction (not necessarily 10x wall-clock acceleration).

The observed rank8 chemical-potential error (~1.0e-7) is much smaller than its Cauchy-Schwarz bound (~2.18e-2); this observed accuracy is **specific to the two tested derivative sources**. Rank16 provides a tighter conservative discrete bound (~1.61e-6 for mu), while the observed error is at floating-point precision.

## Full costs and lack of efficiency proof

At low T Nk7 depth3:
- Full correlated steady ADO preparation ~49 seconds on runner.
- Full sparse forward/adjoint snapshots ~3.75 seconds.
- Initial POD fit ~0.15 seconds.
- Rank8 *current unoptimized implementation* contraction ~0.16 seconds **per derivative direction**, compared with ~0.09–0.10 seconds for contracting the full precomputed dual.
- Thus **no actual wall-clock acceleration is established**, despite ~10x lower snapshot storage.

We should not label this "neural" or "novel" yet. The full adjoint still needs precomputation, and classical goal-oriented adjoints/POD are both known numerical methods.

## Critical next go/no-go test

1. Test residual families beyond mu/U derivatives, including localized high-tier hierarchy residuals and time-window interface mismatches. The success of rank8 on two derivatives is insufficient.
2. Measure amortization **including** steady-state setup, adjoint construction, POD fit, contraction, and any neural training.
3. Compare against cheap depth2 sensitivity, deeper and higher Padé HEOM references, plus direct finite differences.
4. Only attempt an interacting neural correction if the compressed dual improves predictive terminal occupation at matched *end-to-end* compute budget without sacrificing full-trajectory or physical consistency.
5. Verify novelty relative to adjoint-based a posteriori error analysis, POD model reduction and published PINN-DQME.

**Status: Validated sensitivity compression on two physical derivative directions. No original non-Markovian neural-solver result or efficiency improvement yet.**
