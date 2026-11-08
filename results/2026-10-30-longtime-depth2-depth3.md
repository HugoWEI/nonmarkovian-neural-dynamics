# Verified 6 ps source-order HEOM: low-T long-memory convergence breakdown

**Date:** October 30, 2026  
**Cloud run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40974518320  
**Method:** Independent QuTiP spinful interacting Anderson HEOM, each temperature with the Padé order recovered from Cao et al.'s original bath coefficients, compared at depths 2 and 3 on 121 common times over t=0..6 ps.

**This is not the published Cao et al. DQME initial state:** starts from an empty impurity with factorized higher ADOs, instead of authors' nonzero correlated source hierarchy.

| T/Gamma | source Nk | depth | number ADOs | terminal spin-up occupation | min density eigenvalue | integration seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 3 | 2 | 529 | 0.482837704881 | 0.0 | 0.099 |
| 3.0 | 3 | 3 | 5489 | 0.483450983807 | 0.0 | 1.194 |
| 0.3 | 7 | 2 | 2081 | 0.489940311666 | 0.0 | 0.207 |
| 0.3 | 7 | 3 | 43745 | 0.495571249270 | 0.0 | 24.074 |

## Maximum differences over the entire 6 ps window

| T/Gamma | Max absolute depth-2/3 occupation difference | RMS difference | Terminal difference |
|---:|---:|---:|---:|
| 3.0 | 0.00072492894 | 0.00050952003 | 0.00061327893 |
| 0.3 | **0.00563093760** | 0.00260560188 | 0.00563093760 |

The low-temperature depth discrepancy is about 7.77x the high-temperature one on this specific example. Both trajectories were trace-conserving to approximately 1e-15 and had no negative sampled reduced-density eigenvalues. Positivity and trace preservation do **not** establish truncation convergence.

## Interpretation

At low temperature, a 1.5-ps depth-2/3 discrepancy of only 1.48594e-4 grows to a **maximum of 5.63094e-3 by 6 ps**. The low-temperature bath's slowest fitted time constant is approximately 6.98 ps. This is a direct warning against using short-time or shallow HEOM predictions as reliable targets for long-time neural training.

This is hierarchy-truncation sensitivity, **not a measurement of PINN training errors** or proof of the specific error-backflow mechanism derived by Cao et al. The original correlated full ADO state has still not been mapped to QuTiP's hierarchy.

High and low T use different source Padé orders (Nk=3 and Nk=7); that comparison is not a controlled change of temperature alone.

At low temperature with source Nk=7 there are K=64 independent fermionic exponent modes; depth 4 would contain sum_{k=0}^4 binom(64,k) = **679,121 ADOs**, compared to 43,745 at depth 3. This may be prohibitively expensive once sparse RHS, integrator, and steady-state solve overheads are included.

## Next decision: make a truly reliable long-time reference before ML

- Do not call the low-T depth-3 solution fully converged merely because it is a deeper hierarchy.
- For rigorous controlled deep-learning comparisons, use either a reduced spinless/subset bath model allowing much deeper HEOM convergence, or a distinct independently controlled reference (e.g., exact noninteracting solution).
- Work toward a source-consistent full ADO initial state, if attempting direct reproduction of Cao et al.
- Investigate a source-order adaptive hierarchy/tensor approximation only after confirming what standard HEOM or published neural baselines can already do.

The residual-to-observable error identity and its caveats are documented in `docs/observable_adjoint_identity.md`. No original deep-learning improvement has been demonstrated.
