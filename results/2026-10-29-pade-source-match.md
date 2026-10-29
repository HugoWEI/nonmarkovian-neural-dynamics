# Exact-source fermionic Padé reconstruction (independently checked)

**Date:** October 29, 2026.  
**Primary sources:** [Cao et al. NQS-DQME code](https://github.com/caolong-cn/NQS-DQME_fermions/tree/7fb6ffb6115db42b299cc5c5092b27bfa9050628) and [QuTiP fermionic HEOM formulation](https://qutip.readthedocs.io/en/stable/guide/heom/fermionic.html).

## Result

The Padé correlation expansion was reconstructed independently in Python, with:
- `hbar = 0.658211928` meV ps (as in the original `ReadHamilton`);
- Lorentzian width `W = 5` meV;
- spectral coupling `Gamma = 0.1` meV;
- high-temperature `T/Gamma=3` corresponding to `T=0.3` meV and `Nk=3`, hence 4 exponentials including the Lorentzian pole;
- low-temperature `T/Gamma=0.3` corresponding to `T=0.03` meV and `Nk=7`, hence 8 exponentials.

Comparing the reconstructed coefficients with the manually cross-checked first source channel yields:

| Case | Maximum absolute coefficient discrepancy | Maximum decay-rate discrepancy |
|---|---:|---:|
| T/Gamma=3 | ~2.2e-13 | ~9e-16 |
| T/Gamma=0.3 | ~1.0e-11 | ~9e-16 |

These small differences reflect floating-point arithmetic and transcription precision in the manually checked coefficient arrays. The GitHub Actions job runs the same derivation directly against the full-precision files at a **pinned upstream revision**, requiring discrepancies below 1e-8.

The Padé poles and weights are obtained from the eigenvalues of tridiagonal matrices with coefficients described in the QuTiP fermionic-environment documentation. The bath correlation is represented as a Lorentzian pole plus Padé poles with complex weights. The independent script is [`scripts/pade_source_match.py`](../scripts/pade_source_match.py).

## Interpretation

This establishes the normalization and parameterization of the **zero-chemical-potential-phase fitted bath coefficients** in the original source. It corrects an earlier, provisional coupling `Gamma=0.05` meV: the match requires **0.1 meV**. It is a baseline reproducibility achievement, **not** an original theoretical result.

Crucial next work:
1. Verify the source's chemical-potential phase handling and the correspondence to QuTiP's C+ and C-.
2. Converge hierarchy depth, Padé order, and ODE tolerances independently.
3. Compare occupations and currents in the original published setup.
4. Recover or reconstruct the published neural PINN-DQME trajectory (the tutorial references a missing checkpoint).
5. Compare any newly trained error-aware model against the **piecewise** original baseline, not a weak global-network baseline.

The present GitHub Actions HEOM job has been configured, but its successful remote execution and physical convergence have **not** been confirmed in this report.
