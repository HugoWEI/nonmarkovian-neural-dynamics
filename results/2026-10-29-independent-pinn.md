# First independently executed quantum PINN experiment

**Date:** 2026-10-29. **Execution:** Python CPU environment, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0.

**Not a reproduction of Cao et al. finite-temperature Anderson-model PINN-DQME.** This is a separate exactly solvable zero-temperature Lorentzian reservoir, with an excited two-level system coupled to a damped pseudomode. We train an MLP PINN on the embedded equations `G'=-g B, B'=g G-lambda B` and compare with matrix-exponential reference.

Training: 2000 Adam steps, 81 collocation points, 2 hidden layers of width 32, float64, initial condition enforced exactly, seed 7. Terminal adjoint weights are available analytically in this toy; not a scalable DQME approximation.

| Regime | Residual objective | Excited-population RMSE | Absolute terminal-population error | CPU seconds |
|---|---|---:|---:|---:|
| g=0.8, lambda=0.2 (long memory) | Uniform | 0.0048719 | 0.00075969 | 2.15 |
| g=0.8, lambda=0.2 (long memory) | Adjoint-weighted | 0.0197450 | 0.00033023 | 2.21 |
| g=0.2, lambda=2.0 (short memory) | Uniform | 0.0021497 | 0.00147865 | 2.02 |
| g=0.2, lambda=2.0 (short memory) | Adjoint-weighted | 0.00068371 | 0.00048809 | 2.18 |

The strong-memory exact reference exhibits population revival: total upward excursion of the discrete sampled excited-population curve is ~0.658. The short-memory case has no upward excursion over the sampled interval. The norm of the embedded amplitudes obeys `d/dt (G^2+B^2) = -2lambda B^2`, so the reduced qubit evolution is physically bounded.

**Interpretation:** Adjoint weighting improves terminal population error for both examples at this seed and training budget. It *worsens global population RMSE in the strong-memory case*. This is a first experiment, not an improvement claim: one seed, a simple known reference, and no computational benefit over a 2x2 exact solver. Its purpose is to uncover tradeoffs and test the neural code pipeline.

**Next:** replicate the published finite-temperature fermionic benchmark, including HEOM convergence and real computational costs, before testing a research-level proposal.
