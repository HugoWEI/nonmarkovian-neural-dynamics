# Four-initialization check — finite-temperature Anderson PINN

Date: 2026-10-29. **Independent finite-discrete-reservoir Anderson benchmark, NOT published fermionic PINN-DQME/HEOM.**

Conditions: T=0.3, eps_dot=-0.5, U=1, bath energy=-0.7, coupling=0.55, time interval [0,5], two-hidden-layer width-48 networks, Adam lr=0.003, float64, 3000 optimization steps. Initial dot empty, lead prepared in a thermal mixture. Global fit has 81 collocation points per step; piecewise fit has 41 per step within each of five subdomains. Same number of optimizer steps does not imply identical workload.

| Seed | Global PINN occupation RMSE | Five-window PINN occupation RMSE | Global max trace error | Piecewise max trace error |
|---:|---:|---:|---:|---:|
| 1 | 0.265807 | 0.004231 | 0.618273 | 0.012551 |
| 2 | 0.221890 | 0.003762 | 0.462152 | 0.015562 |
| 3 | 0.216354 | 0.004447 | 0.445603 | 0.017743 |
| 7 | 0.238086 | 0.004400 | 0.526630 | 0.017997 |

**Interpretation.** Time-domain decomposition improves the finite-bath global fit dramatically across these four tested initializations, with smaller trace drift. This is a **reproduction of a known methodological advantage in a simpler independent model**, not a novel solution of low-temperature error-backflow in the published DQME. The finite bath's Hamiltonian is temperature-independent: varying temperature changes initial reservoir occupations, not the continuum thermal correlation kernel. It must therefore not be used to infer a general temperature-vs-memory relationship.

This reference is exact diagonalization of a small finite system. It does not benchmark against an independently converged HEOM continuum calculation, and exact diagonalization is cheaper here.

**Next:** reconstruct the published fermionic hierarchy/PINN at low T with a converged HEOM reference and perform a numerical error budget. No original neural improvement has been demonstrated.
