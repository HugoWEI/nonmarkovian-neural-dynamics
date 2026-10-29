# Verified full-ADO HEOM steady-state and voltage-quench experiment

**Date:** October 30, 2026  
**Source run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40973286595  
**Job outputs inspected via connected GitHub job logs.**

The calculation prepares the full HEOM auxiliary-density-operator state via `HEOMSolver.steady_state()` under **zero voltage bias**, then uses that *correlated hierarchy* as the initial condition for a quench of the right reservoir to 0.05 meV (left remains 0). This avoids the earlier **factorized empty-dot initial state**, but is still **NOT** the source authors' stored and correlated initial condition.

## Measured depth-1 results at source-matching Padé orders

| Diagnostic | T/Gamma=3, Nk=3 | T/Gamma=0.3, Nk=7 |
|---|---:|---:|
| Number of ADOs | 33 | 65 |
| Pre-quench steady occupation n_up | 0.4999999999999998 | 0.4999999999331334 |
| Stationary trajectory drift | 1.11e-16 | 1.67e-16 |
| Maximum post-quench occupation change | 0.003559720 | 0.002809930 |
| Post-quench terminal n_up | 0.5035597202 | 0.5028099296 |
| Minimum pre-quench density eigenvalue | +0.07943455 | +2.91e-8 |
| Minimum post-quench density eigenvalue | +0.07595268 | **-0.002674824** |
| Maximum post-quench trace error | 4.44e-16 | 5.55e-16 |

**Critical interpretation:** The low-T depth-1 HEOM trajectory has a negative reduced-density-matrix eigenvalue, violating positivity despite essentially exact trace conservation and stationarity before the quench. Therefore this depth-1 result **is not a reliable physical reference**, regardless of the success status of the cloud job.

The source authors' stored reduced populations were n_up≈0.29706 at high T and n_up≈0.11312 at low T—very different from our unbiased steady values of 0.5. This confirms **the independently prepared correlated quench is not their stored state**. It is a useful separate physics benchmark, not a substitute for mapping the original complete auxiliary hierarchy.

## Next investigation

The immediate priority is to check physical positivity at higher hierarchy depths with the same Padé order, while measuring the actual total cost of forming the steady hierarchy. `heom-quench-depth2.yml` compares selected depth-2 runs. Positivity alone will not certify hierarchy and Padé-order convergence.

No new deep-learning results or novelty claims are warranted by this calculation.
