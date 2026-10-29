# Verified two-axis source-anchored HEOM convergence study

**Date:** October 30, 2026.  
**GitHub run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40973022902  
**Commit:** `996121fedf5b06e155886a7c5329408a5146e9ba`

Both matrix jobs (high and low T) **successfully completed**, and their numerical logs were inspected through the GitHub connector. Reference: `scripts/heom_convergence.py`.

## Final spin-up occupations at 1.5 ps

| T/Gamma | Nk | depth 1 | depth 2 | depth 1 vs 2 terminal delta |
|---:|---:|---:|---:|---:|
| 3.0 | 1 | 0.2206280774 | 0.2108242782 | 0.0098037992 |
| 3.0 | 3 | 0.2271143864 | 0.2174702016 | 0.0096441848 |
| 0.3 | 1 | 0.1862578036 | 0.1814531663 | 0.0048046373 |
| 0.3 | 3 | 0.2221144192 | 0.2113821721 | 0.0107322471 |
| 0.3 | 7 | 0.2341220987 | 0.2228014645 | 0.0113206342 |

## Maximum occupation differences along entire trajectory

| T/Gamma | Comparison at fixed parameter | Max difference |
|---:|---|---:|
| 3.0 | depth 1 vs 2, Nk=3 | 0.009644185 |
| 3.0 | Nk 1 vs 3, depth=2 | 0.006685039 |
| 0.3 | depth 1 vs 2, Nk=7 | 0.01132063 |
| 0.3 | Nk 3 vs 7, depth=2 | 0.01141929 |

Both solvers report trace error below 1e-15; sampled reduced density matrices remained positive and Hermitian within numerical tolerance.

## Critical limitation

**These HEOM results are not converged.** A green workflow and physically normalized density matrix do not establish Padé-order or depth convergence. The previous depth-2, Nk=1 benchmark was a numerical smoke test.

The chosen initial condition is an **empty impurity with factorized ADOs**, NOT the original authors' correlated initial DQME hierarchy. These results should not be used as published-reference physical ground truth. Their role is an implementation and two-axis convergence diagnostic.

## Next action

1. Test depth 3 at fixed Nk where feasible.
2. Test additional Padé order at depth 2 to determine if source Nk approximates a converged bath.
3. Independently converge ODE tolerances.
4. Prepare a *full correlated ADO state* using `HEOMSolver.steady_state()`, and validate its stationarity before any quench, as an independent benchmark. The exact authors' initial ADO hierarchy is still unmatched.
5. Only after physical reference convergence proceed to neural sensitivity training.
