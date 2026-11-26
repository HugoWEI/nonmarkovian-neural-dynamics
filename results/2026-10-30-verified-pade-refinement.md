# Verified higher-order Padé refinement from GitHub Actions artifacts

**Date:** October 30, 2026  
**GitHub run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40973756268

High- and low-temperature jobs completed. The actual `pade-refinement-high` and `pade-refinement-low` ZIP artifacts were downloaded and their `results.json` files parsed independently. The calculation uses a factorized **empty impurity**, evolution from 0 to 1.5 ps, and fixed fermionic HEOM **hierarchy depth 2**. **This is not the paper's original initial ADO state.**

## Maximum full-trajectory occupation difference at fixed depth 2

| T/Gamma | Nk increment | Max difference | Terminal difference |
|---:|---|---:|---:|
| 3.0 | 3 -> 5 | 4.90161384e-5 | 1.92604788e-5 |
| 3.0 | 5 -> 7 | 1.43384881e-6 | 5.18086763e-7 |
| 0.3 | 3 -> 5 | 0.01022571024 | 0.01022571024 |
| 0.3 | 5 -> 7 | 0.00162054597 | 0.00119358210 |
| 0.3 | 7 -> 9 | 0.00030736663 | 0.00011652121 |
| 0.3 | 9 -> 11 | 0.00006876382 | 0.00002458556 |

High-T source order `Nk=3` gives a <5e-5 maximum change compared to Nk5, while low-T source order `Nk=7` gives a <3.1e-4 maximum change compared to Nk9 on this *fixed-depth* test.

### Scientific interpretation

The Padé representation appears substantially less limiting than truncating the hierarchy at depth 2 on this specified 1.5-ps benchmark, because earlier depth-1 vs depth-2 discrepancies at source Nk were about 0.0096 (high) and 0.0113 (low).

**This is not full numerical convergence.** It does not control:
- hierarchy depth beyond 2;
- ODE tolerance bias independently;
- much longer-time propagation;
- source-correlated ADO initial state;
- original PINN-DQME observables and timing.

It does show that QuTiP's source-anchored fermionic bath and sampled observable are numerically stable against further Padé expansion, providing a valuable input to the next depth study.

The raw trajectories and numeric results are retained as GitHub Actions artifacts, not only transcribed into this report.
