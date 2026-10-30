# First-order interacting HEOM memory feedback: quench-strength robustness and full CPU cost

**Date:** October 30, 2026  
**GitHub Actions:** baseline [0.05 meV quench](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40983944363); prespecified [0.025/0.10/0.20 meV quench sweep](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40984134551). All jobs completed. Numerical outputs retrieved directly from GitHub execution logs.

We simulate an interacting spinful Anderson impurity (U=1meV) with one Lorentzian lead per spin, factorized only with respect to **omitted top-tier auxiliary operators**: the retained depth2 initial state is a correlated HEOM steady state at mu=0, and the high-depth reference starts from the same embedded initial state with depth3 omitted-tier ADOs zero. This is NOT the fully equilibrated depth3 correlated state and NOT Cao et al.'s original two-lead benchmark.

## Accuracy versus independent depth3 evolution

Relative reduction in absolute terminal occupation error, compared with uncorrected depth2:

| Quench mu_R (meV) | High T / Nk3 | Low T / Nk7 |
|---:|---:|---:|
| 0.025 | 92.0% | 73.0% |
| 0.050 | 92.0% | 73.1% |
| 0.100 | 92.1% | 73.9% |
| 0.200 | 92.3% | 77.4% |

This is a **first-order Born/Volterra omitted-tier response**, using a depth2 forward/adjoint and the sparse coupling blocks B,C,D of the depth3 generator. It does not train a neural network and assumes zero omitted-tier ADOs initially. The accuracy has only been evaluated on a limited quench-strength family; it has not been established across Hubbard U, reservoir bandwidth, multiple states, or longer times.

## Complete on-runner CPU cost, counting low-depth steady setup and BOTH generator constructions

| Quench mu (meV) | Low T Nk7 predictor seconds | Low T Nk7 direct depth3 seconds from same initial state | Relative reduction |
|---:|---:|---:|---:|
| 0.025 | 1.980 | 2.335 | 15.2% |
| 0.100 | 2.538 | 3.183 | 20.2% |
| 0.200 | 2.984 | 3.621 | 17.6% |

At **high temperature**, full predictor and direct depth3 costs are approximately identical (~1.2–1.9 seconds), sometimes with the predictor slightly slower.

These are paired calculations on each GitHub hosted CPU runner; no repeated-run timing distributions or statistically precise hardware benchmark are available. Neither approach includes the expensive fully correlated **depth3** steady-state construction because the test explicitly uses the same embedded **depth2** steady state. Thus these timings **cannot** be compared directly with the previous ~48–54 sec depth3 fully correlated steady-state setup cost as though both solved identical physical preparation problems.

## Novelty and interpretation

- Schur-complement / Volterra elimination of auxiliary operators is standard.
- Prior work includes Vadimov, *On truncations of hierarchical equations of motion for finite-dimensional systems*, SciPost Physics Core 9 (2026) 062, DOI 10.21468/SciPostPhysCore.9.3.062; Adriazola/Roszak (2026) on learning Volterra memory kernels, DOI 10.1103/bc2w-c6dj.
- This validates a useful numerical correction on a limited interacting fermionic family, with a **modest ~1.2x end-to-end CPU advantage at low T** under a controlled shared-initial-state setup, not a novel physical theorem or a new deep learning solver.
- The strongest next checks are a 12ps memory-horizon stress test, state preparation with nonzero omitted correlated ADOs, robustness to multiple Hamiltonians, and an actual neural approximation that beats this *strong classical* first-order baseline including training and preprocessing costs.

**Go/no-go rule:** Do not advance an ML claim if the classical memory correction itself is faster and more accurate under identical budgets.
