# Physical normalization and Hamiltonian sensitivity — verified multi-seed tests

**Date:** October 30, 2026  
**Norm-penalty run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40978368394 — 70 model trainings, 6 regression tests passed.  
**Hamiltonian-parameter run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40978618960 — 48 model trainings. Both verified via GitHub job logs.

We are testing an **exactly solvable noninteracting, finite 20-bath-mode** fermion model. This is NOT the original correlated interacting SIAM/HEOM/PINN-DQME problem. The proposed lookahead explicitly uses its exact small-system propagator, so computational savings over direct simulation are not plausible.

## Prespecified norm-penalty study

Means across seeds (1,2,3,7,11), 1,200 training steps each, 4 sequential windows:

| T (meV) | objective | terminal error | trajectory RMSE | max norm drift | training sec |
|---:|---|---:|---:|---:|---:|
| .03 | uniform | .024487 | .031399 | .045469 | 1.78 |
| .03 | uniform + norm penalty 1 | .031616 | .036055 | .014052 | 1.70 |
| .03 | global goal, weight .25 | .014722 | .034545 | .053846 | 2.09 |
| .03 | global goal, weight .50 | **.004621** | .038624 | .063089 | 2.07 |
| .03 | global goal, weight .75 | .007020 | .045320 | .074096 | 2.08 |
| .03 | global goal, weight .50, norm penalty .1 | .016622 | .034060 | .011994 | 2.20 |
| .03 | global goal, weight .50, norm penalty 1 | .017353 | .055150 | .013723 | 2.19 |
| .30 | uniform | .024627 | .030488 | .045469 | 1.57 |
| .30 | uniform + norm penalty 1 | .028032 | .033603 | .014052 | 1.69 |
| .30 | global goal, weight .25 | .015043 | .033872 | .056038 | 2.08 |
| .30 | global goal, weight .50 | **.005065** | .038295 | .067934 | 2.08 |
| .30 | global goal, weight .75 | .006671 | .045727 | .082581 | 2.09 |
| .30 | global goal, weight .50, norm penalty .1 | .015852 | .033177 | .011923 | 2.20 |
| .30 | global goal, weight .50, norm penalty 1 | .014434 | .048733 | .013855 | 2.19 |

A norm penalty substantially improves norm accuracy but reduces the terminal-observable improvement. Even the best norm-penalized setting has ~1.2% norm drift: it is not an exact physical constraint. Strong training losses do not automatically enforce valid quantum time evolution.

## Prespecified Hamiltonian variation

Three seeds (1,3,7); results below are means of terminal absolute occupation error. Each model is **retrained from scratch**; this is parameter sensitivity, not transfer learning.

| Parameters (eps_d meV, Gamma meV) | T | Uniform error | Global lookahead error | Change |
|---|---:|---:|---:|---:|
| Original (-0.50,0.10) | .03 | .02343 | .00405 | -82.7% |
| Original (-0.50,0.10) | .30 | .02368 | .00454 | -80.8% |
| Detuned (-0.25,0.10) | .03 | .01880 | .01542 | -18.0% |
| Detuned (-0.25,0.10) | .30 | .01773 | .01502 | -15.3% |
| Stronger coupling (-0.50,0.25) | .03 | .03051 | .00386 | -87.3% |
| Stronger coupling (-0.50,0.25) | .30 | .03288 | .00196 | -94.0% |
| Joint shift (-0.85,0.20) | .03 | .20163 | .19694 | -2.3% |
| Joint shift (-0.85,0.20) | .30 | .19555 | .19424 | -0.7% |

The joint-shift model fails badly for **both** objectives. The lookahead improvement is highly problem-dependent and does NOT make the solver reliable over the tested Hamiltonian family. Some regimes also exhibit large wavefunction norm drift.

## Research implication

A narrow optimization effect is reproducible in selected settings. It is NOT robust physical simulation and not a computational speedup. The next controlled ablation replaces soft norm penalties by a **hard differentiable unit-wavefunction parametrization**, enforcing norm to machine precision. We must still compare against existing PINNs and accurate many-body solvers, and construct a meaningful strong-memory interacting benchmark.

The global-terminal sensitivity used here is a known goal-oriented idea. A novel paper must establish a substantial improvement on difficult non-Markovian physics **without** relying on an exact solution already cheaper than the neural model.
