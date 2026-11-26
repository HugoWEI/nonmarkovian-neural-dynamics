# Hard-normalized global adjoint parameter stress test — verified negative result

**Date:** October 30, 2026  
**GitHub run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40979149100  
**Evidence:** 48 completed neural trainings, one physical-norm smoke test passed, logs parsed directly through GitHub.

Quadratic finite 20-bath-mode model, each Hamiltonian **retrained separately**, T=0.03/0.3 meV, three seeds (1,3,7), 1,200 Adam optimization steps. Both losses have exactly unit-normalized wavefunctions (maximum sampled norm drift <1e-15).

| Dot energy (meV), hybridization (meV) | T (meV) | Standard hard-norm terminal error | Global hard-norm terminal error | Relative improvement |
|---|---:|---:|---:|---:|
| (-0.5, 0.10) original | 0.03 | .07256485 | .01393003 | +80.8% |
| (-0.5, 0.10) original | 0.30 | .05069268 | .00739193 | +85.4% |
| (-0.25, 0.10) detuned | 0.03 | .02968594 | .03534225 | **-19.1%** |
| (-0.25, 0.10) detuned | 0.30 | .01716088 | .02675615 | **-55.9%** |
| (-0.5, 0.25) stronger coupling | 0.03 | .13854275 | .06410135 | +53.7% |
| (-0.5, 0.25) stronger coupling | 0.30 | .10749008 | .05509795 | +48.7% |
| (-0.85, 0.20) joint shift | 0.03 | .12161812 | .17272343 | **-42.0%** |
| (-0.85, 0.20) joint shift | 0.30 | .16898304 | .18880528 | **-11.7%** |

**Interpretation:** The global-terminal optimization effect is parameter-sensitive. In two of four Hamiltonian families, hard normalization plus global lookahead produces **worse** final occupation than the same hard-normalized standard PINN. Moreover, the stronger-coupling and joint-shift cases have large absolute errors, so neither method should be treated as a reliable physical solver merely because it meets the norm constraint.

### Independent absolute compute cost

[Cost benchmark](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40979357712): 200 repeats of constructing and diagonalizing the 21×21 Hamiltonian and evaluating 201 occupation time points on a hosted CPU runner give median **0.0002219 s** per direct exact solution. Neural training runs typically take **0.7–1.1 s** on another hosted CPU run, several thousand times slower. These are indicative platform measurements rather than a rigorously matched hardware microbenchmark.

## Scientific conclusion

1. The goal-oriented PINN idea is **not new** (see [prior-art audit](../docs/prior_art_goal_oriented_quantum_pinn.md)).
2. The prespecified small-model tests give both positive and negative evidence.
3. No efficient or broadly reliable quantum solver has emerged from these tests.
4. Further tuning on this easy model would risk post-hoc optimization without physics payoff.
5. The scientifically valuable next task is a **smaller but genuinely interacting open quantum system** with correlated full-ADO initial states, HEOM convergence, and measurable costly long-time memory. Only then should an approximate error-backflow-aware neural technique be evaluated against serious baselines.
