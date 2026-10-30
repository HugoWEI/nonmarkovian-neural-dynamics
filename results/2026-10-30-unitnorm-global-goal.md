# Hard-normalized global-terminal fermionic PINN — verified hosted experiment

**Date:** October 30, 2026  
**GitHub Actions:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40978888253  
**Verification:** 40 complete training runs, five random seeds × two temperatures × four training configurations. Regression test of hard unit norm passed. Numerical values parsed independently from GitHub job logs.

## Experimental method

One spinless quantum resonant level with **20 discrete finite-bath fermionic modes** (21-dimensional single-particle Hamiltonian). Four sequential neural training windows, 1,200 optimization steps, T=0.03 or 0.3 meV. The model is **not** the source paper's correlated interacting SIAM/PINN-DQME.

The new \`unit_norm=True\` architecture returns
\`\`\`
q(t) = [w_initial + t * NN(t)] / ||w_initial + t * NN(t)||,
\`\`\`
which enforces norm 1 at every evaluated time (to floating point precision). It is distinct from the previous soft penalty and remains differentiable.

The global-target adjoint is computed using the **known exact small-system propagator**, which makes this an optimization proof-of-concept rather than an efficient solver.

## Measured means across five seeds

| Temperature | Objective | Terminal occupation error | Full trajectory RMSE | Max norm drift | Mean training seconds |
|---:|---|---:|---:|---:|---:|
| .03 | standard piecewise, unconstrained | .02448654 | .03139916 | .04546916 | .775 |
| .03 | global-terminal, unconstrained | .00462138 | .03862363 | .06308908 | .916 |
| .03 | standard piecewise, hard norm | .05564828 | .05698897 | 7.55e-16 | .839 |
| .03 | **global-terminal, hard norm** | **.01289247** | .06955950 | **6.44e-16** | 1.106 |
| .30 | standard piecewise, unconstrained | .02462657 | .03048765 | .04546916 | .653 |
| .30 | global-terminal, unconstrained | .00506530 | .03829547 | .06793354 | .906 |
| .30 | standard piecewise, hard norm | .03760861 | .05577914 | 7.55e-16 | .854 |
| .30 | **global-terminal, hard norm** | **.00868207** | .06466156 | **6.88e-16** | 1.109 |

### Interpretation

1. The hard unit-norm architecture gives machine-precision norm conservation for every sampled neural trajectory.
2. At low T, global-terminal hard norm improves terminal error by 76.8% relative to standard **hard norm** (0.05565→0.01289). At high T, it improves by 76.9% (0.03761→0.00868).
3. The global-terminal hard norm also improves terminal error over the **unconstrained** standard baseline, but by less than the unconstrained global-terminal loss.
4. Full-trajectory RMSE remains **worse** than the standard PINN, which means the model is **not** a more accurate all-purpose quantum simulator. Unit norm is not a replacement for valid quantum dynamics or complete positivity of an arbitrary dynamical map.
5. The exact 21×21 direct quantum propagator is much cheaper than neural training, so **no computational benefit** has been demonstrated.
6. Only the fixed Hamiltonian was tested in the hard-norm experiment; the previous unconstrained parameter sweep showed strong sensitivity to dot energy and coupling.

## Next scientific gate

Test the hard-norm global goal on preregistered shifted Hamiltonian parameters, including a regime where previous unconstrained methods badly failed, with fixed computational budgets and strong error controls. This is necessary before using the approach in interacting, strongly non-Markovian DQME calculations.

**No original publication-level result is claimed.** Exact wave normalization and goal-oriented adjoints are established techniques; their combination on this small toy is an optimization study.
