# Interacting correlated low-temperature HEOM at source Padé order Nk=7

**Date:** October 30, 2026  
**Run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40980060182  
**Verification:** Both depth2 and depth3 jobs completed. GitHub job logs inspected; numerical JSON artifact independently downloaded and parsed.

Physical model: spinful Anderson impurity, \(\epsilon_\uparrow=\epsilon_\downarrow=-0.5\)meV, Hubbard \(U=1\)meV, **one Lorentzian lead per spin** (\(\Gamma=0.1\)meV, \(W=5\)meV), \(T/\Gamma=0.3\). Prepare full *correlated* HEOM steady-state ADOs at mu=0, apply mu=0.05meV quench to that lead at t=0, evolve 6ps.

The model is source-**inspired** and has source-like Nk=7 low-temperature Padé order, but it is **NOT** the original two-lead SIAM initial hierarchy in Cao et al.

## Verified Padé and hierarchy matrix

| Nk | depth | ADOs | terminal n_up | steady-state preparation seconds | minimum eigenvalue |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 137 | 0.500514945 | 0.031 | +0.032605 |
| 3 | 3 | 697 | 0.502172746 | 0.183 | +0.036513 |
| 5 | 2 | 301 | 0.502139157 | 0.069 | +0.023846 |
| 5 | 3 | 2325 | 0.503728900 | 4.499 | +0.027499 |
| 7 | 2 | 529 | 0.502192614 | 0.119 | +0.023404 |
| 7 | 3 | 5489 | 0.503778679 | **47.928** | +0.027049 |

## Independent full-trajectory comparisons

| Padé Nk | Max depth-2/depth-3 occupation difference over 6ps | RMS difference |
|---:|---:|---:|
| 3 | 0.001657801052 | 0.000932920134 |
| 5 | 0.001589743916 | 0.000896544107 |
| 7 | **0.001586065808** | **0.000894363618** |

At fixed depth3, the final occupation moves by +0.001556154 when Nk=3→5, but only +0.000049779 when Nk=5→7. Thus **Padé refinement appears substantially closer to convergence**, while depth3 remains visibly different from depth2. There is NO certified hierarchy convergence yet.

The unbiased HEOM ADO state is stationary within ~1e-12 occupation drift at Nk7 depth3. Post-quench reduced states have positive sampled eigenvalues and preserve trace to roundoff. These facts validate basic physical consistency, **not convergence**.

### Computational cost

The actual low-temperature Nk7 depth3 steady state required ~48 CPU seconds to prepare, compared to ~0.12 seconds at depth2, and evolution of the 6ps trajectory took ~0.65 seconds. Thus **steady-state hierarchy preparation**, not time propagation, dominates this benchmark's measured CPU cost.

A further depth increase to 4 entails a much larger fermionic ADO hierarchy; naive sparse steady-state factorization may become costly or infeasible. If the research targets many repeated physical runs/observables/parameters, a learned surrogate might become *plausibly* useful, but no speedup has been measured and training costs may outweigh benefits.

## Next rigorous research gate

- Check integrator tolerance and longer-time sensitivity.
- Try a manageable *depth4* benchmark at a reduced Padé order before attempting source Nk7 depth4.
- Establish a controlled deeper reference or an independent hierarchy-extrapolation error bound.
- Only then design approximate error-backflow-aware neural training that avoids assuming an exact, cheap physical propagator.
- Compare against already-published piecewise PINNs, HEOM and goal-oriented neural error control.

**No original neural physics result or publication-grade novelty is claimed.**
