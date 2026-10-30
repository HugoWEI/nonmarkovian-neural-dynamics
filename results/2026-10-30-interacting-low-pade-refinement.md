# Low-temperature interacting one-lead Padé and hierarchy refinement

**Date:** October 30, 2026  
**Nk=3 workflow:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40979679933  
**Nk=5 workflow:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40979902895  
**Evidence:** GitHub numerical logs and independently downloaded results.json files.

Spinful single-impurity Anderson, U=1 meV, eps_up/down=-0.5 meV, **one** Lorentzian continuum reservoir per spin, Gamma=0.1 meV, W=5 meV, T/Gamma=0.3. Exact correlated *numerical HEOM equilibrium ADO hierarchy* at mu=0, then quench to mu=0.05meV for 6ps. Not the published two-lead SIAM state.

## Verified results

| Low T Nk | Depth | ADOs | Terminal n_up | Min density eigenvalue | Steady-state setup seconds |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 137 | 0.500514945 | +0.032605 | 0.031 |
| 3 | 3 | 697 | 0.502172746 | +0.036513 | 0.183 |
| 5 | 2 | 301 | 0.502139157 | +0.023846 | 0.069 |
| 5 | 3 | 2325 | 0.503728900 | +0.027499 | 4.499 |

The full-trajectory depth-2/3 maximum occupation discrepancies are 0.0016578011 at Nk=3 and 0.0015897439 at Nk=5. At fixed depth=3, raising Padé order Nk3→Nk5 changes final occupation by **0.001556154**. Thus both Padé and hierarchy corrections are still significant.

All trajectories retain trace at roundoff and sampled reduced density matrices are positive, but physicality does **not** certify convergence. The unbiased full ADO hierarchy is stationary to <3e-13 occupation drift.

## Interpretation and next gate

The source-inspired Padé expansion is underspecified at Nk3 for this low-T interacting quench. A source-order **Nk7**, depth2/3 pilot is configured in \`interacting-low-nk7.yml\`, with a strict 240-second numerical cap. Even that will not establish full HEOM convergence without a depth4, higher Padé, and integrator-tolerance study.

**No neural error-backflow improvement or numerical speed advantage has been demonstrated.** The purpose is to establish a valid physically nontrivial reference for a future comparison.
