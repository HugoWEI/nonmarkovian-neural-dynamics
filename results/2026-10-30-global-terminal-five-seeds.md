# Full-horizon terminal-adjoint ablation — verified GitHub run

**Date:** 2026-10-30  
**Run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40977959682  
**Results:** 50 neural training runs; all five modes × five seeds × two temperatures completed. Four numerical regression tests passed, including residual-error reconstruction and piecewise jump testing.

## Experimental protocol

Quadratic **20-bath-mode spinless** fermionic toy, four sequential PINN time windows, 1,200 Adam steps, same network architecture, training collocation grid, seed set (1,2,3,7,11), and 6 ps horizon. Two temperatures: T=0.03 and T=0.3 meV. Goal is the dot occupation at the **final** time.

- uniform: ordinary equation-residual mean squared error;
- projected/integrated: adjoint targeted to each *local* window terminal occupation;
- global_projected/global_integrated: adjoint targeted to a *predicted global-terminal occupation*, by propagating the current window endpoint to T=6 ps with the known small-system quantum propagator.

This propagator already solves the small quadratic model! Consequently these experiments **do not demonstrate a competitive quantum solver**.

## Results: means across five independent seeds

| T (meV) | Mode | Terminal occupation absolute error | Full-trajectory RMSE | Max wavefunction norm drift | Mean train seconds |
|---:|---|---:|---:|---:|---:|
| .03 | uniform | .02448654 | .03139916 | .04546916 | 1.18194 |
| .03 | projected | .04072243 | .02314814 | .02986527 | 1.33176 |
| .03 | integrated | .03245474 | .02398904 | .03482463 | 1.32871 |
| .03 | **global_projected** | **.00462138** | .03862363 | .06308908 | 1.37482 |
| .03 | global_integrated | .00473394 | .03839381 | .06513039 | 1.37326 |
| .30 | uniform | .02462657 | .03048765 | .04546916 | 1.03339 |
| .30 | projected | .03945104 | .02310701 | .02815368 | 1.33068 |
| .30 | integrated | .03011523 | .02373642 | .03408535 | 1.32957 |
| .30 | **global_projected** | **.00506530** | .03829547 | .06793354 | 1.37104 |
| .30 | global_integrated | .00533676 | .03740629 | .06980907 | 1.37006 |

Relative reduction in mean **terminal occupation error**, global_projected versus uniform: 81.1% (low T), 79.4% (high T). This result is a **targeted optimization effect**, not an improved overall physical trajectory.

### Limitations

1. Only five seeds: indicative, not publication-grade confidence bounds.
2. The new loss increases norm drift and full-trajectory RMSE.
3. It uses explicit exact future propagation for a small quadratic Hamiltonian; direct diagonalization is cheaper.
4. It does not model continuum-bath HEOM, interacting correlated SIAM, or novel quantum error-backflow.
5. The loss is **global-terminal lookahead**, not the full exact defect integral: future network errors are ignored at the time the current window is trained.
6. The training uses one Hamiltonian family and fixed parameters; it is not evidence for transfer to unseen quantum systems.

### Next falsification test

Run prespecified weights and physical norm penalties, holding all optimizer steps and architecture fixed. Evaluate the Pareto trade-off of terminal occupation error, full-trajectory RMSE, norm drift and end-to-end runtime. Check whether improvements persist across meaningful changes in Hamiltonian parameters. The fundamental go/no-go remains a computational advantage on an interacting problem where direct exact propagation is expensive.
