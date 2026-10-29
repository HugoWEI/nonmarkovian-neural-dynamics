# Verified correlated HEOM physicality diagnostics at hierarchy depths 1 and 2

**Date:** October 30, 2026  
**Depth-1 run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40973286595  
**Depth-2 run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40973468367

All numeric entries below were extracted from the respective GitHub Actions job logs using the connected GitHub tool. They describe an independently prepared, **full-ADO unbiased HEOM steady state**, followed by a voltage quench to mu_R=0.05 meV, with mu_L=0. They are **NOT** the authors' stored initial conditions.

| T/Gamma | Nk | depth | ADOs | steady n_up | terminal post-quench n_up | smallest post-quench eigenvalue |
|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 3 | 1 | 33 | 0.500000 | 0.503559720 | +0.07595268 |
| 3.0 | 3 | 2 | 529 | 0.500000 | 0.503339310 | +0.10831708 |
| 0.3 | 3 | 1 | 33 | 0.500000 | 0.502186444 | +0.001877567 |
| 0.3 | 3 | 2 | 529 | 0.500000 | 0.502793239 | +0.06185195 |
| 0.3 | 7 | 1 | 65 | 0.500000 | 0.502809930 | **-0.002674824** |

**Interpretation:** At low T and published source Nk=7, the depth-1 quench is **nonphysical** (a negative reduced-density-matrix eigenvalue) despite trace conservation and stationary unbiased initial evolution. At low Nk=3, increasing depth from 1 to 2 improves the positivity margin. But a different Nk is a different bath approximation; **Nk=3 depth-2 cannot be substituted for source Nk=7 depth-2**.

All verified calculations retain trace to approximately floating-point precision and show stationarity of the unbiased full ADO state at <=6e-16 occupation drift. These basic checks are necessary but not sufficient for physical/numerical convergence.

The source authors' original stored occupations (n_up ≈0.113 at low T and 0.297 at high T) do not match our independently prepared equilibrium n_up≈0.5; therefore these are **not reproductions of the published state**. Any claim otherwise would be incorrect.

## Next numerical gate

Run **low T, Nk=7, depth=2** with a controlled resource budget to see if the source-order, correlated quench becomes positive. If successful, carry out full trajectory difference calculations between depth2 and higher depth; then increase Nk beyond source order to evaluate spectral decomposition convergence. Direct fitting and model validation remain prerequisites for neural training.
