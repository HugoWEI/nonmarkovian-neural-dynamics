# Low-temperature interacting correlated HEOM depth4 pilot

**Run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40980361654  
**Date:** October 30, 2026  
**Verification:** GitHub job logs and downloaded trajectory/result JSON artifact independently checked.

Independent source-inspired one-lead spinful Anderson impurity with U=1meV, thermal Lorentzian fermionic leads (one per spin), full steady-state ADOs at mu=0, quench to mu=0.05meV, t=0..6ps, T/Gamma=0.3.

We deliberately use the smaller **Padé Nk=3** to afford a deeper hierarchy; the source's low-T Nk=7 is *not* used in this depth4 run.

| Nk | Depth | ADOs | Terminal n_up | Steady-state CPU sec | Min density eigenvalue |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 137 | 0.500514945 | 0.031 | +0.032605 |
| 3 | 3 | 697 | 0.502172746 | 0.177 | +0.036513 |
| 3 | 4 | 2517 | 0.501998121 | 4.392 | +0.035900 |

Maximum occupation difference across the **entire** 6ps trajectory:
- depth2 vs depth3 at Nk3: 0.001657801.
- depth3 vs depth4 at Nk3: **0.000174625** (RMS 0.000107762).

So the depth error drops by about **9.5×** between these adjacent comparisons. This supports convergence of the reduced Padé-order hierarchy but **does not** imply convergence for source low-T Nk7, because increasing Nk from 5 to 7 at depth3 changes the final occupation by ~4.98e-5 yet the depth2/3 discrepancy at Nk7 remains ~0.001586.

All tested reduced states are positive and trace-conserving, and the unbiased ADO state is stationary, but those are necessary rather than sufficient for an accurate physical reference.

**Conclusion:** a tractable interacting correlated HEOM hierarchy can be made substantially more accurate by increasing depth, while low-temperature source-order Nk7 may require a much larger computation for the same depth criterion. This is a *reproducible numerical benchmark*, not a novel solver or a neural speedup. Next choose a physically defensible convergence/error target before developing an approximate adjoint.
