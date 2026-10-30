# Interacting full-correlated-ADO one-lead HEOM, depths 2 versus 3

**Run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40979679933  
**Date:** October 30, 2026  
**Source:** GitHub workflow logs **and** independently downloaded JSON/trajectory artifacts (\`interacting-correlated-high\`, \`interacting-correlated-low\`).

This is a genuinely **interacting** spinful Anderson impurity, \(U=1\)meV, \(\epsilon_\uparrow=\epsilon_\downarrow=-0.5\)meV, a single Lorentzian fermionic lead per spin, \(\Gamma=0.1\)meV, \(W=5\)meV. We create a **complete correlated HEOM auxiliary hierarchy** with \`steady_state()\` at \(\mu=0\), and quench the reservoir chemical potential to \(0.05\)meV for \(t\in[0,6]\)ps.

It is **NOT** the Cao et al. two-lead SIAM with its authors' stored correlated initial hierarchy.

## Verified outcomes (Padé Nk=3)

| T/Gamma | Depth | ADOs | Initial n_up | Terminal n_up | Minimum density eigenvalue | Steady state drift |
|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 2 | 137 | 0.5000000 | 0.510815681 | +0.0868117 | 7.84e-14 |
| 3.0 | 3 | 697 | 0.5000000 | 0.510876052 | +0.0872786 | 1.70e-13 |
| 0.3 | 2 | 137 | 0.5000000 | 0.500514945 | +0.0326050 | 7.39e-14 |
| 0.3 | 3 | 697 | 0.5000000 | 0.502172746 | +0.0365125 | 1.14e-13 |

All sampled reduced states are positive at these truncations; traces agree with 1 to numerical precision. The complete unbiased ADO state remains stationary when evolved under its own Liouvillian before the voltage quench.

**Full-trajectory depth-2/depth-3 differences (parsed from downloaded JSON):**

- High T=3.0: maximum occupation difference **6.037090864e-5**, RMS **3.589289584e-5**.
- Low T=0.3: maximum difference **1.657801052e-3**, RMS **9.329201345e-4**.

This is promising because the high-T result is much closer to depth convergence. The low-T result is still appreciably depth-sensitive. At T/Gamma=0.3, Nk=3 is also *below* the original source's Nk=7 Padé order, so one should **not** use this as a converged neural-learning target yet.

### What is new scientifically?

The computation is an independently reproducible **reference benchmark**, not a new theorem, new quantum simulator, or new neural loss. The associated physics and numerical methods (HEOM/Padé/steady ADOs) are established.

### Next gate

Increase low-temperature Padé order while holding physical model fixed (starting with Nk=5, then source-like Nk=7 where feasible), check depth and integration convergence, and only then test a neural model on the *interacting system*. The neural loss must not take an exact inexpensive propagator as input, unlike our quadratic toy ablation.
