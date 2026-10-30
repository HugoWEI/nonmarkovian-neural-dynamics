# Interacting sparse HEOM adjoint: verified parameter sensitivities and truncation instability

**Date:** October 30, 2026  
**Verification runs:** [original three-case gradient checks](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40982217172); [expanded matrix with shared adjoint](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40982494292).

We solve a truly *interacting* spinful Anderson impurity with U=1 meV, one Lorentzian fermionic lead per spin, T/Gamma=0.3 or 3, full correlated HEOM steady state at mu=0, and a post-preparation quench to mu=0.05 meV over 6 ps.

The sensitivity is of the final spin-up occupation **holding the correlated initial hierarchy fixed**. It is NOT the total parameter derivative of a separately re-equilibrated initial steady state. The source authors' original two-lead SIAM correlated hierarchy is not reproduced.

## Identity and implementation

With the finite truncated HEOM generator L(p), y'=Ly and O(T)=Re(c* y(T)), a perturbation p at fixed y(0) gives

\[
\partial_p O(T)=\operatorname{Re}\int_0^T z(t)^\dagger (\partial_p L)y(t)\,dt,\quad
z(t)=e^{L^\dagger(T-t)}c.
\]

We access QuTiP's sparse \`solver.rhs(0)\`, use sparse Krylov matrix-exponential **actions** (not full exact many-body diagonalization), and validate the adjoint by independent central finite differences of the output at fixed ADO state. QuTiP matrix construction is an exact *finite-hierarchy* operation; hierarchy-depth and Padé truncation remain open.

The forward and backward trajectories are now computed **once per parameter family** and reused for derivatives of multiple parameters. This optimization is established adjoint sensitivity analysis, not a new quantum machine-learning result.

## Verified derivatives, in units of occupation per meV

| T/Gamma | Nk | depth | ∂mu n_up(6ps) | ∂U n_up(6ps) |
|---:|---:|---:|---:|---:|
| 3.0 | 3 | 2 | +0.21672207 | -0.11977835 |
| 0.3 | 3 | 2 | +0.01779966 | -0.02235125 |
| 0.3 | 3 | 3 | +0.04905146 | -0.04025730 |
| 0.3 | 3 | 4 | +0.04565670 | -0.03820841 |
| 0.3 | 5 | 3 | +0.08003517 | -0.05331352 |
| 0.3 | 7 | 3 | +0.08102976 | -0.05372393 |

The source-order low-T Nk7 depth3 finite-difference checks give:
- mu: 0.0810329801, absolute discrepancy 3.22e-6 (~0.004%).
- U: -0.0537393655, absolute discrepancy 1.54e-5 (~0.029%).

**Key finding:** *Correct for a finite numerical model* does not mean *physically converged*. In the low-T Nk3 model, depth2→3 changes the mu sensitivity by 176%, whereas depth3→4 changes it by ~6.9%. At fixed depth3, Nk3→5 changes the mu sensitivity by ~63%, and Nk5→7 by only ~1.24%. These variations are far more concerning than a terminal occupation alone would suggest.

## Compute cost (hosted CPU, Nk7 depth3)

- Complete correlated ADO steady-state preparation: **49.42 s**.
- One shared sparse forward + backward trajectory (81 time samples): **3.70 s**.
- Each additional adjoint contraction with a parameter derivative: **0.09–0.10 s**, after constructing the perturbed generators.
- Independent central finite differences of final occupation: **~2.0 s per parameter**, after constructing perturbed generators.

A shared adjoint is attractive for many parameters/residual evaluations, but there is **no end-to-end learning speedup**, and the upfront HEOM reference preparation remains the largest measured expense.

## Next falsification gate

Test *compressed* or otherwise approximately evaluated duals, with an error bound on the **discrete contraction** and cost accounting that includes the full adjoint precomputation. Compare against direct finite differences and a converged interacting HEOM baseline. Do not claim a new learning method merely from a faster derivative evaluation.

**This note reports a verified sensitivity-diagnostic result, not an original neural solver or a publication-worthy theorem.**
