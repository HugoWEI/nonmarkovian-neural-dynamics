# Verified GitHub Actions run #6 — continuum bath and HEOM audit

**Run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40972267107  
**Commit:** `126e7cdfe82060ca5b8bb6a09ee0ba2b63b74347`  
**Date:** October 29–30, 2026 (Europe/Paris).  
**Evidence:** Read via GitHub connector `fetch_workflow_run_jobs` and both `fetch_workflow_job_logs`, not inferred from the green badge.

Both jobs concluded **success**; both produced a downloadable artifact.

## Published bath reconstruction

| Temperature | Largest coefficient error | Largest decay-rate error |
|---|---:|---:|
| T/Gamma=3.0 | 2.220e-16 | 8.882e-16 |
| T/Gamma=0.3 | 6.661e-16 | 8.882e-16 |

These compare the independent Lorentzian-Padé formula with the original pinned coefficient files. They establish source-correlation agreement to floating-point precision; not a dynamics reproduction.

QuTiP's built-in and independent fermionic bath correlations were compared for both C+ and C-, across mu=0 and 0.05 meV:

| Temperature | Worst C+/C- absolute error |
|---|---:|
| T/Gamma=3.0 | 5.5511e-17 |
| T/Gamma=0.3 | 1.1102e-16 |

## Initial HEOM smoke test

**IMPORTANT:** The HEOM calculation below was made with **Nk=1**, **hierarchy depths 1 and 2**, and an initially empty dot in a short-time interval t<=1.5 ps. It does **not** use the full fitted Padé orders Nk=3 and Nk=7, nor the original correlated initial ADO state.

| T/Gamma | Nk | depth 1 n_up(t_end) | depth 2 n_up(t_end) | Difference |
|---|---:|---:|---:|---:|
| 3.0 | 1 | 0.220628077303 | 0.210824278751 | 0.009803798552 |
| 0.3 | 1 | 0.186257803706 | 0.181453166766 | 0.004804636940 |

The spin-down occupations are identical to the spin-up occupations to numerical precision in this symmetric setup.

**The depth-1/depth-2 differences exceed the illustrative 0.001 threshold for both temperatures, so hierarchy-depth convergence has not been established.** Padé-order convergence is completely untested in these runs.

## Scientific conclusion

- The GitHub pipeline now **actually runs** source-correlation tests and QuTiP HEOM, with both artifacts uploaded.
- Both correlation comparisons passed to machine precision.
- The HEOM outputs are preliminary, not converged. A green Actions status is not proof of physical/numerical correctness.
- Next: source-equivalent initial conditions/ADO hierarchy, increasing Nk and depth independently, and comparing physical observables with original model.
- No neural improvement or novelty claim is supported by this run.
