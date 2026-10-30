# Error-Backflow-Aware Neural Dynamics

Exploratory research on goal-oriented deep learning for strongly non-Markovian open quantum systems.

**Status:** Initial mathematical checks only. No new deep-learning improvement or research novelty has yet been demonstrated.

## Hypothesis

Can physical error-propagation sensitivities focus neural-solver training on residuals that matter for quantum observables, improving reliable simulation horizon or total cost in low-temperature fermionic impurity benchmarks?

## Quick start (CPU, no GPU required)

```bash
python -m pip install -e '.[dev]'
python scripts/run_toy.py
python -m pytest -q
```

The first experiment verifies an **exact adjoint-residual identity** in a simplified linear exponential-memory model. This is a pedagogical mathematical check, **not** a fermionic DQME or a quantum hardware experiment.

## Independent quantum PINN validation (2026-10-29)

A **separate** zero-temperature Lorentzian-bath qubit/pseudomode benchmark is available at [`scripts/quantum_pinn.py`](scripts/quantum_pinn.py). It compares uniformly weighted PINN training with adjoint-weighted training against a two-state exact solution, and is intentionally **not** called a reproduction of the finite-temperature fermionic Anderson PINN-DQME study.

- [Initial four neural runs](results/2026-10-29-independent-pinn.md)
- [Four-seed robustness check](results/2026-10-29-seed-robustness.md): the present weighting worsened strong-memory global RMSE for all seeds; no new improvement established
- [Upstream reproduction audit](docs/upstream_reproduction_audit.md) and [missing checkpoint issue](../../issues/2)
- Automated runs: [quantum PINN](../../actions/workflows/quantum-pinn.yml) and [seed robustness](../../actions/workflows/seed-robustness.yml), both CPU-only

For local execution: `python -m pip install -e '.[research,dev]'` then `python scripts/quantum_pinn.py --steps 2000`.

## Phase 1: finite-temperature interacting fermion benchmark (2026-10-29)

A **finite-bath spinful Anderson impurity with Hubbard interaction U=1** is now implemented and checked against exact many-body unitary time evolution. The bath starts in a finite-temperature thermal state. Its small finite reservoir is **not equivalent to the continuum-bath fermionic DQME/HEOM model in Cao et al.**

- [Exact interacting reference](src/nonmarkovian_neural_dynamics/finite_anderson.py)
- [Full-interval fermionic PINN](scripts/finite_anderson_pinn.py) — [remote workflow](.github/workflows/finite-anderson.yml)
- [Time-domain-decomposed PINN](scripts/finite_anderson_piecewise.py) — [remote workflow](.github/workflows/piecewise-anderson.yml)
- [Detailed local results, assumptions, and limitations](results/2026-10-29-finite-anderson-reference.md)

In a single-seed test using 3000 optimization steps, the T=0.3 global model has occupation RMSE 0.238 on [0,5]; the five-window PINN achieves RMSE 0.00440. The time-domain decomposition is **published baseline methodology**, not a new research result. The two methods also use different collocation counts per step and are not controlled at perfectly equal wall-clock cost.

The outstanding task is still to reproduce the **published low-temperature fermionic PINN-DQME error** against a converged HEOM solution. See [missing-checkpoint blocker](../../issues/2).

## Phase 1B — Original continuum bath recovered (2026-10-29)

**Reproducible result:** We independently reconstructed the published Lorentzian-Padé bath coefficients for `T/Gamma=3.0` and `T/Gamma=0.3` from the physical parameters `Gamma=0.1 meV`, `W=5 meV`, `hbar=0.658211928 meV ps`. The slowest fitted decay times differ by a factor of 10. Coefficients match the pinned original files to numerical precision.

- [Padé derivation, checks, and source-matching results](results/2026-10-29-pade-source-match.md)
- [Source coefficient audit](results/2026-10-29-source-bath-check.md)
- [QuTiP continuum-bath HEOM code](scripts/fermionic_heom_reference.py)
- [Exact reproduction/convergence requirements](docs/continuum_heom_protocol.md)
- [CPU remote jobs](.github/workflows/continuum-heom.yml)
- [Open convergence task](../../issues/3)

**Limitations:** The current HEOM implementation is a source-anchored *intermediate* model. Full equivalence of chemical-potential phases, hierarchy construction, and Padé/HEOM convergence has not been verified; the upstream PINN tutorial still requires a missing checkpoint. No novel neural-solver performance has been demonstrated.

## Phase 1C — Source-state, lead-phase, and HEOM complexity audit (2026-10-30)

- [Original source-state and chemical-potential phase audit](results/2026-10-30-source-state-and-phase-audit.md): the original stored state is **not** an empty impurity, and the full correlated initial hierarchy is not recovered merely by copying four occupations.
- [Source-phase check](scripts/upstream_phase_check.py) and [QuTiP vs source correlation test](scripts/verify_quutip_correlations.py): validate both fermionic C+ and C- sectors at both source temperatures and lead chemical potentials.
- [Fermionic hierarchy growth analysis](scripts/heom_resource_envelope.py): source-matched exponent count at high/low temperatures is 32/64, giving 529/2081 ADOs at depth 2.
- [GitHub Actions HEOM and input audit](.github/workflows/continuum-heom.yml) now gates the dynamical calculation on bath-correlation matching and records original-state checks.

**Status:** These source checks and cloud test scripts have been committed; successful remote QuTiP execution, hierarchy convergence, and published PINN-DQME reproduction remain **unverified**. See [Issue #3](../../issues/3).

## Phase 1D — Verified convergence and long-time numerical reference (2026-10-30)

Cloud jobs are now fully inspectable via the connected GitHub interface: both exact bath checks and numerical HEOM runs post their run IDs to [Issue #3](../../issues/3).

- [Two-axis Nk/depth study](results/2026-10-30-heom-two-axis-run1.md)
- [Source-beyond-order Padé refinement](results/2026-10-30-verified-pade-refinement.md)
- [Correlated full-ADO voltage quench and physicality check](results/2026-10-30-correlated-physicality-depth2.md)
- [**6 ps long-time hierarchy comparison**](results/2026-10-30-longtime-depth2-depth3.md)
- [Rigorous observable-specific residual/adjoint identity](docs/observable_adjoint_identity.md)

**Main numerical finding:** at source Padé orders, full-trajectory depth-2 vs depth-3 occupation differences on [0,6 ps] were 0.000725 for high T and **0.005631 for low T**. The low-T discrepancy grows markedly from its 1.5-ps counterpart, so a depth-2 prediction is not an accurate long-time reference.

We have **not** reproduced the original correlated DQME initial state or demonstrated a novel neural solver. Before training a model, we need a deeper/converged reference or a reduced physical system allowing an exact independent benchmark. High low-T HEOM depth is computationally expensive (679,121 ADOs at depth 4 for source Nk=7).

## Phase 2 — Reduced spinless fermionic benchmark, cross-validated (2026-10-30)

We now have an independent *quadratic fermion* physical reference and a one-bath QuTiP HEOM implementation for a spinless resonant-level model; the physical parameters and thermal bath are matched across both methods.

- [Full results and important limitations](results/2026-10-30-reduced-fermionic-benchmark.md)
- [Quadratic exact reference](src/nonmarkovian_neural_dynamics/exact_rlm.py)
- [Source-type HEOM depth sweep](scripts/spinless_rlm_heom.py) — [cloud workflow](.github/workflows/reduced-rlm-heom.yml)
- [Higher-order Padé comparison](scripts/spinless_pade_refinement.py) — [cloud workflow](.github/workflows/reduced-rlm-pade.yml)
- [Small finite-bath piecewise neural PINN ablation](scripts/reduced_fermionic_pinn.py) — [cloud workflow](.github/workflows/reduced-pinn.yml)

**Verified numerical result:** over 6 ps, low-temperature HEOM depth-5 at source-type Nk=7 differs from the independent exact spinless result by at most 1.93e-4; increasing Nk to 13 at depth 4 reduces this to 5.87e-6, close to spectral cutoff sensitivity. High T likewise improves to ~6e-6 at higher Padé order. The spinless reduced model avoids the large long-time hierarchy error found for our earlier interacting spinful model.

**Neural caution:** Early local goal-oriented adjoint objectives improve mean full-trajectory RMSE but worsen the terminal occupation they are meant to target; no new neural method has yet proven superior. The neural baseline's GitHub execution status must be checked separately.

**This is not a reproduction of the authors' correlated interacting SIAM DQME model.**

## Phase 2B — Global-terminal neural error control (2026-10-30)

- [Exact residual/observable identity, window jumps, and finite-difference gradient verification](docs/global_terminal_observable_control.md)
- [Verified 50-run multi-seed global-target ablation](results/2026-10-30-global-terminal-five-seeds.md) — [GitHub Actions run](../../actions/runs/40977959682)
- [Global-goal cloud workflow](.github/workflows/global-goal-ablation.yml)
- [Preregistered norm/accuracy trade-off workflow](.github/workflows/global-norm-pareto.yml)
- [Phase 2 research-go/no-go issue](../../issues/4)

**Observed:** In an exactly solvable 20-mode finite-bath model, explicitly forecasting the *global terminal occupation* reduces mean terminal error by 81.1% (T=0.03 meV) and 79.4% (T=0.3 meV) relative to uniform-residual piecewise PINN across five seeds. But full-trajectory RMSE and wavefunction norm drift get worse, and the method uses a known exact quadratic propagator. **This is an optimization diagnostic, not a computational physics breakthrough or novel algorithm.**

## Phase 3 — Genuine interacting quantum-memory benchmark (2026-10-30)

After the quadratic neural objective failed Hamiltonian-robustness and compute-cost checks, the next benchmark keeps the **spinful Hubbard interaction U=1 meV** and reduces to one fermionic lead per spin. It prepares a **complete correlated HEOM steady-state hierarchy**, then performs a chemical-potential quench.

- [Verified 6 ps interacting HEOM result](results/2026-10-30-interacting-onelead-correlated-heom.md)
- [Reproducible solver implementation](scripts/interacting_onelead_heom.py)
- [Hosted runs and artifacts](.github/workflows/interacting-onelead.yml)
- [Low-T Nk5 refinement workflow](.github/workflows/interacting-low-nk5.yml)
- [Phase 3 research checklist](../../issues/5)
- [Overall scientific status and negative results](docs/research_status_2026-10-30.md)
- [Hard-norm neural failures under parameter changes](results/2026-10-30-unitnorm-hamiltonian-stress-test.md)

At Nk=3, increasing depth from 2 to 3 changes occupation by at most **6.04e-5 at T/Gamma=3** versus **1.66e-3 at T/Gamma=0.3**, on the 6 ps correlated quench. This is a *genuine interacting numerical benchmark*, not the authors' two-lead initial state nor a converged low-temperature target. **No original neural solver improvement is yet demonstrated.**

## Phase 4 — Interacting hierarchy backflow, adjoints and memory correction (2026-10-30)

Validated on a **genuinely interacting** source-inspired one-lead Anderson model, with correlated depth2/3 HEOM initial states and fixed quench dynamics:

- [Verified sparse interacting adjoint sensitivities](results/2026-10-30-interacting-adjoint-gradient-audit.md): gradients with respect to lead chemical potential and Hubbard U independently checked against finite differences. Low-T source-order Nk7 depth3 ∂μ n_up(6ps) ≈ 0.08103 per meV. Sensitivities can be much more hierarchy- and Padé-sensitive than occupations.
- [Compressed reusable adjoints](results/2026-10-30-compressed-interacting-heom-dual.md): rank8 compresses sampled dual by ~10x storage and reproduces two tested gradients to ~1e-7; no demonstrated end-to-end speedup.
- [Exact observable-specific hierarchy backflow decomposition](results/2026-10-30-interacting-hierarchy-backflow-identity.md): at low T Nk7, true depth2–depth3 occupation error -0.00139136 decomposes into ~20% omitted initial correlated ADOs and ~80% dynamic omitted-tier feedback; verified to ~4e-11 quadrature error.
- [First-order omitted-tier memory feedback](results/2026-10-30-first-order-memory-feedback.md) and [parameter/cost audit](results/2026-10-30-memory-voltage-robustness.md): corrects ~73–77% of 6ps low-T terminal error, with only **~15–20% faster paired CPU time** under the same depth2-prepared state. At 12ps low-T improvement drops to ~36%.
- [New learned higher-order closure experiment](.github/workflows/interacting-neural-closure.yml): train a small network on a **12ps interacting HEOM** parameter grid to learn only the residual missing after the classical first-order memory approximation; evaluate held-out and extrapolated quench/Hubbard parameters against **classical interpolation** and direct high-depth HEOM.
- [Phase 3/4 tasks and research caveats](../../issues/5).

**Do not claim novelty yet.** Schur-complement HEOM terminators, memory-kernel projections and learned Volterra kernels have close prior art, including Vadimov (SciPost Phys. Core 2026) and Adriazola–Roszak (Phys. Rev. A accepted September 2026). The first-order approximation is established physics; the new neural experiment is a *falsification test* that must beat strong classical baselines at complete amortized cost. The shared low-depth initial state has no nonzero ADOs in omitted higher tiers; it does not reproduce the original paper's full two-lead correlated state.

## Research milestones

1. Pin upstream NQS-DQME code and reproduce a published low-temperature benchmark.
2. Construct and converge a consistent HEOM reference for physical observables.
3. Derive observable-specific error sensitivities and implement a neural training objective.
4. Compare against uniform PINN training, conventional adaptive sampling, and non-neural methods.
5. Report observable error, long-time horizon, physical validity, statistical spread and total cost (including training).

See [experiment protocol](docs/experiment_protocol.md) and [prior art](docs/literature.md).

Reference implementation: https://github.com/caolong-cn/NQS-DQME_fermions

The upstream README warns that its error-propagation paper data are not fully compatible with its current code. Do not claim reproduction until this is resolved. No upstream source has been copied into this repository; check the upstream license before reuse.
