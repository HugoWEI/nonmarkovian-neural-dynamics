# Upstream PINN-DQME reproduction audit — October 29, 2026

## Paper and implementation

- Cao et al., **Simulating non-Markovian open quantum dynamics by exploiting physics-informed neural network**, arXiv:2603.08081.
- Upstream repository: https://github.com/caolong-cn/NQS-DQME_fermions
- Inspected upstream default-tree revision: `7fb6ffb6115db42b299cc5c5092b27bfa9050628` (from GitHub code search).
- Key files: `tutorials/pinn.md`, `src/nqsdqme/solver/pinn_solver.py`, `examples/pinn_case1_T=3.0/main_evolution.py`, `examples/case1_T=0.3/main_evolution.py`.

## Findings

The authors' paper studies a **finite-temperature single-impurity Anderson model** using a PINN applied to a fermionic dissipaton-embedded quantum master equation, with comparison against HEOM. It reports loss of accuracy in the low-temperature strong-memory regime.

The documented PINN tutorial provides a **time window 4.5 to 6**, using a **pre-existing neural checkpoint** (e.g. `rho_t_4.5_0.1`). The example script `examples/pinn_case1_T=3.0/main_evolution.py` also calls `torch.load('rho_BFGS_ll_tr_4.5_0.1')` and uses a fixed start time `t_0=4.5`. It cannot be treated as an initial-condition-free, complete end-to-end reproduction script.

The available `examples/case1_T=0.3/main_evolution.py` is not the low-temperature PINN code path. It imports `RBM` and `TDVP`, with checkpoint `para{count}`.

The upstream README warns that the error-propagation-paper data are not completely compatible with the current library. The README also labels the PINN link with the NQS arXiv ID, which requires checking against the paper itself.

## What this means

**We have NOT reproduced** the low-temperature PINN-DQME failure and have NOT established a converged HEOM baseline yet.

We instead implemented a **separate zero-temperature Lorentzian-reservoir qubit/pseudomode PINN benchmark**, with a 2x2 exact numerical reference. It is a physically meaningful, minimal open-system calculation, but not the same model or temperature regime as the paper. See `scripts/quantum_pinn.py` and `results/2026-10-29-independent-pinn.md`.

## Requirements for real reproduction

1. Identify all required upstream model inputs, table files, checkpoint provenance, and temperature-specific PINN cases.
2. Pin a working upstream release, establish whether existing checkpoints are included, and verify its license.
3. Reconstruct from published model initial conditions or reconstruct missing checkpoints; use the **same** Hamiltonian, bath decomposition, truncations, temperature, and physical observable as the published PINN.
4. Verify convergence against independent HEOM and report numerical tolerance/error budget, not only visual trajectory agreement.
5. Measure wall-clock and training cost before proposing physics-guided modifications.
