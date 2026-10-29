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

## Research milestones

1. Pin upstream NQS-DQME code and reproduce a published low-temperature benchmark.
2. Construct and converge a consistent HEOM reference for physical observables.
3. Derive observable-specific error sensitivities and implement a neural training objective.
4. Compare against uniform PINN training, conventional adaptive sampling, and non-neural methods.
5. Report observable error, long-time horizon, physical validity, statistical spread and total cost (including training).

See [experiment protocol](docs/experiment_protocol.md) and [prior art](docs/literature.md).

Reference implementation: https://github.com/caolong-cn/NQS-DQME_fermions

The upstream README warns that its error-propagation paper data are not fully compatible with its current code. Do not claim reproduction until this is resolved. No upstream source has been copied into this repository; check the upstream license before reuse.
