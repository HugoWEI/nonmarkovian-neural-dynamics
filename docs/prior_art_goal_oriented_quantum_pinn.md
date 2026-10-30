# Prior art: goal-oriented and structure-preserving neural quantum dynamics

**Literature check: October 30, 2026.** This is a targeted rather than exhaustive systematic review. Do not use a claim of novelty based on this note alone.

## Strongest direct methodological competitors

1. **Multigoal-oriented dual-weighted-residual error estimation using PINNs** (2025), https://link.springer.com/article/10.1007/s44379-025-00012-4. Combines primal/adjoint PINNs and a posteriori goal-oriented DWR estimates across several target functionals. This establishes that **goal-specific PINN error weighting is not novel**.
2. **Time-Dependent Neural Galerkin Method for Quantum Dynamics**, Phys. Rev. Lett. 136, 120402 (March 2026), https://doi.org/10.1103/kqvx-dl54. Neural quantum dynamics from a global-in-time variational principle. Global temporal structure itself is not new.
3. **Simulating non-Markovian open quantum dynamics by exploiting physics-informed neural network**, J. Chem. Phys. (2026), https://pubmed.ncbi.nlm.nih.gov/42138408/. Time-domain-decomposed PINN-DQME for single-impurity Anderson physics, reporting low-T strong-memory error accumulation. The actual motivating published failure.
4. **Learning and inverting driven open quantum systems via physics-informed neural networks**, Machine Learning: Science and Technology (March 2026), https://doi.org/10.1088/2632-2153/ae4b84. Tests neural quantum master-equation solutions and parameter robustness.
5. **Noether-constrained PINNs for the nonlinear Schrödinger equation**, Physics Letters A (September 2026), https://doi.org/10.1016/j.physleta.2026.131802; and **Structure-preserving conservative phase-amplitude PINNs**, Journal of Computational and Applied Mathematics (October 2026), https://doi.org/10.1016/j.cam.2026.117607. Conservation-constrained and structure-preserving training is established.
6. **Physics-Informed Neural Networks for Solving the 2D Time-Dependent Schrödinger Equation with Norm Preservation** (SSRN preprint, April 2026), https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6580129. Norm preservation in neural Schrödinger solvers is already explicitly proposed.

## What our benchmark actually shows

Our independently verified hosted five-seed tests demonstrate a **target-specific optimization effect**: forecasting final occupation in the adjoint loss can lower terminal occupation error compared to uniform PINN in a 21-dimensional, exactly solvable quadratic quantum model, while increasing total trajectory error. Hard wavefunction normalization can ensure norm 1 but does not ensure correct quantum channels.

This is *not* a computational improvement: it explicitly uses the exact small-system propagator to build goal sensitivities; diagonalization is much cheaper than neural training. The result is too close to prior art to be described as a new PINN methodology.

## Genuine unmet standard for a publishable quantum contribution

A future method would need all of:
- a carefully source-matched interacting, strongly non-Markovian finite-temperature benchmark and fully converged reference;
- a materially better trade-off of **physical-observable error versus total cost** than time-domain PINN and strong non-neural solvers (HEOM, pseudomodes, tensor networks);
- valid quantum states/maps as appropriate, avoiding hidden violations from memory truncation or neural parametrization;
- robustness across sufficiently distinct physical parameters, long times and meaningful observables;
- mathematically justified use of approximated error-backflow sensitivities that does not assume the exact propagator is already known more cheaply;
- a renewed detailed theorem/method prior-art review.

The result may still be a good portfolio artifact documenting careful physical numerics and negative tests. It should not yet be called an original physics research finding.
