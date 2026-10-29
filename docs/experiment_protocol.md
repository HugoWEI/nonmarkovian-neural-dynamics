# Experiment protocol (pre-results)

## Scientific question
Can physics-derived error-propagation sensitivities improve the cost-versus-accuracy tradeoff of deep-learning solvers for strongly non-Markovian fermionic impurity dynamics?

### Phase 0 — Mathematical sanity check
- Verify a linear adjoint-residual identity for an exponential-memory toy.
- This is a software validation, not a result in quantum impurity physics.

### Phase 1 — Reproduce literature
- Pin a specific upstream NQS-DQME Git commit and check its license.
- Resolve warning that error-propagation paper data are not fully compatible with current library.
- Reproduce published occupation/current with documented units and parameters.
- Converge HEOM hierarchy, bath expansion and solver tolerances; compare common observables.
- Record hardware, seeds, numerical truncation and wall-clock times.

### Phase 2 — Proposed experiment
- Uniform-residual PINN and ordinary adaptive residual sampling.
- Compute observable-specific adjoint or error-backflow sensitivities.
- Train goal-oriented neural loss with physically motivated weights.
- Compare simpler sensitivity/conditioning weights and larger model capacity.
- Account for the cost of calculating sensitivity information.

### Phase 3 — Robustness and validation
- Vary temperature, coupling, bath memory, time horizon and initial state.
- Track physical observables versus converged references.
- Check density-matrix positivity and trace where applicable.
- Do not claim CPTP dynamics merely from valid individual output states.
- Report computational cost including training, sampling and diagnostics.

## Go/no-go
Advance only if a new method improves physical-observable error at fixed total cost, or reduces total cost at a fixed error tolerance, beyond credible neural and non-neural baselines.
