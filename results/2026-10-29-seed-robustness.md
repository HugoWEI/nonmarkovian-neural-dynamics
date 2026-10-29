# Preliminary initialization-robustness test (CPU, October 29 2026)

**Independent zero-temperature Lorentzian reservoir**, not the Cao et al. finite-temperature fermionic DQME model. The data here were computed in the local CPU environment. 2000 steps, PyTorch double precision, g=0.8, lambda=0.2, time horizon 10, same architecture and collocation grid.

| Seed | Uniform global RMSE | Adjoint global RMSE | Uniform terminal population error | Adjoint terminal error |
|---:|---:|---:|---:|---:|
| 1 | 0.00305660 | 0.03265909 | 0.000333658 | 0.000842321 |
| 2 | 0.00501294 | 0.04107316 | 0.000570531 | 0.000906955 |
| 3 | 0.00286270 | 0.01153595 | 0.000294046 | 0.000886326 |
| 7 | 0.00487193 | 0.01974500 | 0.000759691 | 0.000330229 |

**Interpretation:** The chosen adjoint-weighted loss worsens global population RMSE in all 4 seeds. Terminal population error improves for seed 7 but worsens for seeds 1, 2 and 3. We have **no convincing improvement** in the strong-memory case. This is a useful negative result for *this particular objective* and numerical setup, not a theorem ruling out physics-informed error weighting.

The target of the adjoint is terminal amplitude G(T) while the reported terminal observable is its square; optimizing the former does not necessarily optimize the latter. A real DQME target-specific adjoint must be derived for the chosen observable and checked against strong alternatives.

No publication-grade novelty, training speed benefit or published-paper reproduction is claimed.
