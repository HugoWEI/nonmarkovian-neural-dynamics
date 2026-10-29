# CPU validation results — 2026-10-29

**Execution location:** ChatGPT container, Python CPU runtime. **Not Google Colab.**

Command-equivalent notebook cells executed in this environment:
```bash
python scripts/run_toy.py
python -m pytest -q
```

| Memory decay lambda | Exact observable perturbation | Adjoint residual integral | Absolute identity discrepancy |
|---|---:|---:|---:|
| 2.0 | 0.03 | 0.029999999999667553 | 3.3244587638314727e-13 |
| 0.2 | 0.03 | 0.029999999999979158 | 2.084096784038536e-14 |
| 0.02 | 0.03 | 0.02999999999998906 | 1.0939166239509746e-14 |

**Tests:** 5 passed (pytest; approximately 0.44 seconds).

**Runtime:** NumPy 2.3.5; SciPy 1.17.0; PyTorch 2.10.0+cpu; GPU not available.

## Interpretation

The linear exponential-memory toy validates its own adjoint-residual consistency identity over three regimes. The perturbed trajectory is artificially prescribed, not trained by a neural network. This does not reproduce the low-temperature fermionic PINN-DQME benchmark, establish a physical solver advantage, or prove research novelty.

## Next milestone

See [Issue #1](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/issues/1) for the required reproduction of the published low-temperature error accumulation against a converged reference.
