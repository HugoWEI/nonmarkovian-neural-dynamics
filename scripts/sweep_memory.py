"""CPU-only sensitivity baseline for the exact linear memory toy.

The result is not a fermionic quantum simulation or deep-learning result.
"""
from pathlib import Path
import csv
import json

import numpy as np
from scipy.integrate import simpson

from nonmarkovian_neural_dynamics.toy import (
    adjoint_weights, check_adjoint_identity, terminal_observable
)


def sweep(output_dir: Path, *, t_end: float = 4.0, g: float = 1.0) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, t_end, 201)
    rows = []
    for lam in (0.02, 0.1, 0.2, 0.5, 1.0, 2.0):
        weights = np.array([adjoint_weights(float(t), t_end, g, lam) for t in grid])
        sensitivity = np.linalg.norm(weights, axis=1)
        identity = check_adjoint_identity(g=g, lam=lam, t_end=t_end)
        rows.append({
            "lambda": lam,
            "final_amplitude": terminal_observable(t_end, g, lam),
            "sensitivity_L1": float(simpson(sensitivity, x=grid)),
            "max_sensitivity": float(sensitivity.max()),
            "identity_error": identity["absolute_identity_error"],
        })
    with (output_dir / "memory_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "memory_sweep.json").write_text(json.dumps(rows, indent=2))
    summary = [
        "# Exponential-memory sensitivity sweep",
        "",
        "Exactly solvable linear toy; not a neural model or reproduction of fermionic DQME.",
        "",
        "| Memory decay | Terminal amplitude | Sensitivity integral | Identity error |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        summary.append(
            f"| {r['lambda']:.2f} | {r['final_amplitude']:.6f} | "
            f"{r['sensitivity_L1']:.6f} | {r['identity_error']:.2e} |"
        )
    summary.extend([
        "",
        "The sensitivity integral is the L1 norm of an adjoint weight and bounds",
        "the terminal-observable response to a bounded residual.",
        "The toy is dissipative and does not itself show error-backflow amplification.",
    ])
    (output_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    return rows


if __name__ == "__main__":
    for row in sweep(Path("outputs/toy_sweep")):
        print(json.dumps(row))
