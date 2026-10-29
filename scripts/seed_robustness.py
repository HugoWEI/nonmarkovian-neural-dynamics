"""Compare both neural objectives across seeds in the difficult-memory case."""
import argparse
import csv
import json
from pathlib import Path

from quantum_pinn import benchmark


def run(out: Path, steps: int = 2000, seeds=(1, 2, 3, 7)):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for mode in ("uniform", "adjoint"):
            result = benchmark(g=.8, lam=.2, horizon=10., steps=steps,
                               seed=seed, mode=mode)
            result.pop("_curves")
            rows.append(result)
    with (out / "seed_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / "seed_results.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# Strong-memory neural PINN seed robustness", "",
        "**Independent zero-temperature qubit/pseudomode benchmark; NOT the published fermionic PINN-DQME result.**", "",
        "Models train on the same collocation grid, optimizer, number of steps and seed-specific initialization.",
        "A goal-oriented loss can improve one observable while worsening others; no assumed winner.", "",
        "| Seed | Uniform global population RMSE | Adjoint global RMSE | Uniform terminal error | Adjoint terminal error |",
        "|---:|---:|---:|---:|---:|"
    ]
    for seed in seeds:
        u = next(r for r in rows if r["seed"] == seed and r["mode"] == "uniform")
        a = next(r for r in rows if r["seed"] == seed and r["mode"] == "adjoint")
        lines.append(
            f"| {seed} | {u['population_rmse']:.5e} | {a['population_rmse']:.5e} | "
            f"{u['terminal_population_abs_error']:.5e} | {a['terminal_population_abs_error']:.5e} |"
        )
    lines.extend([
        "", "Both methods use an exactly imposed initial condition and exact reference.",
        "This toy is solvable essentially for free without learning.",
        "Terminal amplitude is the adjoint target; we score its **squared modulus** (excited population).",
        "Single-horizon experiments cannot establish long-time efficacy in a many-body environment.",
    ])
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=2000, type=int)
    parser.add_argument("--output", default="outputs/seed_robustness")
    args = parser.parse_args()
    run(Path(args.output), steps=args.steps)
