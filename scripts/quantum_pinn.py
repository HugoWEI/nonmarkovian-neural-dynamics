"""Independent zero-temperature Lorentzian-reservoir PINN benchmark.

NOT a reproduction of Cao et al. (2026) finite-temperature fermionic PINN-DQME.
The excited amplitude G and bath pseudomode B obey G'=-gB, B'=gG-lam B.
The reduced-qubit excited population is |G|**2 and has a closed-form reference.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def matrix(g: float, lam: float) -> np.ndarray:
    if g <= 0 or lam <= 0:
        raise ValueError("g and lam must be strictly positive")
    return np.array([[0., -g], [g, -lam]], dtype=float)


def exact_states(times, g: float, lam: float) -> np.ndarray:
    a = matrix(g, lam)
    return np.array([expm(a * float(t)) @ np.array([1., 0.]) for t in times])


def benchmark(g=0.8, lam=0.2, horizon=10., steps=2000,
              ncoll=81, seed=7, mode="uniform"):
    import torch

    if mode not in ("uniform", "adjoint"):
        raise ValueError(mode)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    a_np = matrix(g, lam)
    a = torch.tensor(a_np, dtype=dtype)
    t = torch.linspace(0., horizon, ncoll, dtype=dtype).reshape(-1, 1)
    t.requires_grad_(True)
    x0 = torch.tensor([1., 0.], dtype=dtype)
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 2)
    ).to(dtype=dtype)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.005)

    weights_np = np.array([
        expm(a_np.T * (horizon - float(s))) @ np.array([1., 0.])
        for s in t.detach().numpy().ravel()
    ])
    weights = torch.tensor(weights_np, dtype=dtype)
    norm = weights.square().sum(dim=1).mean()
    start = time.perf_counter()

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Rebuild the t-dependent computational graph every iteration.
        u = t / horizon
        estimate = x0[None, :] + u * network(u)  # exact initial condition
        derivatives = torch.cat([
            torch.autograd.grad(
                estimate[:, j].sum(), t, create_graph=True, retain_graph=True
            )[0] for j in range(2)
        ], dim=1)
        residual = derivatives - estimate @ a.T
        uniform_loss = residual.square().sum(dim=1).mean()
        if mode == "adjoint":
            projected = (weights * residual).sum(dim=1)
            loss = 0.25 * uniform_loss + 0.75 * projected.square().mean() / norm
        else:
            loss = uniform_loss
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start
    times = np.linspace(0., horizon, 401)
    exact = exact_states(times, g, lam)
    with torch.no_grad():
        u = torch.tensor((times / horizon)[:, None], dtype=dtype)
        prediction = (x0[None, :] + u * network(u)).numpy()

    p = exact[:, 0] ** 2
    p_hat = prediction[:, 0] ** 2
    return {
        "mode": mode, "g": g, "lambda": lam, "horizon": horizon,
        "steps": steps, "seed": seed,
        "final_loss": float(loss.detach()),
        "population_rmse": float(np.sqrt(np.mean((p_hat - p) ** 2))),
        "terminal_population_abs_error": float(abs(p_hat[-1] - p[-1])),
        "runtime_seconds": elapsed,
        "exact_terminal_population": float(p[-1]),
        "pred_terminal_population": float(p_hat[-1]),
        "population_revival_total": float(np.maximum(np.diff(p), 0).sum()),
        "_curves": (times, p, p_hat),
    }


def run(output: Path, steps=2000):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    cases = [("long_memory", .8, .2), ("short_memory", .2, 2.)]
    for regime, g, lam in cases:
        for mode in ("uniform", "adjoint"):
            r = benchmark(g=g, lam=lam, steps=steps, mode=mode)
            times, exact_pop, pred_pop = r.pop("_curves")
            r["regime"] = regime
            results.append(r)
            with (output / f"{regime}_{mode}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "exact_population", "neural_population"])
                writer.writerows(zip(times, exact_pop, pred_pop))
            print(json.dumps(r), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Independent quantum PINN benchmark", "",
        "**Not a reproduction of the finite-temperature Anderson-model PINN-DQME paper.**",
        "",
        "Zero-temperature Lorentzian reservoir; exact two-dimensional pseudomode reference.",
        "Adjoint loss targets terminal amplitude, not global population RMSE.",
        "",
        "| Regime | Objective | Population RMSE | Terminal population error | CPU seconds |",
        "|---|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['regime']} | {r['mode']} | {r['population_rmse']:.5e} | "
            f"{r['terminal_population_abs_error']:.5e} | {r['runtime_seconds']:.2f} |"
        )
    lines += ["", "Single seed, no significance claim. A 2x2 matrix exponential",
              "is much cheaper than any of the neural models here.",
              "Revivals are a physical memory phenomenon, distinct from neural error backflow.",
              "Adjoint weighting has no guarantee of global accuracy improvement."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", default="outputs/quantum_pinn")
    args = parser.parse_args()
    run(Path(args.output), steps=args.steps)
