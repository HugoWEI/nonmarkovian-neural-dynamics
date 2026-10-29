"""Finite-temperature interacting Anderson PINN (finite-bath benchmark).

A thermal mixture of fermionic Fock states is propagated as weighted pure
components with a time-encoded neural network. This is NOT PINN-DQME/HEOM.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from nonmarkovian_neural_dynamics.finite_anderson import (
    anderson_hamiltonian, thermal_diagonal, exact_evolution,
)


def train(temperature=0.3, interaction=1.0, steps=3000, seed=7,
          t_final=2.0, width=48, n_collocation=81):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    energies, couplings = (-0.7,), (0.55,)
    h_np = anderson_hamiltonian(energies, couplings, interaction=interaction)
    h = torch.tensor(h_np.real, dtype=dtype)
    probs = thermal_diagonal(energies, temperature)
    configs = np.flatnonzero(probs > 1e-15)
    n_config, dim = len(configs), len(probs)
    initial = np.zeros((dim, n_config))
    initial[configs, np.arange(n_config)] = np.sqrt(probs[configs])
    psi0 = torch.tensor(initial, dtype=dtype)

    collocation = torch.linspace(0., t_final, n_collocation,
                                 dtype=dtype).reshape(-1, 1)
    network = torch.nn.Sequential(
        torch.nn.Linear(1, width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), torch.nn.Tanh(),
        torch.nn.Linear(width, 2 * dim * n_config)
    ).to(dtype=dtype)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)

    def evaluate(t):
        tau = t / t_final
        correction = (t * network(tau)).reshape(-1, 2, dim, n_config)
        return correction[:, 0] + psi0[None], correction[:, 1]

    delta = 1e-3
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        re, im = evaluate(collocation)
        re_plus, im_plus = evaluate(collocation + delta)
        re_minus, im_minus = evaluate(collocation - delta)
        d_re = (re_plus - re_minus) / (2 * delta)
        d_im = (im_plus - im_minus) / (2 * delta)
        # Schrödinger equation: d(re+i im)/dt = -i H (re+i im).
        residual_re = d_re - torch.einsum("ab,tbc->tac", h, im)
        residual_im = d_im + torch.einsum("ab,tbc->tac", h, re)
        loss = (residual_re.square() + residual_im.square()).sum((1, 2)).mean()
        loss.backward()
        optimizer.step()
    seconds = time.perf_counter() - start

    tgrid = np.linspace(0., t_final, 151)
    reference = exact_evolution(h_np, probs, tgrid)
    with torch.no_grad():
        re, im = evaluate(torch.tensor(tgrid[:, None], dtype=dtype))
        populations = (re.square() + im.square()).sum(dim=2).numpy()
        up = populations[:, np.array([(s & 1) > 0 for s in range(dim)])].sum(1)
        down = populations[:, np.array([((s >> 1) & 1) > 0 for s in range(dim)])].sum(1)
        trace = populations.sum(axis=1)
    predicted = np.stack((up, down), axis=1)
    truth = reference[:, 1:3]
    return {
        "temperature": float(temperature), "interaction": float(interaction),
        "horizon": float(t_final), "steps": int(steps), "seed": int(seed),
        "runtime_seconds": seconds, "final_loss": float(loss.detach()),
        "occupation_rmse": float(np.sqrt(np.mean((predicted-truth)**2))),
        "terminal_occupation_error": float(np.max(np.abs(predicted[-1]-truth[-1]))),
        "max_trace_error": float(np.max(np.abs(trace-1))),
        "_curves": (tgrid, truth, predicted, trace),
    }


def run(output_dir: Path, steps=3000):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for horizon in (2.0, 5.0):
        for temperature in (0.3, 3.0):
            result = train(temperature=temperature, t_final=horizon, steps=steps)
            times, truth, prediction, trace = result.pop("_curves")
            rows.append(result)
            with (output_dir / f"T{temperature}_h{horizon}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "reference_up", "reference_down",
                                 "pinn_up", "pinn_down", "neural_trace"])
                for i in range(len(times)):
                    writer.writerow([times[i], *truth[i], *prediction[i], trace[i]])
            print(json.dumps(result), flush=True)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# Finite-temperature Anderson PINN baseline", "",
        "**Finite discrete bath, interacting U=1. NOT Cao et al. PINN-DQME/HEOM reproduction.**", "",
        "| T | time horizon | population RMSE | terminal error | trace error | CPU seconds |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['temperature']} | {row['horizon']} | "
            f"{row['occupation_rmse']:.5e} | {row['terminal_occupation_error']:.5e} | "
            f"{row['max_trace_error']:.5e} | {row['runtime_seconds']:.1f} |"
        )
    lines += [
        "", "Initial conditions imposed exactly. The PINN is a weighted thermal "
        "wavefunction mixture, not a DQME approximation.",
        "Discrete reservoir levels produce artificial recurrences. No continuum "
        "or thermodynamic limit is claimed.",
        "The exact small Hamiltonian solver is cheaper than the neural model.",
        "These runs have one seed; no evidence for general numerical superiority."
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines)+"\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", default="outputs/finite_anderson")
    args = parser.parse_args()
    run(Path(args.output), steps=args.steps)
