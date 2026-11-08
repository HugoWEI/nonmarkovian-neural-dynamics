"""Time-domain decomposition baseline for finite-temperature Anderson PINN.

This is a small discrete-bath Hamiltonian, not fermionic DQME or HEOM.
Time-domain decomposition is existing methodology, NOT a novel contribution.
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


def train_piecewise(temperature=0.3, interaction=1.0, nwindows=5,
                    steps_per_window=600, window_length=1., seed=7, width=48):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dtype = torch.float64
    energies, couplings = (-0.7,), (0.55,)
    h_np = anderson_hamiltonian(energies, couplings, interaction=interaction)
    h = torch.tensor(h_np.real, dtype=dtype)
    probabilities = thermal_diagonal(energies, temperature)
    indices = np.flatnonzero(probabilities > 1e-15)
    dim, n_config = len(probabilities), len(indices)
    start_state = np.zeros((dim, n_config), dtype=complex)
    start_state[indices, np.arange(n_config)] = np.sqrt(probabilities[indices])
    sample_times, solutions, losses = [], [], []
    begin = time.perf_counter()
    for window in range(nwindows):
        initial_re = torch.tensor(start_state.real, dtype=dtype)
        initial_im = torch.tensor(start_state.imag, dtype=dtype)
        net = torch.nn.Sequential(
            torch.nn.Linear(1, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 2 * dim * n_config)
        ).to(dtype=dtype)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

        def evaluate(t):
            correction = (t * net(t/window_length)).reshape(-1, 2, dim, n_config)
            return initial_re[None]+correction[:, 0], initial_im[None]+correction[:, 1]

        grid = torch.linspace(0, window_length, 41, dtype=dtype)[:, None]
        delta = 1e-3
        for _ in range(steps_per_window):
            optimizer.zero_grad(set_to_none=True)
            re, im = evaluate(grid)
            re_plus, im_plus = evaluate(grid+delta)
            re_minus, im_minus = evaluate(grid-delta)
            d_re = (re_plus-re_minus)/(2*delta)
            d_im = (im_plus-im_minus)/(2*delta)
            residual_re = d_re-torch.einsum("ab,tbc->tac", h, im)
            residual_im = d_im+torch.einsum("ab,tbc->tac", h, re)
            loss = (residual_re.square()+residual_im.square()).sum((1, 2)).mean()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
        local = np.linspace(0, window_length, 61)
        with torch.no_grad():
            re, im = evaluate(torch.tensor(local[:, None], dtype=dtype))
            values = re.numpy()+1j*im.numpy()
        if window > 0:
            local, values = local[1:], values[1:]
        sample_times.extend(window*window_length+local)
        solutions.extend(values)
        start_state = values[-1]

    runtime = time.perf_counter()-begin
    times = np.asarray(sample_times)
    diagonal = np.asarray([np.abs(psi)**2 for psi in solutions]).sum(axis=2)
    occupations = np.stack([
        diagonal[:, np.array([(state >> mode)&1 for state in range(dim)], dtype=bool)].sum(axis=1)
        for mode in (0,1)
    ], axis=1)
    trace = diagonal.sum(axis=1)
    truth = exact_evolution(h_np, probabilities, times)[:, 1:3]
    return {
        "temperature": temperature, "horizon": nwindows*window_length,
        "total_steps": nwindows*steps_per_window, "nwindows": nwindows,
        "population_rmse": float(np.sqrt(np.mean((occupations-truth)**2))),
        "terminal_population_error": float(np.max(np.abs(occupations[-1]-truth[-1]))),
        "max_trace_error": float(np.max(np.abs(trace-1))),
        "loss_per_window": losses, "runtime_seconds": runtime,
        "_curves": (times, truth, occupations, trace),
    }


def run(output: Path, steps_per_window=600, nwindows=5):
    output.mkdir(parents=True, exist_ok=True)
    results=[]
    for temp in (0.3,3.0):
        r=train_piecewise(temperature=temp, nwindows=nwindows,
                          steps_per_window=steps_per_window)
        t, truth, estimate, trace = r.pop("_curves")
        results.append(r)
        with (output/f"T{temp}_piecewise.csv").open("w", newline="") as handle:
            writer=csv.writer(handle)
            writer.writerow(["time","ref_up","ref_down","pinn_up","pinn_down","neural_trace"])
            for i in range(len(t)):
                writer.writerow([t[i],*truth[i],*estimate[i],trace[i]])
        print(json.dumps(r), flush=True)
    (output/"results.json").write_text(json.dumps(results, indent=2))
    lines = [
        "# Piecewise finite-temperature Anderson PINN", "",
        "**Finite discrete reservoir with interacting U=1 spinful dot; NOT the original fermionic DQME/HEOM calculation.**",
        "Time-domain decomposition is an existing technique, not new research.", "",
        "| T | horizon | total steps | occupation RMSE | terminal error | trace error | CPU seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in results:
        lines.append(
            f"| {r['temperature']} | {r['horizon']} | {r['total_steps']} | "
            f"{r['population_rmse']:.5e} | {r['terminal_population_error']:.5e} | "
            f"{r['max_trace_error']:.5e} | {r['runtime_seconds']:.2f} |"
        )
    lines += ["", "One seed only; no statistical conclusion about generalization."]
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps-per-window",type=int,default=600)
    parser.add_argument("--nwindows",type=int,default=5)
    parser.add_argument("--output",default="outputs/piecewise_anderson")
    args=parser.parse_args()
    run(Path(args.output),args.steps_per_window,args.nwindows)
