"""Four-seed finite-bath Anderson comparison, phone-independent CPU workflow.

This tests a known time-domain PINN baseline, not PINN-DQME or new physics.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from finite_anderson_pinn import train
from finite_anderson_piecewise import train_piecewise


def run(output: Path, seeds=(1, 2, 3, 7), total_steps=3000):
    output.mkdir(parents=True, exist_ok=True)
    if total_steps % 5 != 0:
        raise ValueError("total_steps must be divisible by five")
    rows=[]
    for seed in seeds:
        global_fit = train(temperature=0.3, t_final=5.0,
                           steps=total_steps, seed=seed)
        window_fit = train_piecewise(temperature=0.3, nwindows=5,
                                     steps_per_window=total_steps//5,
                                     seed=seed)
        global_fit.pop("_curves")
        window_fit.pop("_curves")
        row={
            "seed":seed,
            "global_rmse":global_fit["occupation_rmse"],
            "piecewise_rmse":window_fit["population_rmse"],
            "global_trace":global_fit["max_trace_error"],
            "piecewise_trace":window_fit["max_trace_error"],
            "global_seconds":global_fit["runtime_seconds"],
            "piecewise_seconds":window_fit["runtime_seconds"],
        }
        rows.append(row)
        print(json.dumps(row),flush=True)
    with (output/"seed_comparison.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"results.json").write_text(json.dumps(rows,indent=2))
    summary=[
        "# Four-seed finite-bath Anderson PINN comparison", "",
        "Independent interacting finite-bath exact reference, NOT published PINN-DQME/HEOM.",
        "T=0.3, U=1, horizon=5. Each fit has the same TOTAL number of optimization steps.",
        "Global model uses 81 collocation times per step; piecewise uses 41 per step.", "",
        "| Seed | Global RMSE | Piecewise RMSE | Global trace drift | Piecewise trace drift |",
        "|---:|---:|---:|---:|---:|"
    ]
    for r in rows:
        summary.append(f"| {r['seed']} | {r['global_rmse']:.5e} | "
                       f"{r['piecewise_rmse']:.5e} | {r['global_trace']:.5e} | "
                       f"{r['piecewise_trace']:.5e} |")
    summary += ["", "Time-domain decomposition is already established in the literature;",
                "these results are a controlled intermediate baseline, not a new method."]
    (output/"SUMMARY.md").write_text("\n".join(summary)+"\n")
    return rows


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=3000)
    ap.add_argument("--output",default="outputs/anderson_seeds")
    args=ap.parse_args()
    run(Path(args.output),total_steps=args.steps)
