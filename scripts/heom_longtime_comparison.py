"""Long-time two-depth reference for source-anchored fermionic HEOM.

Same source-matching Padé Nk as the authors, with empty-dot factorized ADO
initialization. Does NOT reproduce their original correlated initial state.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from heom_convergence import solve,compare


def run(ratio,output,horizon=6.,samples=121):
    nk=3 if ratio==3. else 7
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    traces={}
    records=[]
    for depth in (2,3):
        x=solve(ratio,nk,depth,horizon,samples)
        traces[depth]=x
        record={k:v for k,v in x.items() if k not in ("times","occupations")}
        records.append(record)
        print(json.dumps(record),flush=True)
    difference=compare(traces[2],traces[3])
    if np.max(np.abs(traces[2]["times"]-traces[3]["times"]))>1e-12:
        raise AssertionError("Time grids differ")
    with (out/"trajectories.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["time_ps","n_up_depth2","n_up_depth3","n_down_depth2","n_down_depth3"])
        for i,t in enumerate(traces[2]["times"]):
            writer.writerow([t,
              traces[2]["occupations"][i,0],traces[3]["occupations"][i,0],
              traces[2]["occupations"][i,1],traces[3]["occupations"][i,1]])
    report={"ratio":ratio,"Nk":nk,"horizon_ps":horizon,
            "reference_initialization":"empty impurity and factorized ADOs",
            "runs":records,
            "depth2_vs3":difference,"converged":False}
    (out/"results.json").write_text(json.dumps(report,indent=2))
    lines=[
        f"# Source-Padé fermionic HEOM at T/Gamma={ratio} to t={horizon} ps", "",
        "**Original correlated initial hierarchy NOT reproduced: independent empty-dot benchmark.**",
        f"Source Nk={nk}; compare depth 2 versus 3 across {samples} time samples.", "",
        "| Depth | ADOs | terminal n_up | minimum density eigenvalue | integration sec |",
        "|---:|---:|---:|---:|---:|"
    ]
    for r in records:
        lines.append(f"| {r['depth']} | {r['n_ados']} | {r['terminal_n_up']:.8f} | "
                     f"{r['min_density_eigenvalue']:.3e} | {r['integrate_seconds']:.2f} |")
    lines += ["", "## Depth sensitivity over full time grid", "",
              f"- Maximum occupation discrepancy: {difference['max_occupation_difference']:.8e}",
              f"- RMS discrepancy: {difference['rms_occupation_difference']:.8e}",
              f"- Terminal discrepancy: {difference['terminal_occupation_difference']:.8e}",
              "", "Physical validity and numerical convergence are separate properties.",
              "A third depth/order or integrator tolerance check remains necessary.",
              "The calculation is *not* a PINN or full original-paper DQME reproduction."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"depth2_vs3":difference}),flush=True)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ratio",type=float,required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--horizon",type=float,default=6.0)
    a=ap.parse_args()
    run(a.ratio,a.output,a.horizon)
