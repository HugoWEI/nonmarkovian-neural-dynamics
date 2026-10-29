"""Fixed-hierarchy Padé-order refinement beyond paper's source decomposition.

Empty-dot/factorized initial ADO benchmark ONLY. Not the paper's correlated
initial condition; no claims about published PINN-DQME error.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from heom_convergence import solve, compare


def run(ratio, depth, out, t_end=1.5):
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    orders=(3,5,7) if ratio==3.0 else (3,5,7,9,11)
    traces={}
    runs=[]
    for nk in orders:
        solved=solve(ratio,nk,depth,t_end,31)
        traces[nk]=solved
        run={k:v for k,v in solved.items() if k not in ("times","occupations")}
        runs.append(run)
        print(json.dumps(run),flush=True)
        with (dest/f"Nk{nk}.csv").open("w",newline="") as file:
            writer=csv.writer(file)
            writer.writerow(["time_ps","n_up","n_down"])
            writer.writerows(zip(solved["times"],solved["occupations"][:,0],solved["occupations"][:,1]))
    differences=[]
    for low,high in zip(orders[:-1],orders[1:]):
        differences.append({"Nk_pair":[low,high],**compare(traces[low],traces[high])})
    result={"T_over_Gamma":ratio,"depth":depth,"runs":runs,
            "adjacent_order_differences":differences,
            "converged":False,
            "initial_state":"factorized empty impurity"}
    (dest/"results.json").write_text(json.dumps(result,indent=2))
    lines=[
        f"# Padé refinement T/Gamma={ratio}, depth={depth}","",
        "**Independent empty-dot HEOM; original correlated source initial state NOT matched.**",
        "This isolates Padé-order variation at a fixed, potentially underconverged hierarchy depth.",
        "Source orders: Nk=3 at high T, Nk=7 at low T.","",
        "| Nk | n ADOs | n_up terminal | minimum density eigenvalue |",
        "|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(f"| {r['Nk']} | {r['n_ados']} | {r['terminal_n_up']:.8f} | {r['min_density_eigenvalue']:.3e} |")
    lines += ["","| Nk pair | maximum full-trajectory occupation difference |",
              "|---|---:|"]
    for d in differences:
        lines.append(f"| {d['Nk_pair']} | {d['max_occupation_difference']:.6e} |")
    lines += [
        "", "Only convergence of the Padé expansion at this particular hierarchy depth",
        "is tested; hierarchy depth and time integrator are independent error sources.",
        "No conclusive full HEOM convergence or neural advantage claimed."
    ]
    (dest/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ratio",required=True,type=float)
    parser.add_argument("--depth",type=int,default=2)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    run(args.ratio,args.depth,args.output or f"outputs/pade_refinement/T{args.ratio}")
