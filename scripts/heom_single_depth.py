"""Single deeper HEOM hierarchy calculation with strict outcome diagnostics.

Factorized empty-dot initial condition only, NOT the authors' original ADO
state or convergence-certified continuum reference. Used to quantify depth
error after separately checking Padé order.
"""
import argparse
import csv
import json
from pathlib import Path

from heom_convergence import solve


def run(ratio,nk,depth,output,t_end=1.5):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    result=solve(ratio,nk,depth,t_end,31)
    record={k:v for k,v in result.items() if k not in ("times","occupations")}
    (out/"results.json").write_text(json.dumps(record,indent=2))
    with (out/"trajectory.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("time_ps","n_up","n_down"))
        writer.writerows(zip(result["times"],result["occupations"][:,0],result["occupations"][:,1]))
    report=[
        f"# HEOM deeper hierarchy T/Gamma={ratio}, Nk={nk}, depth={depth}",
        "", "**This is an independent factorized-empty-dot test, not Cao et al.'s original correlated DQME state.**",
        "", "| Diagnostic | Result |","|---|---:|"
    ]
    for k,v in record.items():
        report.append(f"| {k} | {v} |")
    report += ["", "Compare the full trajectory against matching Nk at depth 2",
               "before claiming any convergence; check ODE tolerance separately."]
    (out/"SUMMARY.md").write_text("\n".join(report)+"\n")
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",type=float,required=True)
    p.add_argument("--nk",type=int,required=True)
    p.add_argument("--depth",type=int,default=3)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    run(a.ratio,a.nk,a.depth,a.output)
