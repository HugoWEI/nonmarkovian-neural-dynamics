"""Audit zero-tier reduced density matrices in pinned upstream state files.

These data are *not* by themselves a full system-bath initial hierarchy.
"""
import csv
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np

REV="7fb6ffb6115db42b299cc5c5092b27bfa9050628"
ROOT=f"https://raw.githubusercontent.com/caolong-cn/NQS-DQME_fermions/{REV}/examples"
CASES={"low":("case1_T=0.3/table_cut3.data",.3),
       "high":("pinn_case1_T=3.0/table_cut3.data",3.0)}


def parse_first_diagonal(source):
    lines=[line.split() for line in source.splitlines() if line.strip()]
    if len(lines)<5:
        raise ValueError("Expected four zero-tier diagonal entries and further ADO data")
    probabilities=np.array([float(row[0]) for row in lines[:4]],dtype=float)
    imag=np.array([float(row[1]) for row in lines[:4]])
    for row in lines[:4]:
        if any(int(item)!=0 for item in row[2:-4]):
            raise ValueError("Nonzero dissipatons in first four rows")
        if tuple(map(int,row[-4:-2])) != tuple(map(int,row[-2:])):
            raise ValueError("First four rows not diagonal reduced-density elements")
    np.testing.assert_allclose(probabilities.sum(),1.,atol=1e-8)
    np.testing.assert_allclose(imag,0.,atol=1e-12)
    if not np.all(probabilities>=0):
        raise ValueError("Negative diagonal populations")
    return probabilities


def run(output="outputs/upstream_initial_state"):
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    results={}
    for name,(filename,temp) in CASES.items():
        with urlopen(f"{ROOT}/{filename}",timeout=40) as resp:
            raw=resp.read().decode("utf-8")
        p=parse_first_diagonal(raw)
        results[name]={"T_over_Gamma":temp,"diagonal":p.tolist(),
                       "occupation_up":float(p[1]+p[3]),
                       "occupation_down":float(p[2]+p[3]),
                       "total_occupation":float(p[1]+p[2]+2*p[3])}
        print(json.dumps({name:results[name]}),flush=True)
    (path/"results.json").write_text(json.dumps(results,indent=2))
    with (path/"zero_tier_probabilities.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["case","empty","up","down","double","total_occupation"])
        for k,v in results.items():
            writer.writerow([k,*v["diagonal"],v["total_occupation"]])
    lines=[
        "# Original source reduced-density diagnostic",
        "",
        "**These are ZERO-TIER stored source populations; NOT a full initial correlated ADO hierarchy.**",
        "", "| T/Gamma | empty | spin-up | spin-down | double | total occupation |",
        "|---:|---:|---:|---:|---:|---:|"
    ]
    for v in results.values():
        a=v["diagonal"]
        lines.append(f"| {v['T_over_Gamma']} | {a[0]:.6f} | {a[1]:.6f} | "
                     f"{a[2]:.6f} | {a[3]:.6f} | {v['total_occupation']:.6f} |")
    lines+=["", "Using only these reduced populations as initial rho0 with all higher ADOs set to zero",
            "does NOT reproduce the authors' correlated stationary/initial dissipaton state.",
            "The current HEOM empty-dot initial condition is therefore *not* a source-matched comparison."]
    (path/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return results


if __name__=="__main__":
    run()
