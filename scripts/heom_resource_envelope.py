"""Conservative fermionic HEOM hierarchy size estimates.

For K independent fermionic exponent labels and max_depth d, the number of
allowed 0/1 occupancy ADOs is sum_{k=0}^d binom(K,k).
The estimate ignores possible structural reductions, symmetry, RHS sparsity and
integrator work arrays, and is NOT a runtime/peak-memory predictor.
"""
import csv
import json
from math import comb
from pathlib import Path


def ado_count(n_exponents, depth):
    if n_exponents < 0 or depth < 0:
        raise ValueError("Nonnegative inputs required")
    return sum(comb(n_exponents, k) for k in range(min(depth, n_exponents)+1))


def size_summary(num_spins=2, num_leads=2, cases=(("high",3),("low",7)),
                 max_depth=4):
    rows=[]
    for name,nk in cases:
        n_exp=num_spins*num_leads*2*(nk+1)
        for depth in range(1,max_depth+1):
            n_ado=ado_count(n_exp,depth)
            # Two spin orbitals -> Hilbert-space dimension 4, so 16 matrix entries per ADO.
            complex_entries=n_ado*16
            rows.append({
                "case":name,"Nk":nk,"n_exponents":n_exp,"depth":depth,
                "n_ados":n_ado,"complex_state_entries":complex_entries,
                "bare_state_megabytes":16*complex_entries/1e6,
            })
    return rows


def run(directory="outputs/heom_resource_envelope"):
    output=Path(directory);output.mkdir(parents=True,exist_ok=True)
    rows=size_summary()
    with (output/"estimates.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    (output/"estimates.json").write_text(json.dumps(rows,indent=2))
    lines=["# Fermionic HEOM hierarchy size envelope","",
           "4 independent spin/lead baths, each with two fermionic sign sectors.",
           "K=8*(Nk+1), ADOs up to depth d = sum binom(K,k).",
           "Bare complex state arrays only; actual sparse operators and ODE memory",
           "may be far larger. These are not measured runtimes.","",
           "| Case | Nk | depth | exponent modes | ADOs | bare state MB |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['Nk']} | {r['depth']} | "
                     f"{r['n_exponents']} | {r['n_ados']} | {r['bare_state_megabytes']:.3f} |")
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return rows


if __name__=="__main__":
    run()
