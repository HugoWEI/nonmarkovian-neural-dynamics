"""Padé refinement against exact quadratic single-fermion dynamics.

Same physical one-lead RLM and 6ps initial-value problem, fixed HEOM depth.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from nonmarkovian_neural_dynamics.exact_rlm import reference_with_checks
from spinless_rlm_heom import heom

def run(ratio,output,depth=4):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    t=np.linspace(0,6,121);T=.1*ratio
    exact,reference_errors=reference_with_checks(t,T)
    orders=(3,5,7,9) if ratio==3.0 else (7,9,11,13)
    rows=[];curves={}
    for nk in orders:
        pred,result=heom(t,T,nk,depth)
        result.update({"T_over_Gamma":ratio,
                       "max_vs_exact":float(max(abs(pred-exact))),
                       "rmse_vs_exact":float(np.sqrt(np.mean((pred-exact)**2)))})
        curves[nk]=pred;rows.append(result)
        print(json.dumps(result),flush=True)
    with (out/"trajectories.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["time_ps","exact"]+[f"Nk{n}" for n in orders])
        w.writerows([t[i],exact[i]]+[curves[n][i] for n in orders]
                    for i in range(len(t)))
    data={"T_over_Gamma":ratio,"depth":depth,"orders":rows,
          "reference_resolution_checks":reference_errors,
          "independent_reference":True,"source_SIAM_reproduction":False}
    (out/"results.json").write_text(json.dumps(data,indent=2))
    lines=[f"# Spinless RLM source-order and higher Padé refinement, T/Gamma={ratio}","",
           "Quadratic exact noninteracting benchmark; NOT original spinful correlated SIAM.",
           f"Fixed HEOM depth {depth}; time 0..6ps.", "",
           "| Nk | Max error vs quadratic | RMSE | Depth | Runtime s |",
           "|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['Nk']} | {r['max_vs_exact']:.6e} | "
                     f"{r['rmse_vs_exact']:.6e} | {depth} | {r['integrate_seconds']:.2f} |")
    lines+=["",f"Discretization resolution error upper diagnostic: {reference_errors['resolution_max']:.3e}.",
            f"Spectral-cutoff diagnostic: {reference_errors['cutoff_max']:.3e}.",
            "A cutoff comparison alone is not a rigorous bound. Neither hierarchy nor ODE tolerance convergence is certified solely by this test."]
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return data
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ratio",type=float,required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();run(a.ratio,a.output)
