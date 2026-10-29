"""Check QuTiP fermionic-bath conventions against Cao et al. source.

Two independent routes:
 - LorentzianPadeBath convenience constructor (QuTiP)
 - FermionicBath built explicitly from the source-matching Padé coefficients
The comparison includes +/- sectors and nonzero lead chemical potential.
"""
import csv
import json
from pathlib import Path

import numpy as np

from pade_source_match import reconstruct, HBAR


def source_fermionic_bath(Q, thermal_energy_mev, nk, mu_mev, tag=None):
    from qutip.solver.heom import FermionicBath

    ck, rates = reconstruct(thermal_energy_mev, nk)
    # gamma from source: real gamma negative + i*(-1)**(sigma_index+1)*mu/hbar.
    # QuTiP rates vk=positive_rate - sigma*i*mu/hbar, sigma=+1 for C+.
    mu = mu_mev / HBAR
    vk_plus = rates - 1j * mu
    vk_minus = rates + 1j * mu
    return FermionicBath(Q, list(ck), list(vk_plus),
                         list(ck), list(vk_minus), tag=tag)


def correlations(bath, times):
    """Assemble C_+ and C_- directly from QuTiP BathExponent ck and vk."""
    result = {"+":np.zeros(len(times), dtype=complex),
              "-":np.zeros(len(times), dtype=complex)}
    for exp in bath.exponents:
        if exp.type == exp.types["+"]:
            sign = "+"
        elif exp.type == exp.types["-"]:
            sign = "-"
        else:
            raise ValueError(f"Unexpected exponent type: {exp.type}")
        result[sign] += exp.ck * np.exp(-exp.vk * times)
    return result


def compare_case(temperature_ratio, nk, mu_mev):
    from qutip import destroy
    from qutip.solver.heom import LorentzianPadeBath

    Q = destroy(2)
    temperature_mev = .1 * temperature_ratio
    builtin = LorentzianPadeBath(Q, .1/HBAR, 5./HBAR,
                                  mu_mev/HBAR, temperature_mev/HBAR, nk)
    manual = source_fermionic_bath(Q, temperature_mev, nk, mu_mev)
    times = np.linspace(0, 1.5, 61)
    b, m = correlations(builtin,times), correlations(manual,times)
    errors = {sign: float(np.max(abs(b[sign]-m[sign]))) for sign in ("+","-")}
    return {
        "T_over_Gamma": temperature_ratio, "Nk": nk, "mu_mev": mu_mev,
        "max_plus_correlation_error": errors["+"],
        "max_minus_correlation_error": errors["-"],
        "max_error": max(errors.values()),
        "built_in_exponents": len(builtin.exponents),
        "manual_exponents": len(manual.exponents),
    }


def run(output="outputs/quutip_correlation_match"):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for temp,nk in ((3.,3),(.3,7)):
        for mu in (0.,.05):
            r=compare_case(temp,nk,mu)
            rows.append(r)
            print(json.dumps(r),flush=True)
            if not r["max_error"] < 1.e-8:
                raise AssertionError(f"QuTiP/source mismatch {r}")
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    with (out/"results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    lines=[
        "# QuTiP built-in versus independently reconstructed source correlations",
        "", "Checks C+ and C- in both lead chemical-potential cases for T/Gamma=0.3, 3.",
        "The source coefficients match the original res_corr file separately.",
        "This does not validate the many-body hierarchy or initial auxiliary state.",
        "", "| T/Gamma | mu (meV) | plus max error | minus max error |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['T_over_Gamma']} | {r['mu_mev']} | "
                     f"{r['max_plus_correlation_error']:.3e} | "
                     f"{r['max_minus_correlation_error']:.3e} |")
    (out/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    return rows


if __name__=="__main__":
    run()
