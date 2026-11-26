"""Parse and audit the ACTUAL published exponential fermionic bath coefficients.

The original data are read from an immutable upstream GitHub commit at runtime;
the source file is not copied into our repository.
"""
import csv
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np

COMMIT = "7fb6ffb6115db42b299cc5c5092b27bfa9050628"
ROOT = f"https://raw.githubusercontent.com/caolong-cn/NQS-DQME_fermions/{COMMIT}/examples"
SOURCES = {"low": "case1_T=0.3/res_corr.data",
           "high": "pinn_case1_T=3.0/res_corr.data"}


def parse_coefficients(source):
    lines = [s.split() for s in source.splitlines() if s.strip()]
    if len(lines) < 6:
        raise ValueError("Expected five header dimensions and coefficients")
    dims = tuple(int(lines[i][0]) for i in range(5))
    if len(lines) - 5 != int(np.prod(dims)):
        raise ValueError("Header dimensions and coefficient count disagree")
    groups = {}
    for tokens in lines[5:]:
        if len(tokens) < 12:
            raise ValueError("Bad coefficient row")
        _, sign, spin, orbital, lead, k = map(int, tokens[:6])
        eta = complex(float(tokens[6]), float(tokens[7]))
        decay = complex(float(tokens[10]), float(tokens[11]))
        if decay.real >= 0:
            raise ValueError("Unexpected growing exponential")
        groups.setdefault((sign, spin, orbital, lead), []).append((k, eta, decay))
    for key, terms in groups.items():
        terms.sort(key=lambda a: a[0])
        if [term[0] for term in terms] != list(range(1, dims[-1]+1)):
            raise ValueError(f"Incomplete terms for {key}")
    return dims, groups


def analyze_text(source, window=20.0):
    dims, groups = parse_coefficients(source)
    t = np.linspace(0, window, 4001)
    summaries = []
    for key, terms in sorted(groups.items()):
        eta = np.asarray([z[1] for z in terms])
        exponent = np.asarray([z[2] for z in terms])
        corr = np.exp(exponent[:, None] * t[None, :]).T @ eta
        rate = float(np.min(-exponent.real))
        summaries.append({
            "sign": key[0], "spin": key[1], "orbital": key[2], "lead": key[3],
            "n_terms": len(terms), "slowest_rate": rate,
            "slowest_time_constant": 1./rate,
            "abs_C0": float(abs(corr[0])),
            "integral_abs_C_0_20": float(np.trapezoid(abs(corr), t)),
        })
    return dims, summaries


def run(out="outputs/upstream_bath"):
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    data = {}
    for label, rel in SOURCES.items():
        with urlopen(f"{ROOT}/{rel}", timeout=30) as response:
            raw = response.read().decode("utf-8")
        dims, rows = analyze_text(raw)
        data[label] = {"dimensions": list(dims), "groups": rows, "source": rel}
    with (path/"coefficients_audit.csv").open("w", newline="") as f:
        entries = [{"case": case, **row} for case, item in data.items()
                   for row in item["groups"]]
        writer = csv.DictWriter(f, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    (path/"results.json").write_text(json.dumps(data, indent=2))
    low = min(x["slowest_rate"] for x in data["low"]["groups"])
    high = min(x["slowest_rate"] for x in data["high"]["groups"])
    lines = [
        "# Pinned upstream bath-correlation audit", "",
        f"Upstream revision: `{COMMIT}`",
        "The fitted correlation in each channel is sum eta_k * exp(gamma_k t).",
        "This checks correlation fitting coefficients only: NOT HEOM or trained PINN.",
        "", "| Case | Coefficient dimensions | Slowest rate | Decay time |",
        "|---|---|---:|---:|",
    ]
    for case, x in data.items():
        r = min(z["slowest_rate"] for z in x["groups"])
        lines.append(f"| {case} | {x['dimensions']} | {r:.8f} | {1/r:.6f} |")
    lines += ["", f"Low-/high-temperature slow-mode timescale ratio: {high/low:.6f}.",
              "The T values in filenames are T/Gamma, not temperatures in kelvin.",
              "One slow fitted exponential does not establish network error amplification."]
    (path/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return data


if __name__ == "__main__":
    run()
