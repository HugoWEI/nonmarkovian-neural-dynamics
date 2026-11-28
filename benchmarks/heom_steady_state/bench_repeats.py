"""Repeated QuTiP direct solves vs exact deepest-tier elimination (3 runs each, medians).

Run from this directory: python bench_repeats.py [machine label]
Writes results/repeats.json. Records Python, NumPy, SciPy and QuTiP versions.
"""
import gc, json, os, sys, time, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy, qutip
import bench
out = {"machine": sys.argv[1] if len(sys.argv) > 1 else platform.processor(),
       "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "python": platform.python_version(),
       "numpy": np.__version__, "scipy": scipy.__version__, "qutip": qutip.__version__,
       "T_over_Gamma": bench.RATIO, "mu_meV": bench.MU, "repeats": 3, "cases": []}
plan = [(3, 3, True), (3, 4, True), (7, 3, True), (5, 4, False), (7, 4, False)]
for nk, d, direct in plan:
    solver, nup, _ = bench.build(nk, d)
    labels = list(solver.ados.labels); L = bench.generator(solver)
    rec = {"Nk": nk, "depth": d, "unknowns": int(L.shape[0])}
    te = []
    for _ in range(3):
        gc.collect(); t = time.perf_counter(); y = bench.eliminated(L, labels, 4); te.append(time.perf_counter() - t)
    rec["eliminated_all_s"] = te; rec["eliminated_median_s"] = float(np.median(te))
    rec["n_up_eliminated"] = bench.n_up_of(y, 4, nup)
    if direct:
        td = []
        for _ in range(3):
            gc.collect(); t = time.perf_counter(); _, ado = solver.steady_state(use_mkl=False); td.append(time.perf_counter() - t)
        rec["qutip_all_s"] = td; rec["qutip_median_s"] = float(np.median(td))
        yd = np.asarray(ado._ado_state, dtype=complex).reshape(-1)
        rec["max_abs_diff"] = float(np.max(np.abs(yd - y)))
        rec["speedup_median"] = rec["qutip_median_s"] / rec["eliminated_median_s"]
    out["cases"].append(rec); print(json.dumps(rec), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'repeats.json'), 'w'), indent=1)
print("done")
