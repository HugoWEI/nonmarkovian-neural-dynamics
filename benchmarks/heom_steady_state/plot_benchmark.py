"""Speed comparison figure for the HEOM steady-state benchmark (local-machine results).

Left: exact (direct) solves, QuTiP vs deepest-tier elimination.
Right: iterative GMRES with HierarchicalEOM.jl's default settings, full hierarchy
vs eliminated (matrix-free) system, same tolerance.

Usage: python plot_benchmark.py   -> results/speed_comparison.{svg,png,pdf}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BASE, OURS = "#f47230", "#257de0"   # orange = existing solver, blue = elimination
INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e4e4e4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))


def style(ax, title):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Unknowns in the full hierarchy")
    ax.set_ylabel("Steady-state solve time (s)")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")


# ---- left: exact solves (medians of three runs, one BLAS thread) -------
ax = axes[0]
L = json.loads((HERE / "results" / "local_repeats_2026-11-28.json").read_text())["cases"]
done = [r for r in L if "qutip_median_s" in r]
ax.plot([r["unknowns"] for r in done], [r["qutip_median_s"] for r in done], "-o",
        color=BASE, lw=2, ms=8, mec="white", mew=1.5, label="QuTiP direct solve")
ax.plot([r["unknowns"] for r in L], [r["eliminated_median_s"] for r in L], "-o",
        color=OURS, lw=2, ms=8, mec="white", mew=1.5, label="Exact elimination (direct)")
STOP_N = 207216   # QuTiP stopped after 25 min in an earlier single run
ax.annotate("", xy=(STOP_N, 4000), xytext=(STOP_N, 1500),
            arrowprops=dict(arrowstyle="-|>", color=BASE, lw=2))
ax.plot(STOP_N, 1500, "o", color="white", mec=BASE, mew=2, ms=8)
ax.text(STOP_N * 0.85, 1500, "not finished\nafter 25 min", ha="right",
        va="center", color=INK, fontsize=9)
for r in done:
    s = r["qutip_median_s"] / r["eliminated_median_s"]
    ax.text(r["unknowns"] * 1.08, r["eliminated_median_s"] / 2.6, f"{s:.0f}× faster",
            color=INK, fontsize=9, ha="left", va="top")
t = next(r for r in L if r["unknowns"] == STOP_N)["eliminated_median_s"]
ax.text(STOP_N * 0.95, t / 2.6, f"> {1500 / t:.0f}× faster",
        color=INK, fontsize=9, ha="left", va="top")
style(ax, "(a) Comparison with QuTiP")
ax.set_ylim(0.004, 8000); ax.set_xlim(8000, 1.2e6)
ax.legend(frameon=False, loc="lower right")

# ---- right: end-to-end, same GMRES, one thread, medians -----------------
ax = axes[1]
F = json.loads((HERE / "results" / "laptop_final_2026-11-28.json").read_text())
cases = F["cases"]
x = [c["unknowns"] for c in cases]
base = [c.get("package_steadystate_s", c["full_gmres_end_to_end_s"]) for c in cases]
ax.plot(x, base, "-o", color=BASE, lw=2, ms=8, mec="white", mew=1.5,
        label="HierarchicalEOM.jl steadystate (GMRES, full)")
ax.plot(x, [c["eliminated_end_to_end_s"] for c in cases], "-o", color=OURS, lw=2, ms=8,
        mec="white", mew=1.5, label="HEOMTierElimination.jl (same GMRES)")
for c, b in zip(cases, base):
    s_ = b / c["eliminated_end_to_end_s"]
    ax.text(c["unknowns"], c["eliminated_end_to_end_s"] / 2.0, f"{s_:.1f}×",
            color=INK, fontsize=9, ha="center", va="top")
style(ax, "(b) Comparison with HierarchicalEOM.jl")
ax.set_ylim(0.03, 2000); ax.set_xlim(6e4, 6e6)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
out = HERE / "results"
for ext in ("svg", "png", "pdf"):
    fig.savefig(out / f"speed_comparison.{ext}", dpi=200)
print("saved", out / "speed_comparison.svg")
