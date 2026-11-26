"""Speed comparison figure for the HEOM steady-state benchmark (laptop results).

Left: exact (direct) solves, QuTiP vs deepest-tier elimination.
Right: iterative GMRES with HierarchicalEOM.jl's default settings, full hierarchy
vs eliminated (matrix-free) system, same tolerance.

Usage: python plot_benchmark.py   -> results/speed_comparison.{svg,png,pdf}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = json.loads((HERE / "results" / "laptop_2026-11-26.json").read_text())
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


# ---- left: exact solves --------------------------------------------------
ax = axes[0]
rows = R["qutip_direct_vs_elimination_direct"]
done = [r for r in rows if isinstance(r["qutip_direct_s"], (int, float))]
stopped = [r for r in rows if not isinstance(r["qutip_direct_s"], (int, float))]
ax.plot([r["unknowns"] for r in done], [r["qutip_direct_s"] for r in done], "-o",
        color=BASE, lw=2, ms=8, mec="white", mew=1.5, label="QuTiP direct solve")
ax.plot([r["unknowns"] for r in rows], [r["elimination_direct_s"] for r in rows], "-o",
        color=OURS, lw=2, ms=8, mec="white", mew=1.5, label="Exact elimination (direct)")
for r in stopped:   # stopped run: lower bound of 25 min
    ax.annotate("", xy=(r["unknowns"], 4000), xytext=(r["unknowns"], 1500),
                arrowprops=dict(arrowstyle="-|>", color=BASE, lw=2))
    ax.plot(r["unknowns"], 1500, "o", color="white", mec=BASE, mew=2, ms=8)
    ax.text(r["unknowns"] * 0.85, 1500, "not finished\nafter 25 min", ha="right",
            va="center", color=INK, fontsize=9)
for r in done:
    s = r["qutip_direct_s"] / r["elimination_direct_s"]
    ax.text(r["unknowns"] * 1.08, r["elimination_direct_s"] / 2.6, f"{s:.0f}× faster",
            color=INK, fontsize=9, ha="left", va="top")
for r in stopped:
    ax.text(r["unknowns"] * 0.95, r["elimination_direct_s"] / 2.6, f"> {1500 / r['elimination_direct_s']:.0f}× faster",
            color=INK, fontsize=9, ha="left", va="top")
style(ax, "Exact solves: QuTiP vs elimination")
ax.set_ylim(0.004, 8000); ax.set_xlim(8000, 5e5)
ax.legend(frameon=False, loc="upper left")

# ---- right: GMRES, same settings ------------------------------------------
ax = axes[1]
cases = [c for c in R["gmres_same_settings"]["cases"] if c["unknowns_full"] > 20000]
x = [c["unknowns_full"] for c in cases]
ax.plot(x, [c["full_s"] for c in cases], "-o", color=BASE, lw=2, ms=8, mec="white", mew=1.5,
        label="HierarchicalEOM.jl default (GMRES, full)")
ax.plot(x, [c["eliminated_mf_s"] for c in cases], "-o", color=OURS, lw=2, ms=8, mec="white",
        mew=1.5, label="Elimination + same GMRES")
for c in cases:
    s = c["full_s"] / c["eliminated_mf_s"]
    ax.text(c["unknowns_full"], c["eliminated_mf_s"] / 2.0, f"{s:.1f}×",
            color=INK, fontsize=9, ha="center", va="top")
style(ax, "Same solver, same accuracy: Julia vs elimination")
ax.set_ylim(0.02, 400); ax.set_xlim(6e4, 6e6)
ax.legend(frameon=False, loc="upper left")

fig.text(0.01, 0.005, "Spinful Anderson impurity, T/Γ = 0.3, laptop (Windows, 31.7 GB). "
         "Right: GMRES rtol 1e-12 for both; largest case restarted at 100 for both. "
         "Answers agree to ≤ 2×10⁻¹².", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = HERE / "results"
for ext in ("svg", "png", "pdf"):
    fig.savefig(out / f"speed_comparison.{ext}", dpi=200)
print("saved", out / "speed_comparison.svg")
