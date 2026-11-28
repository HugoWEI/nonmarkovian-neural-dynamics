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
style(ax, "(a) Comparison with QuTiP")
ax.set_ylim(0.004, 8000); ax.set_xlim(8000, 5e5)
ax.legend(frameon=False, loc="upper left")

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

fig.text(0.01, 0.005, "Anderson impurity, T/Γ = 0.3, local machine (i7-13650HX, 32 GB). (b): one thread, medians of 3–5 runs, "
         "same GMRES tolerances; largest case vs full GMRES, both restarted at 100. Answers agree to ≤ 2×10⁻¹¹.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = HERE / "results"
for ext in ("svg", "png", "pdf"):
    fig.savefig(out / f"speed_comparison.{ext}", dpi=200)
print("saved", out / "speed_comparison.svg")
