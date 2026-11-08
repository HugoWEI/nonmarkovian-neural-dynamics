"""Figure for the tier-elimination convergence study (reads outputs/ JSON/npz)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/tier_elimination_convergence"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False})

z = np.load(SRC / "trajectories.npz")
summary = json.loads((SRC / "summary.json").read_text())
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)

# (a) low-T source-order quench, depths 2-4
t = z["times"]
for d, c, ls in ((2, BLUE, "-"), (3, ORANGE, "--"), (4, AQUA, "-")):
    ax[0].plot(t, z[f"T0.3_Nk7_d{d}"], color=c, lw=2, ls=ls, label=f"depth {d}")
ax[0].set(xlabel="time after quench (ps)", ylabel=r"$\langle n_\uparrow\rangle$",
          title="(a) Low-T quench, Padé order Nk=7")
ax[0].legend(loc="lower right")

# (b) change between successive depths
series = ((3.0, 3, BLUE, "o", "T/Γ=3, Nk=3"), (0.3, 3, ORANGE, "s", "T/Γ=0.3, Nk=3"),
          (0.3, 5, AQUA, "^", "T/Γ=0.3, Nk=5"), (0.3, 7, YELLOW, "D", "T/Γ=0.3, Nk=7"))
for ratio, nk, c, m, lab in series:
    ms = {"o": 7, "s": 9, "^": 7, "D": 5}[m]
    pts = [(int(r["depths"].split("->")[1]), r["max_abs_diff"]) for r in summary
           if "depths" in r and r["T_over_Gamma"] == ratio and r["Nk"] == nk]
    x, y = zip(*pts)
    ax[1].semilogy(x, y, color=c, marker=m, ms=ms, lw=2, label=lab,
                   markeredgecolor="white", markeredgewidth=1)
ax[1].set_xticks([3, 4, 5], ["2→3", "3→4", "4→5"])
ax[1].set(xlabel="hierarchy depth step", ylabel=r"max $|\Delta n_\uparrow|$ over 6 ps",
          title="(b) Hierarchy convergence")
ax[1].text(4.05, 2e-4, "three low-T curves overlap:\ndepth error barely\ndepends on Nk",
           color=INK, fontsize=8, va="center")
ax[1].legend(loc="lower left")

# (c) stationary-state cost (measured on the same 2-CPU, 7 GB machine)
ados = [697, 2517, 5489, 12951, 41449]
elim = [0.03, 0.40, 0.46, 11.3, 164]
direct_ados, direct = [697, 2517, 5489], [0.18, 5.8, 75.1]
ax[2].loglog(ados, elim, color=BLUE, marker="o", ms=7, lw=2, label="exact tier elimination",
             markeredgecolor="white")
ax[2].loglog(direct_ados, direct, color=ORANGE, marker="s", ms=7, lw=2, ls="--",
             label="direct QuTiP solve", markeredgecolor="white")
ax[2].scatter([12951], [75.1], marker="x", s=70, color=ORANGE, zorder=5)
ax[2].annotate("direct solve ran out\nof memory (7 GB)", (12951, 75.1),
               textcoords="offset points", xytext=(-4, -30), ha="center",
               color=INK, fontsize=8)
ax[2].set(xlabel="auxiliary density operators", ylabel="steady-state time (s)",
          title="(c) Correlated steady-state cost")
ax[2].legend(loc="upper left")
plt.rcParams["svg.fonttype"] = "none"
for ext in ("svg", "png"):
    out = ROOT / f"results/figures/tier_elimination_convergence.{ext}"
    fig.savefig(out, dpi=180, metadata={"Date": None} if ext == "svg" else None)
    print(out)
