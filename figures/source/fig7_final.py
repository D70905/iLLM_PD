"""
Fig. 7 鈥?OOD robustness and safety mechanisms across eight stress-test scenarios.
Nature Communications compliant. Unified W_PAL palette with Fig. 4/6.

Panels:
  a 鈥?OOD response heatmap (escalation, guard, B3, SCR, reward)
  b 鈥?Ward hierarchical clustering (dendrogram top) + reward bars
  c 鈥?B3 compliance margin lollipop + SCR horizontal bars
  d 鈥?(user adds manually: audit chain)

Data: ood_aggregate_20260723_133102.csv, ood_per_run_20260723_133102.csv
"""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.unicode_minus"] = False

BASE = Path("D:/iLLM_PD_new")
STAMP = "20260723_133102"
OOD_DIR = BASE / "experiments" / "ltpp_data" / "deliverables" / "ood_stress"
AGG_CSV = OOD_DIR / f"ood_aggregate_{STAMP}.csv"
PER_RUN_CSV = OOD_DIR / f"ood_per_run_{STAMP}.csv"
OUT = BASE / "plot" / "fig7_final"

W = ["#083D6B", "#3F8EBD", "#B8D6E5", "#F1EEE8", "#F4B08F", "#C94A40", "#65001F"]
C_TEXT  = "#222222"; C_MUTED = "#888888"; C_GRID = "#DDDDDD"
C_LIMIT = W[5]
CAT_COLOR = {"subgrade": W[0], "material": W[5], "traffic": W[4], "climate": W[1]}
C_CLUSTER = [W[0], W[1], W[4], W[5]]

SCENARIO_ORDER = [
    "cl_extreme_hot", "mat_soft_base", "tr_ultra_light", "tr_super_heavy",
    "sg_ltpp_48_0001", "sg_very_soft", "sg_very_stiff", "cl_extreme_cold",
]
LABEL = {
    "cl_extreme_hot":  "Extreme-hot climate",   "mat_soft_base":   "Below-bound base",
    "tr_ultra_light":  "Ultra-light traffic",   "tr_super_heavy":  "Super-heavy traffic",
    "sg_ltpp_48_0001": "LTPP high-E subgrade",  "sg_very_soft":    "Very soft subgrade",
    "sg_very_stiff":   "Very stiff subgrade",   "cl_extreme_cold": "Extreme-cold climate",
}

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
def num(row, key):
    v = row.get(key, ""); return np.nan if v == "" else float(v)

agg = {r["case_id"]: r for r in read_csv(AGG_CSV)}
per = read_csv(PER_RUN_CSV)

esc   = np.array([num(agg[c], "escalation_rate") * 100 for c in SCENARIO_ORDER])
guard = np.array([num(agg[c], "guards_total")           for c in SCENARIO_ORDER])
b3    = np.array([num(agg[c], "B3_min")                 for c in SCENARIO_ORDER])
scr   = np.array([num(agg[c], "final_scr")              for c in SCENARIO_ORDER])
rew   = np.array([num(agg[c], "mean_reward")            for c in SCENARIO_ORDER])

# Clustering
data = np.column_stack([esc, guard, b3, scr, rew])
m = data.mean(axis=0); s = data.std(axis=0, ddof=1); s[s == 0] = 1
Z = linkage((data - m) / s, method="ward")
K = 4
cl = fcluster(Z, K, criterion="maxclust") - 1
cl_name = {}
for ci in range(K):
    idx = [i for i in range(8) if cl[i] == ci]
    if esc[idx].mean() > 30:       cl_name[ci] = "Escalation-driven"
    elif guard[idx].mean() > 25:   cl_name[ci] = "Guard-saturated"
    elif b3[idx].mean() > 10:      cl_name[ci] = "Cold-stiff"
    else:                          cl_name[ci] = "Controlled"

def panel_label(ax, letter):
    ax.text(-0.06, 1.03, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=C_TEXT, va="bottom", ha="left")

# ---- FIGURE ----
fig = plt.figure(figsize=(182/25.4, 198/25.4), facecolor="white")
outer = fig.add_gridspec(3, 1, left=0.245, right=0.985, top=0.965, bottom=0.070,
                          hspace=0.40, height_ratios=[1.0, 1.08, 1.04])

# ========== PANEL (a): OOD response heatmap ==========
ax_a = fig.add_subplot(outer[0])
panel_label(ax_a, "a")

cols = ["FEA\nescalation", "Guard\nevents", "B3 min\nmargin", "SCR", "Mean\nreward"]
heat = np.column_stack([
    esc / 100.0, guard / max(guard.max(), 1),
    np.clip(1.0 - np.log10(np.maximum(b3, 0.5)), 0, 1),
    1.0 - scr, np.clip(-rew / 1.5, 0, 1),
])
txt = np.array([
    [f"{v:.0f}%" for v in esc], [f"{v:.0f}"  for v in guard],
    [f"{v:.2f}"  for v in b3],  [f"{v:.2f}"  for v in scr],
    [f"{v:+.2f}" for v in rew],
]).T

cmap = LinearSegmentedColormap.from_list(
    "risk", [(0.0, W[2]), (0.35, "#F5F5F5"), (0.65, W[4]), (1.0, W[5])])
ax_a.imshow(heat, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper")
for i in range(8):
    for j in range(5):
        bg = heat[i, j]
        tc = "white" if bg > 0.72 else C_TEXT
        wgt = "bold" if bg > 0.45 else "normal"
        ax_a.text(j, i, txt[i, j], ha="center", va="center",
                  fontsize=8.0, color=tc, fontweight=wgt)

ax_a.set_yticks(range(8))
ax_a.set_yticklabels([LABEL[c] for c in SCENARIO_ORDER], fontsize=8.2)
ax_a.set_xticks(range(5)); ax_a.set_xticklabels(cols, fontsize=8.2)
ax_a.tick_params(length=0)
for spine in ax_a.spines.values(): spine.set_visible(False)
for b in [3, 5, 7]:
    ax_a.axhline(b - 0.5, color=C_GRID, lw=1.5, zorder=5)

# ========== PANEL (b): Clustered reward bars (Ward, k=4) ==========
ax_r = fig.add_subplot(outer[1])
panel_label(ax_r, "b")

# Get dendrogram leaf order (no visual dendrogram)
dn = dendrogram(Z, no_plot=True)
leaf_order = dn["leaves"][::-1]  # top-to-bottom for horizontal bars

# Horizontal reward bars, ordered by clustering
y_pos = range(8)
for yi, li in enumerate(leaf_order):
    r = rew[li]; ci = cl[li]; col = C_CLUSTER[ci % len(C_CLUSTER)]
    ax_r.barh(yi, r, height=0.58, color=col, alpha=0.78, edgecolor="white", lw=0.3, zorder=2)
    g = guard[li]; e = esc[li]
    x_ann = max(r, 0) + 0.14
    if g > 0 and e > 0:
        ax_r.text(x_ann, yi + 0.15, f"Guard: {g:.0f}", fontsize=7.2, color=C_TEXT,
                  va="center", ha="left")
        ax_r.text(x_ann, yi - 0.15, f"FEA esc: {e:.0f}%", fontsize=7.2, color=C_TEXT,
                  va="center", ha="left")
    elif g > 0:
        ax_r.text(x_ann, yi, f"Guard: {g:.0f}", fontsize=7.2, color=C_TEXT,
                  va="center", ha="left")
    elif e > 0:
        ax_r.text(x_ann, yi, f"FEA esc: {e:.0f}%", fontsize=7.2, color=C_TEXT,
                  va="center", ha="left")

# Cluster group separators
prev = cl[leaf_order[0]]
for i in range(1, 8):
    if cl[leaf_order[i]] != prev:
        ax_r.axhline(i - 0.5, color=C_GRID, ls="-", lw=1.2, zorder=0)
        prev = cl[leaf_order[i]]

# Cluster labels on right
px = max(rew.max(), abs(rew.min())) + 0.55
prev_c = -1; c_start = 0
for i in range(8):
    ci = cl[leaf_order[i]]
    if ci != prev_c:
        if prev_c >= 0:
            mid_y = (c_start + i - 1) / 2
            ax_r.text(px, mid_y, cl_name[prev_c], fontsize=7.8,
                      color=C_CLUSTER[prev_c % len(C_CLUSTER)],
                      fontweight="bold", va="center", ha="left")
        c_start = i; prev_c = ci
if prev_c >= 0:
    mid_y = (c_start + 7) / 2
    ax_r.text(px, mid_y, cl_name[prev_c], fontsize=7.8,
              color=C_CLUSTER[prev_c % len(C_CLUSTER)],
              fontweight="bold", va="center", ha="left")

ax_r.axvline(0, color=C_TEXT, ls="--", lw=0.8, zorder=1)
ax_r.set_yticks(list(y_pos))
ax_r.set_yticklabels([LABEL[SCENARIO_ORDER[i]] for i in leaf_order], fontsize=8.0)
ax_r.set_xlabel("Mean step reward", fontsize=9.5)
ax_r.set_xlim(-1.80, px + 0.30)
ax_r.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0])
ax_r.tick_params(axis="x", labelsize=8.5)
ax_r.tick_params(axis="y", labelsize=8.5)
ax_r.grid(axis="x", color=C_GRID, lw=0.35, alpha=0.55, zorder=0)

# ========== PANEL (c): B3 lollipop + SCR bars ==========
gs_c = gspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2],
                                      height_ratios=[1.0, 0.65], hspace=0.06)

# Top: B3 lollipop
ax_b3 = fig.add_subplot(gs_c[0, 0])
panel_label(ax_b3, "c")
y = np.arange(8)[::-1]
for i, yi in enumerate(y):
    cid = SCENARIO_ORDER[i]; v = b3[i]
    clr = C_LIMIT if v < 1.0 else CAT_COLOR[agg[cid]["category"]]
    ax_b3.plot([1.0, v], [yi, yi], color=clr, lw=2.2, alpha=0.50, solid_capstyle="round", zorder=2)
    ax_b3.scatter([v], [yi], s=38, color=clr, edgecolor="white", lw=0.6, zorder=3)
    ha = "center" if v < 3.0 else ("left" if v < 100 else "right")
    if v < 3.0: xoff = v; yoff = 0.22
    else:       xoff = v * 1.10; yoff = 0
    ax_b3.text(xoff, yi + yoff, f"{v:.2f}", fontsize=7.8, ha=ha, va="center",
               color=clr, fontweight="bold")

ax_b3.axvline(1.0, color=C_TEXT, ls="--", lw=0.8, zorder=1)
ax_b3.text(1.06, 7.3, "limit", fontsize=7.2, color=C_TEXT)
ax_b3.set_xscale("log"); ax_b3.set_xlim(0.45, 250)
ax_b3.set_xticks([1, 10, 100])
ax_b3.set_yticks(y)
ax_b3.set_yticklabels([LABEL[c] for c in SCENARIO_ORDER], fontsize=8.0)
ax_b3.tick_params(axis="y", pad=3, labelsize=8.5)
ax_b3.tick_params(axis="x", labelsize=8.5)
ax_b3.set_xlabel("B3 permanent deformation margin (log scale)", fontsize=9.5, labelpad=2)
ax_b3.grid(axis="x", color=C_GRID, lw=0.35, alpha=0.55, zorder=0)

# Bottom: SCR bars
ax_scr = fig.add_subplot(gs_c[1, 0])
for i, yi in enumerate(y):
    cid = SCENARIO_ORDER[i]; v = scr[i]
    clr = C_LIMIT if v < 1.0 else CAT_COLOR[agg[cid]["category"]]
    ax_scr.barh(yi, v, height=0.55, color=clr, alpha=0.78, edgecolor="white", lw=0.3, zorder=3)
    ax_scr.text(v + 0.018, yi, f"{v:.2f}", fontsize=7.8, ha="left", va="center",
                color=clr, fontweight="bold")

ax_scr.axvline(1.0, color=C_TEXT, ls="--", lw=0.8, zorder=1)
ax_scr.set_xlim(0.60, 1.20)
ax_scr.set_xticks([0.7, 0.8, 0.9, 1.0])
ax_scr.set_yticks([])
ax_scr.tick_params(axis="x", labelsize=8.5)
ax_scr.set_xlabel("SCR (specification compliance rate)", fontsize=9.5, labelpad=2)
ax_scr.grid(axis="x", color=C_GRID, lw=0.35, alpha=0.55, zorder=0)

# Category legend for whole panel (c), upper-right
leg_items = [
    Line2D([0],[0], marker="s", color="none", markerfacecolor=CAT_COLOR["subgrade"], markersize=7.5, label="Subgrade"),
    Line2D([0],[0], marker="s", color="none", markerfacecolor=CAT_COLOR["material"], markersize=7.5, label="Material"),
    Line2D([0],[0], marker="s", color="none", markerfacecolor=CAT_COLOR["traffic"],  markersize=7.5, label="Traffic"),
    Line2D([0],[0], marker="s", color="none", markerfacecolor=CAT_COLOR["climate"],  markersize=7.5, label="Climate"),
]
ax_b3.legend(handles=leg_items, loc="upper right", bbox_to_anchor=(1.00, 1.28),
             ncol=4, frameon=False, fontsize=7.8, handletextpad=0.3, columnspacing=1.2)

# Export
for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 450})]:
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
print(f"[OK] {OUT}")
print(f"Clusters: {cl_name}")


