"""
Fig. 2 | Sequential design trajectory and delivered-design selection
ALL REAL DATA from ltpp_inference_steps_20260625_133730.jsonl (16_1010, seed=0).
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family":"DejaVu Sans","font.size":9,
    "axes.linewidth":0.6,"axes.edgecolor":"#555",
    "savefig.dpi":300,"figure.dpi":150,
})

# ============== REAL TRAJECTORY (16_1010 seed=0, from 0625 inference) ==============
STEPS = np.arange(21)

# DSR: from ltpp_inference_steps JSONL
DSR  = np.array([1.000,0.9165,0.9951,1.000,1.000,1.000,1.000,1.000,1.000,1.000,
                 1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000,1.000])
# SCR (running): from JSONL scr_running
SCR  = np.array([1.000,0.5000,0.3333,0.5000,0.6000,0.6667,0.7143,0.7500,0.7778,0.8000,
                 0.8182,0.8333,0.8462,0.8571,0.8667,0.8750,0.8824,0.8889,0.8947,0.9000,0.9048])
# Reward: from JSONL
REW  = np.array([np.nan,0.0371,0.0666,0.8088,0.8478,0.8476,0.8430,0.8401,0.8368,0.8338,
                 0.8309,0.8282,0.8257,0.8222,0.8192,0.8192,float('nan'),float('nan'),-1.500,-1.500,-1.500])
# Cost trajectory: from delivered-design pipeline
COST = np.array([42.09,35.88,36.95,38.31,39.31,40.36,41.41,42.44,43.39,44.29,
                 45.00,45.54,46.13,46.74,47.18,47.53,47.90,48.40,48.40,48.40,48.40])

# B1/B3/B4 margins (steps 1-20, from JSONL evaluation.margins)
B1 = np.array([0.917,0.995,1.159,1.325,1.499,1.680,1.866,2.048,2.224,2.394,
               3.861,2.605,2.763,2.903,3.034,3.156,3.269,3.269,3.269,3.269])
B3 = np.array([1.947,1.860,1.788,1.732,1.688,1.652,1.621,1.591,1.565,1.544,
               1.633,1.518,1.502,1.488,1.476,1.465,1.455,1.455,1.455,1.455])
B4 = np.array([1.944,1.960,2.039,2.116,2.193,2.271,2.348,2.424,2.497,2.568,
               2.922,2.675,2.744,2.808,2.870,2.931,2.990,2.990,2.990,2.990])
STEPS_B = np.arange(1, 21)  # 20 inference steps

H_INIT  = [4.0, 6.0, 8.0, 30.0, 25.0]
H_FINAL = [2.0, 9.8, 11.6, 29.6, 30.0]
LLAB    = ["AC upper","AC mid","AC lower","Base","Subbase"]
ACOL    = ["#a6cee3","#4393c3","#2166ac","#b07d3c","#e6d5b8"]

C1="#2C7BB6"; C2="#D7191C"; C3="#7B3294"; C4="#009E73"

# ========================== FIGURE ==========================================
fig = plt.figure(figsize=(11.5, 8.2))
gs = fig.add_gridspec(3, 4, height_ratios=[0.85, 1.05, 1.0], hspace=0.55, wspace=0.48)

# ---- Panel A: MDP schematic ----
axA = fig.add_subplot(gs[0,:]); axA.axis("off")
axA.set_title("a", fontsize=11, fontweight="bold", loc="left")
boxes = [
    ("Initial design $x_0$\n(five-layer)", 0.02, C1),
    ("State $s_t$\n[log B1, log B2,\nlog B3, log B4,\nt/T, ||a||]", 0.21, C3),
    ("Action $a_t=(\\Delta h,\\Delta E)$\nbounded, layer-wise", 0.42, C4),
    ("Physics & spec\nFEA/surrogate\n$\\to$ margins", 0.62, C2),
    ("Reward $r_t$\ncompliance+cost\n+feasibility", 0.82, "#B8860B"),
]
for txt, xx, c in boxes:
    axA.add_patch(plt.Rectangle((xx, 0.30), 0.15, 0.45, transform=axA.transAxes,
                  fc="white", ec=c, lw=1.6, zorder=2))
    axA.text(xx+0.075, 0.52, txt, transform=axA.transAxes, ha="center", va="center",
             fontsize=7.2, zorder=3)
for x0 in [0.17, 0.38, 0.58, 0.78]:
    axA.annotate("", (x0+0.04, 0.52), (x0, 0.52), xycoords=axA.transAxes,
                 arrowprops=dict(arrowstyle="->", lw=1.3, color="#444"))
axA.annotate("next state $S_{t+1}$", (0.285, 0.18), (0.66, 0.18), xycoords=axA.transAxes,
             ha="center", fontsize=7, color="#666",
             arrowprops=dict(arrowstyle="->", lw=1.1, color="#888",
                             connectionstyle="arc3,rad=0.25"))
axA.text(0.5, 0.04, "Fixed horizon: 20 adjustment steps -> 21 evaluated states",
         transform=axA.transAxes, ha="center", fontsize=7.5, style="italic", color="#555")

# ---- Panel B: four real trajectories ----
# B1: DSR + Running SCR
axB1 = fig.add_subplot(gs[1,0])
axB1.plot(STEPS, DSR, "o-", color=C1, lw=1.6, ms=3.5, markeredgecolor="white", markeredgewidth=0.2, label="DSR")
axB1.plot(STEPS, SCR, "s--", color=C2, lw=1.3, ms=3.5, markeredgecolor="white", markeredgewidth=0.2, label="Running SCR")
axB1.axhline(1.0, color="#888", ls="--", lw=0.7, alpha=0.7)
axB1.axvspan(17.5, 20.5, color="#fff3e0", alpha=0.35, zorder=0)
axB1.axvline(3, color="#003366", ls="--", lw=0.7, alpha=0.4)
axB1.set_xlabel("Step $t$", fontsize=8)
axB1.set_ylabel("DSR / SCR", fontsize=8)
axB1.set_title("b", fontsize=10, fontweight="bold", loc="left")
axB1.legend(fontsize=6.5, loc="lower right", framealpha=0.9, edgecolor="#ddd")
axB1.grid(True, ls=":", lw=0.4, color="#ddd", zorder=0)
axB1.tick_params(labelsize=7)

# B2: Construction cost
axB2 = fig.add_subplot(gs[1,1])
axB2.plot(STEPS, COST, "o-", color=C4, lw=1.8, ms=3.5, markeredgecolor="white", markeredgewidth=0.2)
axB2.axvspan(17.5, 20.5, color="#fff3e0", alpha=0.35, zorder=0)
axB2.axvline(3, color="#003366", ls="--", lw=0.7, alpha=0.4)
# Star marker at delivered design (step 3)
axB2.scatter(3, COST[3], s=140, marker="*", color="#003366", zorder=5, edgecolor="white", linewidth=0.3)
axB2.annotate("Delivered\n38.3 USD m$^{-2}$", xy=(3, COST[3]), xytext=(11.5, 40),
            fontsize=6.5, color="black", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
# Hollow marker at step 1
axB2.scatter(1, COST[1], s=80, marker="o", facecolor="white", edgecolor="#888", lw=1.5, zorder=4)
axB2.annotate("Lowest-cost state\nnon-compliant", xy=(1, COST[1]), xytext=(12, 36.5),
            fontsize=7, color="black", ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.3, alpha=0.92))
axB2.set_xlabel("Step $t$", fontsize=8)
axB2.set_ylabel("Cost (USD m$^{-2}$)", fontsize=8)
axB2.set_ylim(34, 50)
axB2.set_title(" ", fontsize=9, loc="left")
axB2.grid(True, ls=":", lw=0.4, color="#ddd", zorder=0)
axB2.tick_params(labelsize=7)

# B3: Reward trajectory
axB3 = fig.add_subplot(gs[1,2])
axB3.plot(STEPS, REW, "D-", color=C3, lw=1.6, ms=3.5, markeredgecolor="white", markeredgewidth=0.2)
axB3.axhline(0, color="#ccc", ls="--", lw=0.7, alpha=0.7)
axB3.axvspan(17.5, 20.5, color="#fff3e0", alpha=0.35, zorder=0)
axB3.axvline(3, color="#003366", ls="--", lw=0.7, alpha=0.4)
axB3.annotate("NumericalGuard\ninterceptions\n(reward = -1.5)", xy=(13, -0.8), fontsize=7,
              color="black", ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.3, alpha=0.92))
axB3.set_xlabel("Step $t$", fontsize=8)
axB3.set_ylabel("Reward", fontsize=8)
axB3.set_ylim(-1.8, 1.1)
axB3.set_title(" ", fontsize=9, loc="left")
axB3.grid(True, ls=":", lw=0.4, color="#ddd", zorder=0)
axB3.tick_params(labelsize=7)

# B4: Applicable compliance margins
axB4 = fig.add_subplot(gs[1,3])
axB4.plot(STEPS_B, B1, "o-", color="#2C7BB6", lw=1.2, ms=2.8, markeredgecolor="white", markeredgewidth=0.1, label="B1, asphalt fatigue")
axB4.plot(STEPS_B, B3, "s-", color="#D7191C", lw=1.2, ms=2.8, markeredgecolor="white", markeredgewidth=0.1, label="B3, permanent deformation")
axB4.plot(STEPS_B, B4, "^-", color="#009E73", lw=1.2, ms=2.8, markeredgecolor="white", markeredgewidth=0.1, label="B4, subgrade strain")
axB4.axhline(1.0, color="#888", ls="--", lw=0.7, alpha=0.7)
axB4.text(19, 1.08, "Compliance threshold", fontsize=5.5, color="#888", ha="right")
axB4.axvspan(17.5, 20.5, color="#fff3e0", alpha=0.35, zorder=0)
axB4.axvline(3, color="#003366", ls="--", lw=0.7, alpha=0.4)
axB4.set_ylim(0.8, 5.5)
axB4.set_xlabel("Step $t$", fontsize=8)
axB4.set_ylabel("Margin", fontsize=8)
axB4.set_title("   Applicable compliance margins", fontsize=8, loc="left")
axB4.legend(fontsize=5.5, loc="upper left", framealpha=0.9, edgecolor="#ddd")
axB4.grid(True, ls=":", lw=0.4, color="#ddd", zorder=0)
axB4.tick_params(labelsize=7)

# ---- Panel C left: Layer structure comparison ----
axC1 = fig.add_subplot(gs[2,:2])
y = np.arange(5); h = 0.33
for i in range(5):
    axC1.barh(y[i]+h/2, H_INIT[i], h*0.8, left=0, color=ACOL[i], edgecolor="white",
              lw=0.4, alpha=0.45, zorder=2)
    axC1.barh(y[i]-h/2, H_FINAL[i], h*0.8, left=0, color=ACOL[i], edgecolor="white",
              lw=0.4, zorder=3)
    axC1.text(H_INIT[i]+0.8, y[i]+h/2, f"{H_INIT[i]:.0f}", va="center", fontsize=7.2, color="#888")
    axC1.text(H_FINAL[i]+0.8, y[i]-h/2, f"{H_FINAL[i]:.1f}", va="center", fontsize=7.2,
              color="#222", fontweight="bold")
axC1.set_yticks(y); axC1.set_yticklabels(LLAB, fontsize=7.5)
axC1.set_xlabel("Thickness (cm)", fontsize=9)
axC1.set_xlim(0, 38)
axC1.set_title("c", fontsize=11, fontweight="bold", loc="left")
from matplotlib.patches import Patch
leg = [Patch(fc=ACOL[i], label=LLAB[i]) for i in range(5)]
axC1.legend(handles=leg, fontsize=6.5, loc="lower right", ncol=2, framealpha=0.9, edgecolor="#ddd")
axC1.text(0.02, 0.96, "Initial (step 0)", transform=axC1.transAxes,
          fontsize=7, color="#888", va="top", style="italic")
axC1.text(0.02, 0.06, "Delivered (step 3)", transform=axC1.transAxes,
          fontsize=7, color="#222", va="bottom", fontweight="bold", style="italic")
axC1.spines[["top","right"]].set_visible(False)
axC1.grid(True, axis="x", ls=":", lw=0.4, color="#ddd", zorder=0)
axC1.tick_params(labelsize=7)

# ---- Panel C right: text summary ----
axC2 = fig.add_subplot(gs[2,2:]); axC2.axis("off")
axC2.set_title(" ", fontsize=9, loc="left")
txt = (
    "16_1010 (flexible, E_sub = 78 MPa)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"  Initial DSR:  1.000\n"
    f"  Lowest DSR:   0.917 (step 1)\n"
    f"  Final DSR:    1.000 (step 3+)\n"
    f"  Final SCR:    0.905\n"
    f"  Delivered cost: 38.3 USD m$^{{-2}}$\n"
    "\n"
    "  AC thickness:  18.0 -> 17.3 cm\n"
    "  AC-1 (upper):  4.0 -> 2.0 cm\n"
    "  AC-2 (mid):    6.0 -> 6.7 cm\n"
    "  AC-3 (lower):  8.0 -> 8.6 cm\n"
    "  Base (GAB):    30.0 -> 29.8 cm\n"
    "  Subbase (GAB): 25.0 -> 25.6 cm\n"
    "\n"
    "  Guard interceptions: 3 (steps 18-20)\n"
    "  Policy probes lower bound\n"
    "  of base modulus; Guard holds\n"
    "  design, preventing unsafe action."
)
axC2.text(0.02, 0.95, txt, transform=axC2.transAxes, fontsize=7.5, va="top",
          family="monospace", color="#333",
          bbox=dict(boxstyle="round,pad=0.5", fc="#fafafa", ec="#ddd", lw=0.5))

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig2_convergence_real.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
out_pdf = out.replace(".png", ".pdf")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Saved PNG: {out}")
print(f"Saved PDF: {out_pdf}")
print(f"Guard interceptions: steps 18-20 ({sum(1 for r in REW if r == -1.5)} total)")
