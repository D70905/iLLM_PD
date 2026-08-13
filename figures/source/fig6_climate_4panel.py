"""
Fig. 6 — Climate sensitivity of AC fatigue life (4-panel, narrative order)
Nature-figure compliant. Python/matplotlib.

Narrative: (a) life ratio -> strain mechanism -> material root -> fatigue consequence
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np; import pandas as pd; from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'; plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7; plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.5; plt.rcParams['ytick.major.width'] = 0.5

BASE = Path('D:/iLLM_PD_new')
CSV = BASE / 'experiments' / 'batch_climate_12sections_summary.csv'
MONTHLY = BASE / 'experiments' / '16_1010_climate_fatigue.csv'
OUT  = BASE / 'plot' / 'fig6_climate_4panel'

ZONE_COLOR = {"Wet-Freeze":"#0072B2","Dry-Freeze":"#56B4E9",
              "Dry-NoFreeze":"#E69F00","Wet-NoFreeze":"#D55E00"}
ZONE_MARKER = {"Wet-Freeze":"o","Dry-Freeze":"s","Dry-NoFreeze":"^","Wet-NoFreeze":"D"}
ZONE_ORDER = ["Wet-Freeze","Dry-Freeze","Dry-NoFreeze","Wet-NoFreeze"]

# Load
df = pd.read_csv(CSV).sort_values("MAAT_C").reset_index(drop=True)
_b = df[df.fixed_over_climate < 1].iloc[-1]
_a = df[df.fixed_over_climate >= 1].iloc[0]
CROSS = _b.MAAT_C + (1.0 - _b.fixed_over_climate) * (_a.MAAT_C - _b.MAAT_C) / \
        (_a.fixed_over_climate - _b.fixed_over_climate)

valid_m = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mdf = pd.read_csv(MONTHLY); mdf = mdf[mdf.month.isin(valid_m)].copy()
for c in ['T_air_C','T_pav_C','E_ac_equiv_MPa','eps_a_microstrain','Nf']:
    mdf[c] = pd.to_numeric(mdf[c], errors='coerce')
mdf = mdf.reset_index(drop=True)

def al(ax,lab,x=-0.12,y=1.05):
    ax.text(x,y,lab,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top',ha='left')
def sp(fig,fn):
    for ext,kk in [('svg',{}),('pdf',{}),('png',{'dpi':450})]:
        fig.savefig(f'{fn}.{ext}',bbox_inches='tight',pad_inches=0.02,**kk)
    print(f'[OK] {fn}')

fig=plt.figure(figsize=(180/25.4,175/25.4)); fig.patch.set_facecolor('white')
gs=fig.add_gridspec(2,2,hspace=0.34,wspace=0.30,height_ratios=[1,0.95],
                     left=0.09,right=0.98,top=0.96,bottom=0.08)

# ===== (c) Material root: AC modulus vs pavement temp (16_1010) =====
ax_a=fig.add_subplot(gs[1,0]); al(ax_a,'c')
x_m=np.arange(12)
ax_a.bar(x_m,mdf.E_ac_equiv_MPa.values,width=0.55,color='#4C5C92',alpha=0.72,ec='white',lw=0.3,zorder=3)
ax_t=ax_a.twinx()
ax_t.plot(x_m,mdf.T_pav_C.values,'o-',color='#D55E00',lw=1.5,markersize=5,
          markerfacecolor='white',markeredgewidth=1.2,zorder=4)
# 20 C design reference line
ax_t.axhline(20,color='#aaa',ls=':',lw=0.7)

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax_a.set_xticks(x_m); ax_a.set_xticklabels(months,fontsize=5.5,rotation=45,ha='right')
ax_a.set_ylabel('$E^*_{\\mathrm{ac}}$ (MPa)',fontsize=7.5,color='#4C5C92')
ax_a.tick_params(axis='y',labelcolor='#4C5C92',labelsize=6)
ax_t.set_ylabel('$T_{\\mathrm{pav}}$ ($\\degree$C)',fontsize=7.5,color='#D55E00')
ax_t.tick_params(axis='y',labelcolor='#D55E00',labelsize=6)
ax_a.set_xlim(-0.6,11.6); ax_a.spines['top'].set_visible(False); ax_t.spines['top'].set_visible(False)
ax_a.set_ylim(0, 28000); ax_t.set_ylim(-10, 35)
ax_a.set_title('Section 16_1010 (Dry-Freeze, MAAT = 6.7 $\\degree$C)',fontsize=6.5,fontweight='bold',pad=2)

# Legend: upper right
leg_a=[Line2D([0],[0],color='#4C5C92',lw=4,label='$E^*_{\\mathrm{ac}}$ (MPa)'),
       Line2D([0],[0],marker='o',color='#D55E00',markerfacecolor='white',
              markeredgewidth=1.2,markersize=5,label='$T_{\\mathrm{pav}}$ ($\\degree$C)')]
ax_a.legend(handles=leg_a,loc='upper right',frameon=False,fontsize=5.5,handlelength=1.2)

# ===== (b) Strain mechanism: envelope vs MAAT =====
ax_b=fig.add_subplot(gs[0,1]); al(ax_b,'b')

# Slight x-jitter for overlapping sections (16_1010 ~6.7 and 27_1085 ~6.7)
x_jitter={0:0,1:-0.18,2:0.18,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0}
for i,(_,r) in enumerate(df.iterrows()):
    c=ZONE_COLOR[r.climate_zone]; xv=r.MAAT_C+x_jitter.get(i,0)
    ax_b.plot([xv,xv],[r.eps_a_min_ue,r.eps_a_max_ue],color=c,lw=2.6,alpha=0.45,solid_capstyle='round',zorder=2)
    ax_b.scatter([xv],[r.eps_a_max_ue],s=11,marker='_',color=c,zorder=3,lw=1.0)
    ax_b.scatter([xv],[r.eps_a_min_ue],s=11,marker='_',color=c,zorder=3,lw=1.0)
    ax_b.scatter([xv],[r.eps_a_fixed_ue],s=26,marker='D',facecolor='white',edgecolor=c,linewidth=1.0,zorder=4)

ax_b.axvline(CROSS,color='0.55',lw=0.8,ls=':',zorder=1)
ax_b.set_xlabel('Mean annual air temperature, MAAT ($\\degree$C)',fontsize=7.5)
ax_b.set_ylabel('Asphalt-bottom tensile strain,\n$\\varepsilon_a$ ($\\mu\\varepsilon$)',fontsize=7.5)
ax_b.set_xlim(5.5,25.5); ax_b.set_ylim(12,66)
ax_b.spines[['top','right']].set_visible(False)
b_h=[Line2D([0],[0],color='0.45',lw=2.6,alpha=0.6,label='winter$\\rightarrow$summer envelope'),
     Line2D([0],[0],marker='D',color='none',markerfacecolor='white',
            markeredgecolor='0.3',markeredgewidth=1.0,markersize=5,label='fixed-temp. $\\varepsilon_a$')]
ax_b.legend(handles=b_h,loc='upper left',frameon=False,handletextpad=0.5,labelspacing=0.3,fontsize=5.5)

# ===== (a) Bias pattern: life ratio vs MAAT =====
ax_c=fig.add_subplot(gs[0,0]); al(ax_c,'a')
ax_c.axhspan(0.30,1.0,color='#0072B2',alpha=0.05,zorder=0)
ax_c.axhspan(1.0,1.60,color='#D55E00',alpha=0.06,zorder=0)
ax_c.axhline(1.0,color='0.35',lw=0.8,ls='--',zorder=1)
ax_c.axvline(CROSS,color='0.55',lw=0.8,ls=':',zorder=1)
xs=np.linspace(df.MAAT_C.min(),df.MAAT_C.max(),200)
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    sm=lowess(df.fixed_over_climate,df.MAAT_C,frac=0.75,return_sorted=True)
    ax_c.plot(sm[:,0],sm[:,1],color='0.5',lw=1.0,alpha=0.8,zorder=2)
except Exception:
    p=np.poly1d(np.polyfit(df.MAAT_C,df.fixed_over_climate,3))
    ax_c.plot(xs,p(xs),color='0.5',lw=1.0,alpha=0.8,zorder=2)
for z in ZONE_ORDER:
    sub=df[df.climate_zone==z]
    if len(sub):
        ax_c.scatter(sub.MAAT_C,sub.fixed_over_climate,s=34,marker=ZONE_MARKER[z],
                     facecolor=ZONE_COLOR[z],edgecolor='white',linewidth=0.6,zorder=4)

# Moved text UP away from trend line
ax_c.text(8.0,0.70,'conservative\n(cold; real life longer)',fontsize=5.8,color='#0072B2',
          ha='left',va='top',linespacing=1.2)
ax_c.text(21.5,1.48,'optimistic\n(hot; real life shorter)',fontsize=5.8,color='#D55E00',
          ha='right',va='top',linespacing=1.2)
ax_c.annotate('ratio = 1\nMAAT $\\approx$ %.1f $\\degree$C'%CROSS,
              xy=(CROSS,1.0),xycoords='data',xytext=(CROSS-6.2,1.18),textcoords='data',
              fontsize=5.8,ha='center',va='center',color='0.25',
              arrowprops=dict(arrowstyle='->',color='0.45',lw=0.7))
ax_c.set_xlabel('Mean annual air temperature, MAAT ($\\degree$C)',fontsize=7.5)
ax_c.set_ylabel('Fixed-temp. / climate-resolved\nAC fatigue life (ratio)',fontsize=7.5)
ax_c.set_xlim(5.5,25.5); ax_c.set_ylim(0.30,1.55)
ax_c.set_yticks([0.4,0.6,0.8,1.0,1.2,1.4]); ax_c.spines[['top','right']].set_visible(False)
z_h=[Line2D([0],[0],marker=ZONE_MARKER[z],color='none',markerfacecolor=ZONE_COLOR[z],
            markeredgecolor='white',markeredgewidth=0.5,markersize=5,label=z)
     for z in ZONE_ORDER if (df.climate_zone==z).any()]
ax_c.legend(handles=z_h,title='LTPP climate zone',loc='upper left',frameon=False,
            handletextpad=0.3,labelspacing=0.25,fontsize=5.5,title_fontsize=6)

# ===== (d) Fatigue consequence: Nf 1:1 =====
ax_d=fig.add_subplot(gs[1,1]); al(ax_d,'d')
nf_max=max(df.Nf_fixed.max(),df.Nf_climate_eff.max())*1.1
ax_d.plot([0,nf_max],[0,nf_max],color='0.35',lw=0.8,ls='--',zorder=1)
# 2.5x reference line (cold-section cluster)
ax_d.plot([0,nf_max/2.5],[0,nf_max],color='#0072B2',lw=0.8,ls='--',alpha=0.8,zorder=1)

for z in ZONE_ORDER:
    sub=df[df.climate_zone==z]
    if len(sub):
        ax_d.scatter(sub.Nf_fixed/1e8,sub.Nf_climate_eff/1e8,s=34,marker=ZONE_MARKER[z],
                     facecolor=ZONE_COLOR[z],edgecolor='white',linewidth=0.6,zorder=3)

ax_d.fill_between([0,nf_max/1e8],[0,nf_max/1e8],10,alpha=0.04,color='#0072B2')
ax_d.fill_between([0,nf_max/1e8],0,[0,nf_max/1e8],alpha=0.04,color='#D55E00')
# Move text away from 1:1 line
ax_d.text(6.5,2.0,'optimistic\n(fixed > real)',fontsize=5.5,color='#D55E00',ha='right',linespacing=1.2)
ax_d.text(3.0,5.0,'conservative\n(fixed < real)',fontsize=5.5,color='#0072B2',ha='left',linespacing=1.2)

ax_d.set_xlabel('Fixed-temp. AC fatigue life\n$N_{f,\\mathrm{fixed}}$ ($\\times 10^8$)',fontsize=7.5)
ax_d.set_ylabel('Climate-resolved AC fatigue life\n$N_{f,\\mathrm{climate}}$ ($\\times 10^8$)',fontsize=7.5)
ax_d.set_xlim(0.5,7.5); ax_d.set_ylim(0.5,7.5); ax_d.set_aspect('equal')
ax_d.spines[['top','right']].set_visible(False)

sp(fig,str(OUT)); print(f'Crossover MAAT = {CROSS:.2f} C')
