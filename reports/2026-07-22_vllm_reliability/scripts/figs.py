#!/usr/bin/env python3
"""Figures for reportC. Light-mode PNGs, validated reference palette."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

S=Path("/tmp/claude-3014/-workspace1-zhijun/6a801ac6-4b1b-49b8-98d6-d62501f4249c/scratchpad")
F=S/"figs"; F.mkdir(exist_ok=True)
ORDER=["qwen0.8b","qwen2b","qwen2b-fkpp","qwen9b","qwen9b-fkpp","gemma-e4b","intern1b","intern2b"]
MV=["MV_FWD","MV_BACK","MV_LEFT","MV_RIGHT","MV_UP","MV_DOWN"]; EV=["GRASP","RELEASE","DONE"]
SHORT=[a.replace("MV_","") for a in MV]
D={n:[json.loads(l) for l in open(S/"probs"/f"{n}.jsonl")][1:] for n in ORDER}
SUM=json.load(open(S/"probs"/"summary.json"))

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e3e2de"
C1="#2a78d6"; C2="#eb6834"; C3="#1baf7a"; CRIT="#d03b3b"; GOOD="#0ca30c"
SEQ=LinearSegmentedColormap.from_list("blue",["#fcfcfb","#cde2fb","#9ec5f4","#5598e7","#2a78d6","#256abf","#0d366b"])
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,
    "axes.edgecolor":GRID,"grid.color":GRID,"font.size":9,"axes.titlesize":10,
    "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":150,"savefig.bbox":"tight"})
EVSTEP=[24,43,49]
def dist(i): return min(abs(i-e) for e in EVSTEP)

# ── FIG 1: top1 confidence, MV group vs event group ──────────────────────────
fig,ax=plt.subplots(figsize=(9,4.2))
w=0.34
for k,(grp,col,lab) in enumerate([(MV,C1,"MV_* direction samples (n=47)"),(EV,C2,"event samples GRASP/RELEASE/DONE (n=3)")]):
    data=[[r["top1_prob"] for r in D[n] if r["label"] in grp] for n in ORDER]
    pos=[i+(k-0.5)*w for i in range(len(ORDER))]
    bp=ax.boxplot(data,positions=pos,widths=w*0.86,patch_artist=True,showfliers=False,
                  medianprops=dict(color=SURF,lw=2),whiskerprops=dict(color=col,lw=1.4),
                  capprops=dict(color=col,lw=1.4),boxprops=dict(facecolor=col,edgecolor=SURF,lw=2))
    for i,d in enumerate(data):
        x=np.random.RandomState(k*7+i).normal(pos[i],0.035,len(d))
        ax.scatter(x,d,s=11,color=col,alpha=.5,edgecolors=SURF,linewidths=.5,zorder=3)
    ax.plot([],[],'s',color=col,label=lab,ms=8)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
ax.set_ylabel("top-1 action probability"); ax.set_ylim(0.35,1.03); ax.grid(axis="y",lw=.7); ax.set_axisbelow(True)
ax.set_title("Top-1 confidence by model: direction samples vs event samples",loc="left",color=INK,pad=10)
ax.legend(frameon=False,loc="lower left",fontsize=8.5)
fig.savefig(F/"fig1_confidence_box.png"); plt.close(fig)

# ── FIG 2: event probability mass on MV samples, stratified ─────────────────
fig,ax=plt.subplots(figsize=(9,4.4))
FLOOR=1e-10
for k,(sel,col,lab,mk) in enumerate([
        (lambda r: dist(r["i"])>=2, C1, "distance to event step >= 2 (n=42)","o"),
        (lambda r: dist(r["i"])==1, CRIT,"distance to event step = 1 (n=5)","D")]):
    for i,n in enumerate(ORDER):
        vals=[max(sum(r["first_token_probs"][e] for e in EV),FLOOR)
              for r in D[n] if r["label"] in MV and sel(r)]
        x=np.random.RandomState(k*13+i).normal(i+(k-0.5)*0.26,0.045,len(vals))
        ax.scatter(x,vals,s=26 if k else 14,color=col,alpha=.75,marker=mk,
                   edgecolors=SURF,linewidths=.6,zorder=3+k)
    ax.plot([],[],mk,color=col,label=lab,ms=7)
ax.axhline(0.01,color=INK2,lw=1,ls="--",zorder=1)
ax.text(5.28,0.014,"1% trigger threshold",fontsize=8,color=INK2,ha="right")
ax.set_yscale("log"); ax.set_ylim(FLOOR/3,2.2)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
ax.set_ylabel("P(GRASP)+P(RELEASE)+P(DONE)   (log scale)")
ax.set_title("Probability mass on event tokens across the 47 direction samples (false-trigger risk)",loc="left",color=INK,pad=10)
ax.grid(axis="y",lw=.7); ax.set_axisbelow(True); ax.legend(frameon=False,loc="center left",fontsize=8.5)
fig.savefig(F/"fig2_event_mass.png"); plt.close(fig)

# ── FIG 3: 6x6 direction confusion, small multiples ─────────────────────────
_nr=(len(ORDER)+2)//3
fig,axes=plt.subplots(_nr,3,figsize=(16,3.9*_nr))
fig.subplots_adjust(wspace=0.62,hspace=0.42)
for ax,n in zip(axes.ravel(),ORDER):
    C=SUM["C"][n]
    M=np.full((6,6),np.nan)
    for a,lab in enumerate(MV):
        if C[lab]["n"]==0: continue
        for b,p in enumerate(MV): M[a,b]=C[lab][p]
    im=ax.imshow(M,cmap=SEQ,vmin=0,vmax=1,aspect="equal")
    for a in range(6):
        for b in range(6):
            if np.isnan(M[a,b]):
                ax.text(b,a,"–",ha="center",va="center",color=INK2,fontsize=9); continue
            if M[a,b]>=0.005:
                ax.text(b,a,f"{M[a,b]:.2f}",ha="center",va="center",fontsize=8,
                        color=SURF if M[a,b]>0.55 else INK)
    ax.set_xticks(range(6)); ax.set_xticklabels(SHORT,fontsize=8,rotation=45,ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{s}·{SUM['C'][n][l]['n']}" for s,l in zip(SHORT,MV)],fontsize=8)
    ax.set_title(n,loc="left",color=INK)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
fig.suptitle("Direction confusion: row = ground-truth direction, column = mean P(dir | MV) assigned by the model",
             x=0.02,ha="left",fontsize=11,color=INK)
fig.colorbar(im,ax=axes,shrink=0.55,pad=0.02,label="mean conditional probability")
for _ax in axes.ravel()[len(ORDER):]: _ax.set_visible(False)
fig.savefig(F/"fig3_confusion.png",bbox_inches="tight"); plt.close(fig)

# ── FIG 4: trajectory ───────────────────────────────────────────────────────
fig,axes=plt.subplots(len(ORDER),1,figsize=(11,1.85*len(ORDER)),sharex=True)
for ax,n in zip(axes,ORDER):
    xs=[r["i"] for r in D[n]]; ys=[r["top1_prob"] for r in D[n]]
    ax.plot(xs,ys,lw=2,color=C1,zorder=2)
    ok=[(r["i"],r["top1_prob"]) for r in D[n] if r["correct"]]
    bad=[(r["i"],r["top1_prob"]) for r in D[n] if not r["correct"]]
    ax.scatter(*zip(*ok),s=26,color=C1,edgecolors=SURF,linewidths=1,zorder=3)
    if bad: ax.scatter(*zip(*bad),s=46,color=CRIT,marker="X",edgecolors=SURF,linewidths=1,zorder=4)
    for e,lab in zip(EVSTEP,["GRASP","RELEASE","DONE"]):
        ax.axvline(e,color=INK2,lw=1,ls=":",zorder=1)
    ax.set_ylim(0.3,1.05); ax.set_ylabel(n,fontsize=9,rotation=0,ha="right",va="center",labelpad=8)
    ax.grid(axis="y",lw=.7); ax.set_axisbelow(True)
for e,lab in zip(EVSTEP,["GRASP(24)","RELEASE(43)","DONE(49)"]):
    axes[0].text(e,1.09,lab,fontsize=8,color=INK2,ha="center")
axes[-1].set_xlabel("rollout step index")
axes[0].scatter([],[],s=26,color=C1,label="correct"); axes[0].scatter([],[],s=46,color=CRIT,marker="X",label="wrong")
axes[0].legend(frameon=False,fontsize=8,loc="lower left",ncol=2)
fig.suptitle("Top-1 confidence along the rollout (dotted lines = ground-truth event steps)",x=0.02,ha="left",fontsize=11,color=INK)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(F/"fig4_trajectory.png"); plt.close(fig)

# ── FIG 5: calibration, small multiples ─────────────────────────────────────
BINS=[(0,0.5),(0.5,0.7),(0.7,0.9),(0.9,0.99),(0.99,1.0001)]
LBL=["<0.5",".5-.7",".7-.9",".9-.99","≥0.99"]
_nr2=(len(ORDER)+2)//3
fig,axes=plt.subplots(_nr2,3,figsize=(12,3.2*_nr2),sharex=True,sharey=True)
for ax,n in zip(axes.ravel(),ORDER):
    xs,ys,ns,cf=[],[],[],[]
    for j,(a,b) in enumerate(BINS):
        sub=[r for r in D[n] if a<=r["top1_prob"]<b]
        if not sub: continue
        xs.append(j); ys.append(sum(r["correct"] for r in sub)/len(sub))
        ns.append(len(sub)); cf.append(sum(r["top1_prob"] for r in sub)/len(sub))
    ax.plot(range(5),[ (a+min(b,1))/2 for a,b in BINS],lw=1.4,ls="--",color=GRID,zorder=1)
    ax.plot(xs,ys,lw=2,color=C1,marker="o",ms=8,mec=SURF,mew=1.4,zorder=3)
    for x,y,c in zip(xs,ys,ns): ax.annotate(f"n={c}",(x,y),textcoords="offset points",
                                            xytext=(0,9),ha="center",fontsize=7.5,color=INK2)
    ax.set_title(f"{n}   ECE={SUM['E'][n]['ECE']:.3f}",loc="left",color=INK)
    ax.set_ylim(-0.08,1.13); ax.set_xticks(range(5)); ax.set_xticklabels(LBL,fontsize=8)
    ax.grid(axis="y",lw=.7); ax.set_axisbelow(True)
for ax in axes[:,0]: ax.set_ylabel("accuracy within bin")
for ax in axes[-1,:]: ax.set_xlabel("top-1 probability bin")
fig.suptitle("Calibration: is high confidence actually more accurate? (dashed = perfect calibration; n=50, bins are coarse)",
             x=0.02,ha="left",fontsize=11,color=INK)
for _ax in axes.ravel()[len(ORDER):]: _ax.set_visible(False)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(F/"fig5_calibration.png"); plt.close(fig)

# ── FIG 6: first-token vs direction-token confidence (B3 core) ──────────────
fig,ax=plt.subplots(figsize=(9,4.4))
w=0.34
for k,(key,col,lab) in enumerate([("tok0",C1,"first-token 4-way max P  (WHETHER to fire an event)"),
                                  ("dirs",C3,"direction-token 6-way max P  (WHICH way to move)")]):
    data=[]
    for n in ORDER:
        if key=="tok0": data.append([max(r["first_token_probs"].values()) for r in D[n]])
        else: data.append([max(x for x in r["dir_probs"].values() if x is not None)
                           for r in D[n] if r["dir_available"]])
    pos=[i+(k-0.5)*w for i in range(len(ORDER))]
    ax.boxplot(data,positions=pos,widths=w*0.86,patch_artist=True,showfliers=False,
               medianprops=dict(color=SURF,lw=2),whiskerprops=dict(color=col,lw=1.4),
               capprops=dict(color=col,lw=1.4),boxprops=dict(facecolor=col,edgecolor=SURF,lw=2))
    for i,d in enumerate(data):
        x=np.random.RandomState(k*11+i).normal(pos[i],0.035,len(d))
        ax.scatter(x,d,s=11,color=col,alpha=.5,edgecolors=SURF,linewidths=.5,zorder=3)
    ax.plot([],[],'s',color=col,label=lab,ms=8)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
ax.set_ylabel("max probability at that decision layer"); ax.set_ylim(0.4,1.03)
ax.grid(axis="y",lw=.7); ax.set_axisbelow(True)
ax.set_title("Deciding WHETHER to fire an event is far more certain than deciding WHICH way to move",loc="left",color=INK,pad=10)
ax.legend(frameon=False,loc="lower left",fontsize=8.5)
fig.savefig(F/"fig6_first_vs_dir.png"); plt.close(fig)
print("figures:",sorted(p.name for p in F.glob("*.png")))
