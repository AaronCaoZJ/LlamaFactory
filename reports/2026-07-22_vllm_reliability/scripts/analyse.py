#!/usr/bin/env python3
"""Aggregate the per-sample probability records into summary.json + all report tables."""
from __future__ import annotations
import json, math, statistics as st
from pathlib import Path

S=Path("/tmp/claude-3014/-workspace1-zhijun/6a801ac6-4b1b-49b8-98d6-d62501f4249c/scratchpad")
ACTIONS=["MV_FWD","MV_BACK","MV_LEFT","MV_RIGHT","MV_UP","MV_DOWN","GRASP","RELEASE","DONE"]
MV=ACTIONS[:6]; EV=["GRASP","RELEASE","DONE"]
ORDER=["qwen0.8b","qwen2b","qwen2b-fkpp","qwen9b","qwen9b-fkpp","gemma-e4b","intern1b","intern2b"]

def load(n):
    L=[json.loads(l) for l in open(S/"probs"/f"{n}.jsonl")]
    return L[0]["_meta"], L[1:]
D={n:load(n) for n in ORDER}
def q(v,p):
    v=sorted(v)
    if not v: return float('nan')
    k=(len(v)-1)*p; f=math.floor(k); c=math.ceil(k)
    return v[f] if f==c else v[f]+(v[c]-v[f])*(k-f)
def m(v): return sum(v)/len(v) if v else float('nan')

OUT={"meta":{},"A":{},"B1":{},"B2":{},"B3":{},"C":{},"D":{},"E":{},"F":{},"variance":{}}
for n in ORDER:
    meta,rows=D[n]
    OUT["meta"][n]={k:meta[k] for k in ("port","lora","family","train","prompt","mv_prefix")}
    OUT["meta"][n]["token_pieces"]={a:meta["token_pieces"][a]["pieces"] for a in ACTIONS}

# ── A. overall ───────────────────────────────────────────────────────────────
print("\n=== A. 整体指标 ===")
hdr=f"{'model':<11}{'acc':>7}{'argmaxAcc':>11}{'top1':>9}{'margin':>9}{'entropy':>9}{'legal':>9}{'p_label':>9}{'tok0legal':>11}"
print(hdr); print("-"*len(hdr))
for n in ORDER:
    meta,rows=D[n]
    acc=m([r["correct"] for r in rows]); aacc=m([r["argmax_correct"] for r in rows])
    t1=m([r["top1_prob"] for r in rows]); mg=m([r["margin"] for r in rows])
    en=m([r["entropy"] for r in rows]); lm=m([r["legal_mass"] for r in rows])
    pl=m([r["p_label"] for r in rows]); t0=m([r["first_token_legal_mass"] for r in rows])
    OUT["A"][n]=dict(n=len(rows),acc=acc,argmax_acc=aacc,mean_top1=t1,median_top1=q([r["top1_prob"] for r in rows],.5),
                     mean_margin=mg,mean_entropy=en,mean_legal_mass=lm,min_legal_mass=min(r["legal_mass"] for r in rows),
                     mean_p_label=pl,mean_first_token_legal=t0,
                     mean_top1_correct=m([r["top1_prob"] for r in rows if r["correct"]]),
                     mean_top1_wrong=m([r["top1_prob"] for r in rows if not r["correct"]]))
    print(f"{n:<11}{acc:>7.3f}{aacc:>11.3f}{t1:>9.4f}{mg:>9.4f}{en:>9.4f}{lm:>9.4f}{pl:>9.4f}{t0:>11.5f}")

# ── B1. event mass on the 47 MV samples ──────────────────────────────────────
print("\n=== B1. 47 条 MV_* 样本上分给事件 token 的概率 (first-token exact) ===")
hdr=f"{'model':<11}{'P(GR)mean':>11}{'med':>9}{'p95':>9}{'max':>9}{'>1%':>6}{'P(REL)mean':>12}{'max':>9}{'P(DONE)mean':>13}{'max':>9}{'P(anyEV)max':>13}"
print(hdr); print("-"*len(hdr))
for n in ORDER:
    meta,rows=D[n]
    mv=[r for r in rows if r["label"] in MV]
    g=[r["first_token_probs"]["GRASP"] for r in mv]
    rl=[r["first_token_probs"]["RELEASE"] for r in mv]
    dn=[r["first_token_probs"]["DONE"] for r in mv]
    an=[a+b+c for a,b,c in zip(g,rl,dn)]
    OUT["B1"][n]=dict(n=len(mv),
      GRASP=dict(mean=m(g),median=q(g,.5),p95=q(g,.95),max=max(g),n_gt_01=sum(x>0.01 for x in g),n_gt_1=sum(x>0.1 for x in g)),
      RELEASE=dict(mean=m(rl),median=q(rl,.5),p95=q(rl,.95),max=max(rl),n_gt_01=sum(x>0.01 for x in rl),n_gt_1=sum(x>0.1 for x in rl)),
      DONE=dict(mean=m(dn),median=q(dn,.5),p95=q(dn,.95),max=max(dn),n_gt_01=sum(x>0.01 for x in dn),n_gt_1=sum(x>0.1 for x in dn)),
      ANY=dict(mean=m(an),median=q(an,.5),p95=q(an,.95),max=max(an),n_gt_01=sum(x>0.01 for x in an),n_gt_1=sum(x>0.1 for x in an)),
      n_event_argmax=sum(1 for r in mv if r["argmax_action"] in EV),
      event_argmax_idx=[r["i"] for r in mv if r["argmax_action"] in EV])
    print(f"{n:<11}{m(g):>11.2e}{q(g,.5):>9.1e}{q(g,.95):>9.2e}{max(g):>9.3f}{sum(x>0.01 for x in g):>6}"
          f"{m(rl):>12.2e}{max(rl):>9.3f}{m(dn):>13.2e}{max(dn):>9.3f}{max(an):>13.3f}")

# ── B2. the three event samples ──────────────────────────────────────────────
print("\n=== B2. GRASP(24) / RELEASE(43) / DONE(49) 逐条 9 类概率 ===")
for idx,lab in [(24,"GRASP"),(43,"RELEASE"),(49,"DONE")]:
    print(f"\n-- sample {idx}  label={lab}")
    print(f"{'model':<11}{'pred':<10}"+"".join(f"{a:>11}" for a in ACTIONS)+f"{'legal':>9}")
    for n in ORDER:
        meta,rows=D[n]; r=rows[idx]
        ap=r["action_probs"]
        cells="".join((f"{ap[a]:>11.3e}" if ap[a] is not None else f"{'n/a':>11}") for a in ACTIONS)
        print(f"{n:<11}{r['pred_token']:<10}{cells}{r['legal_mass']:>9.4f}")
        OUT["B2"].setdefault(str(idx),{})[n]=dict(label=lab,pred=r["pred_token"],correct=r["correct"],
            action_probs=ap,first_token_probs=r["first_token_probs"],dir_probs=r["dir_probs"],
            dir_available=r["dir_available"],flags=r["prob_flags"],legal_mass=r["legal_mass"],
            top1=r["top1_prob"],margin=r["margin"],entropy=r["entropy"])

# ── B3. first-token (event vs move decision) vs direction confidence ─────────
print("\n=== B3. 首token四分类置信度 vs 方向token置信度 (全部/仅MV样本) ===")
hdr=(f"{'model':<11}{'tok0_top1':>11}{'tok0_med':>10}{'tok0_ent':>10}{'tok0_min':>10}"
     f"{'dir_top1':>10}{'dir_med':>9}{'dir_ent':>9}{'dir_min':>9}{'dir<0.9':>9}{'tok0<0.9':>10}")
print(hdr); print("-"*len(hdr))
for n in ORDER:
    meta,rows=D[n]
    t0=[max(r["first_token_probs"].values()) for r in rows]
    t0e=[]
    for r in rows:
        v=[x for x in r["first_token_probs"].values()]; s=sum(v) or 1e-12
        t0e.append(-sum((x/s)*math.log(x/s) for x in v if x>0))
    mv=[r for r in rows if r["dir_available"]]
    dv=[max(x for x in r["dir_probs"].values() if x is not None) for r in mv]
    dve=[]
    for r in mv:
        v=[x for x in r["dir_probs"].values() if x is not None]; s=sum(v) or 1e-12
        dve.append(-sum((x/s)*math.log(x/s) for x in v if x>0))
    # restrict to MV-labelled samples for a like-for-like comparison
    mvlab=[r for r in rows if r["label"] in MV and r["dir_available"]]
    t0m=[max(r["first_token_probs"].values()) for r in mvlab]
    dvm=[max(x for x in r["dir_probs"].values() if x is not None) for r in mvlab]
    OUT["B3"][n]=dict(
      tok0=dict(n=len(t0),mean=m(t0),median=q(t0,.5),p05=q(t0,.05),min=min(t0),mean_entropy=m(t0e),
                n_below_09=sum(x<0.9 for x in t0),n_below_099=sum(x<0.99 for x in t0)),
      dirs=dict(n=len(dv),mean=m(dv),median=q(dv,.5),p05=q(dv,.05),min=min(dv),mean_entropy=m(dve),
                n_below_09=sum(x<0.9 for x in dv),n_below_099=sum(x<0.99 for x in dv)),
      on_mv_labels=dict(n=len(mvlab),tok0_mean=m(t0m),dir_mean=m(dvm),
                        tok0_min=min(t0m) if t0m else None,dir_min=min(dvm) if dvm else None,
                        tok0_below_09=sum(x<0.9 for x in t0m),dir_below_09=sum(x<0.9 for x in dvm)))
    print(f"{n:<11}{m(t0):>11.4f}{q(t0,.5):>10.4f}{m(t0e):>10.4f}{min(t0):>10.4f}"
          f"{m(dv):>10.4f}{q(dv,.5):>9.4f}{m(dve):>9.4f}{min(dv):>9.4f}{sum(x<0.9 for x in dv):>9}{sum(x<0.9 for x in t0):>10}")

# ── C. 6x6 direction confusion (mean prob mass) ─────────────────────────────
print("\n=== C. 方向混淆 (行=真值, 列=平均 P(dir|MV)) ===")
for n in ORDER:
    meta,rows=D[n]
    conf={}
    for lab in MV:
        sub=[r for r in rows if r["label"]==lab and r["dir_available"]]
        if not sub: conf[lab]={"n":0}; continue
        row={a:m([(r["dir_probs"][a] or 0.0) for r in sub]) for a in MV}
        row["n"]=len(sub); conf[lab]=row
    OUT["C"][n]=conf
    print(f"\n-- {n}")
    print(f"{'true\\pred':<11}{'n':>3}"+"".join(f"{a.replace('MV_',''):>9}" for a in MV))
    for lab in MV:
        c=conf[lab]
        if c["n"]==0: print(f"{lab:<11}{0:>3}"+"".join(f"{'-':>9}" for _ in MV)); continue
        print(f"{lab:<11}{c['n']:>3}"+"".join(f"{c[a]:>9.4f}" for a in MV))
# opposite-axis hesitation
print("\n=== C2. 对立轴概率泄漏 (真值方向 -> 其对立方向的平均概率) ===")
OPP={"MV_FWD":"MV_BACK","MV_BACK":"MV_FWD","MV_LEFT":"MV_RIGHT","MV_RIGHT":"MV_LEFT","MV_UP":"MV_DOWN","MV_DOWN":"MV_UP"}
print(f"{'model':<11}{'FWD->BACK':>11}{'LEFT->RIGHT':>13}{'RIGHT->LEFT':>13}{'UP->DOWN':>11}{'DOWN->UP':>11}")
for n in ORDER:
    conf=OUT["C"][n]; cells=[]
    o={}
    for lab in ["MV_FWD","MV_LEFT","MV_RIGHT","MV_UP","MV_DOWN"]:
        v=conf[lab].get(OPP[lab]) if conf[lab]["n"] else None
        o[f"{lab}->{OPP[lab]}"]=v; cells.append(f"{v:>11.4f}" if v is not None else f"{'-':>11}")
    OUT["C"].setdefault("_opposite",{})[n]=o
    print(f"{n:<11}"+cells[0]+f"{o['MV_LEFT->MV_RIGHT']:>13.4f}{o['MV_RIGHT->MV_LEFT']:>13.4f}"+cells[3]+cells[4])

# ── D. trajectory ────────────────────────────────────────────────────────────
for n in ORDER:
    meta,rows=D[n]
    OUT["D"][n]=[dict(i=r["i"],label=r["label"],pred=r["pred_token"],correct=r["correct"],
                      top1=r["top1_prob"],p_label=r["p_label"],entropy=r["entropy"],
                      p_event=sum(r["first_token_probs"][e] for e in EV)) for r in rows]

# ── E. calibration ───────────────────────────────────────────────────────────
BINS=[(0,0.5),(0.5,0.7),(0.7,0.9),(0.9,0.99),(0.99,1.0001)]
print("\n=== E. 校准 (按 top1 概率分桶) ===")
print(f"{'model':<11}"+"".join(f"{f'[{a},{b})':>16}" for a,b in BINS))
for n in ORDER:
    meta,rows=D[n]; cells=[]; rec={}
    for a,b in BINS:
        sub=[r for r in rows if a<=r["top1_prob"]<b]
        if sub:
            acc=m([r["correct"] for r in sub]); conf=m([r["top1_prob"] for r in sub])
            rec[f"{a}-{b}"]=dict(n=len(sub),acc=acc,mean_conf=conf)
            cells.append(f"{acc:.2f}/{len(sub)}({conf:.2f})".rjust(16))
        else:
            rec[f"{a}-{b}"]=dict(n=0,acc=None,mean_conf=None); cells.append("-".rjust(16))
    ece=sum(v["n"]/len(rows)*abs(v["acc"]-v["mean_conf"]) for v in rec.values() if v["n"])
    rec["ECE"]=ece; OUT["E"][n]=rec
    print(f"{n:<11}"+"".join(cells)+f"   ECE={ece:.3f}")

# ── F. cross-model ───────────────────────────────────────────────────────────
OUT["F"]=dict(scaling_qwen={n:OUT["A"][n] for n in ["qwen0.8b","qwen2b","qwen9b"]},
              internvl={n:OUT["A"][n] for n in ["intern1b","intern2b"]},
              same_scale_2b={n:OUT["A"][n] for n in ["qwen2b","intern2b"]})

# ── variance probe ───────────────────────────────────────────────────────────
print("\n=== 运行间抖动 (同一请求重复 3 次, top1 方向概率极差) ===")
print(f"{'model':<11}{'sample':<8}{'label':<10}{'max_spread_dir':>16}{'texts'}")
for n in ORDER:
    meta,rows=D[n]
    L=[json.loads(l) for l in open(S/"probs"/f"{n}.jsonl")]
    var=L[0].get("_variance",[])
    vr={}
    for v in var:
        spreads=[]
        keys=set()
        for t in v["trials"]: keys|= {k for k,x in (t["dir"] or {}).items() if x is not None}
        for k in keys:
            vals=[t["dir"].get(k) for t in v["trials"] if t["dir"] and t["dir"].get(k) is not None]
            if len(vals)==len(v["trials"]): spreads.append(max(vals)-min(vals))
        sp=max(spreads) if spreads else 0.0
        txts=[t["text"] for t in v["trials"]]
        vr[v["i"]]=dict(label=v["label"],max_dir_spread=sp,texts=txts,unstable=len(set(txts))>1)
        print(f"{n:<11}{v['i']:<8}{v['label']:<10}{sp:>16.4f}  {txts}")
    OUT["variance"][n]=vr

json.dump(OUT, open(S/"probs"/"summary.json","w"), indent=1, ensure_ascii=False)
print(f"\n-> {S/'probs'/'summary.json'}")

# ── B1b. stratify MV samples by distance to the nearest true event step ──────
print("\n=== B1b. MV 样本按「到最近事件步的距离」分层的事件概率 ===")
EVSTEP=[24,43,49]
def dist(i): return min(abs(i-e) for e in EVSTEP)
print(f"{'model':<11}{'d=1 n':>7}{'meanP(EV)':>11}{'maxP(EV)':>10}{'d>=2 n':>8}{'meanP(EV)':>11}{'maxP(EV)':>10}{'d>=2 P>0.01':>13}")
for n in ORDER:
    meta,rows=D[n]
    near=[r for r in rows if r["label"] in MV and dist(r["i"])==1]
    far =[r for r in rows if r["label"] in MV and dist(r["i"])>=2]
    pe=lambda r: sum(r["first_token_probs"][e] for e in EV)
    a=[pe(r) for r in near]; b=[pe(r) for r in far]
    OUT["B1"].setdefault("_stratified",{})[n]=dict(
        near=dict(n=len(a),mean=m(a),max=max(a),ids=[r["i"] for r in near]),
        far =dict(n=len(b),mean=m(b),max=max(b),n_gt_01=sum(x>0.01 for x in b),
                  n_gt_001=sum(x>0.001 for x in b)))
    print(f"{n:<11}{len(a):>7}{m(a):>11.4f}{max(a):>10.4f}{len(b):>8}{m(b):>11.2e}{max(b):>10.2e}{sum(x>0.01 for x in b):>13}")

# ── per-group accuracy ───────────────────────────────────────────────────────
print("\n=== A2. 分组准确率 & 置信度 (MV组 vs 事件组) ===")
print(f"{'model':<11}{'MV acc':>8}{'MV top1':>9}{'EV acc':>8}{'EV top1':>9}{'corr top1':>11}{'wrong top1':>12}")
for n in ORDER:
    meta,rows=D[n]
    mv=[r for r in rows if r["label"] in MV]; ev=[r for r in rows if r["label"] in EV]
    OUT["A"][n]["group"]=dict(mv=dict(n=len(mv),acc=m([r["correct"] for r in mv]),top1=m([r["top1_prob"] for r in mv])),
                              ev=dict(n=len(ev),acc=m([r["correct"] for r in ev]),top1=m([r["top1_prob"] for r in ev]),
                                      detail=[(r["i"],r["label"],r["pred_token"],r["top1_prob"]) for r in ev]))
    print(f"{n:<11}{m([r['correct'] for r in mv]):>8.3f}{m([r['top1_prob'] for r in mv]):>9.4f}"
          f"{m([r['correct'] for r in ev]):>8.3f}{m([r['top1_prob'] for r in ev]):>9.4f}"
          f"{OUT['A'][n]['mean_top1_correct']:>11.4f}{OUT['A'][n]['mean_top1_wrong']:>12.4f}")
json.dump(OUT, open(S/"probs"/"summary.json","w"), indent=1, ensure_ascii=False)
