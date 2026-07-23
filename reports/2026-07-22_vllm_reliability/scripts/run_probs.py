#!/usr/bin/env python3
"""MVTOKEN 6-model confidence / probability-distribution probe.

P(action) is decomposed along the tokenizer's prefix tree, using ONE greedy request per sample
with top_logprobs=20 (per-step top-20 of the generated path):

    P(a) = P(head(a)) * P(direction | MV-prefix) * P(deterministic tail)

* step-0 head distribution  -> EXACT 4-way class split MV / GR / RELEASE / DONE
* direction step            -> EXACT P(dir | MV...) whenever the greedy path went through MV
* deterministic tails       -> GR->ASP, FW->D ; measured where on-path, else assumed 1.0 (flagged)

Why not force the prefix: the servers refuse per-request chat templates
(--trust-request-chat-template unset) and vLLM's continue_final_message path renders the Qwen
template's final-assistant branch, injecting '<think>\n\n</think>\n\n' (off training distribution);
logit_bias is ignored by these servers. See reportC for details.

Request construction follows scripts/qwen3_5/eval/infer.py (media BEFORE text, temperature 0).
"""
from __future__ import annotations
import base64, json, math, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

SCRATCH = Path("/tmp/claude-3014/-workspace1-zhijun/6a801ac6-4b1b-49b8-98d6-d62501f4249c/scratchpad")
EVALSET = Path("/workspace1/zhijun/LlamaFactory/data/agentrobot/ood_sample/v3/rollout_lite.json")
ACTIONS = ["MV_FWD","MV_BACK","MV_LEFT","MV_RIGHT","MV_UP","MV_DOWN","GRASP","RELEASE","DONE"]
MV_ACTIONS = ACTIONS[:6]
EVENTS = ["GRASP","RELEASE","DONE"]

MODELS = {
 "qwen0.8b":  dict(port=8108, lora="mix_22-06_fk-pp_02_08",   family="qwen",     train="mix_22-06_fk-pp_02_exchange_token", prompt="v3"),
 "qwen2b":    dict(port=8102, lora="mix_22_27_v3_2",          family="qwen",     train="mix_22_27_v3_lite",                 prompt="v3"),
 "qwen2b-fkpp": dict(port=8102, lora="mix_22-06_fk-pp_02_2", family="qwen", train="mix_22-06_fk-pp_02_exchange_token", prompt="v3"),
 "qwen9b":    dict(port=8109, lora="mix_22_27_v3_9",          family="qwen",     train="mix_22_27_v3_lite",                 prompt="v3"),
 "qwen9b-fkpp": dict(port=8109, lora="mix_22-06_fk-pp_02", family="qwen", train="mix_22-06_fk-pp_02_exchange_token", prompt="v3"),
 "gemma-e4b": dict(port=8104, lora="gemma4_e4b_mix_22_27_v3", family="gemma",    train="mix_22_27_v3_lite",                 prompt="v3"),
 "intern1b":  dict(port=8201, lora="internvl3.5-1b",          family="internvl", train="mix_22-06_fk-pp_02_exchange_token", prompt="v3"),
 "intern2b":  dict(port=8202, lora="internvl3.5-2b",          family="internvl", train="mix_22-06_fk-pp_02_exchange_token", prompt="v3"),
}
_img_cache = {}; _lock = threading.Lock()

def post(url, obj, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type":"application/json"}, method="POST")
    last=None
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f: return json.loads(f.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from e
        except Exception as e:
            last=e; time.sleep(1.5*(a+1))
    raise RuntimeError(f"failed after retries: {last!r}")

def encode_image(path):
    with _lock:
        if path in _img_cache: return _img_cache[path]
    sfx=Path(path).suffix.lstrip(".").lower()
    mime={"jpg":"jpeg","jpeg":"jpeg","png":"png"}.get(sfx,"jpeg")
    uri=f"data:image/{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
    with _lock: _img_cache[path]=uri
    return uri

def user_content(sample):
    text=sample["instruction"].replace("<image>","").replace("<video>","").strip()
    extra=(sample.get("input") or "").strip()
    if extra: text=text+"\n"+extra
    c=[{"type":"image_url","image_url":{"url":encode_image(p)}} for p in sample["images"]]
    c.append({"type":"text","text":text})
    return c

def token_pieces(base, model):
    out={}
    for a in ACTIONS:
        ids=post(f"{base}/tokenize",{"model":model,"prompt":a,"add_special_tokens":False})["tokens"]
        pcs=[post(f"{base}/detokenize",{"model":model,"tokens":[i]})["prompt"] for i in ids]
        assert "".join(pcs)==a, f"{a} -> {pcs}"
        out[a]=(ids,pcs)
    return out

def generate(base, model, content, max_tokens=8):
    r=post(f"{base}/v1/chat/completions",
           {"model":model,"messages":[{"role":"user","content":content}],"temperature":0.0,
            "logprobs":True,"top_logprobs":20,"max_tokens":max_tokens})
    steps=[{"chosen":s["token"],"p":math.exp(s["logprob"]),
            "top20":[[t["token"],math.exp(t["logprob"])] for t in s["top_logprobs"]]}
           for s in r["choices"][0]["logprobs"]["content"]]
    return r["choices"][0]["message"]["content"], steps

def common_prefix(lists):
    out=[]
    for col in zip(*lists):
        if len(set(col))==1: out.append(col[0])
        else: break
    return out

def analyse(pieces, steps):
    """Return the decomposed probability record for one sample."""
    pcs={a:pieces[a][1] for a in ACTIONS}
    mvpref=common_prefix([pcs[a] for a in MV_ACTIONS])          # ['MV'] qwen/internvl, ['MV','_'] gemma
    k=len(mvpref)
    dist=lambda i: {t:p for t,p in steps[i]["top20"]} if i<len(steps) else {}
    chosen=lambda i: steps[i]["chosen"] if i<len(steps) else None

    d0=dist(0)
    head={"MV":mvpref[0], "GRASP":pcs["GRASP"][0], "RELEASE":pcs["RELEASE"][0], "DONE":pcs["DONE"][0]}
    first={c:d0.get(h,0.0) for c,h in head.items()}
    first_minp=min(d0.values()) if d0 else 0.0
    first_missing=[c for c,h in head.items() if h not in d0]

    # walk the shared MV prefix along the greedy path
    on_mv_path = all(chosen(i)==mvpref[i] for i in range(k))
    p_mvpref = 1.0
    for i in range(k):
        p_mvpref *= dist(i).get(mvpref[i], 0.0)
    dir_probs, dir_minp, dir_avail = {}, None, False
    tails = {}
    if on_mv_path and k < len(steps):
        dd=dist(k); dir_minp=min(dd.values()) if dd else 0.0; dir_avail=True
        for a in MV_ACTIONS:
            dir_probs[a]=dd.get(pcs[a][k])          # None if outside top-20
        # deterministic tail beyond the direction token (gemma: 'FW'->'D')
        for a in MV_ACTIONS:
            tail=1.0; exact=True
            for j in range(k+1, len(pcs[a])):
                if chosen(j-1)==pcs[a][j-1] and j<len(steps):
                    tail*=dist(j).get(pcs[a][j], 0.0)
                else:
                    exact=False                     # assume deterministic continuation
            tails[a]=(tail, exact)
    # GR -> ASP tail (measured only when the greedy path took GR)
    grasp_tail, grasp_tail_exact = 1.0, False
    if len(pcs["GRASP"])>1 and chosen(0)==pcs["GRASP"][0] and 1<len(steps):
        grasp_tail=dist(1).get(pcs["GRASP"][1], 0.0); grasp_tail_exact=True
    return dict(mv_prefix=mvpref, dir_step=k, first=first, first_minp=first_minp,
                first_missing=first_missing, on_mv_path=on_mv_path, dir_avail=dir_avail,
                dir_probs=dir_probs, dir_minp=dir_minp, tails=tails,
                grasp_tail=grasp_tail, grasp_tail_exact=grasp_tail_exact)

def run_model(name, cfg, samples, limit=None, repeats=1):
    base=f"http://localhost:{cfg['port']}"; model=cfg["lora"]
    pieces=token_pieces(base, model)
    outp=SCRATCH/"probs"/f"{name}.jsonl"; outp.parent.mkdir(parents=True, exist_ok=True)
    meta={"model":name, **cfg,
          "token_pieces":{a:{"ids":pieces[a][0],"pieces":pieces[a][1]} for a in ACTIONS},
          "mv_prefix":common_prefix([pieces[a][1] for a in MV_ACTIONS])}
    rows=[]; t0=time.time()
    for idx,s in enumerate(samples[:limit or len(samples)]):
        content=user_content(s)
        text, steps = generate(base, model, content)
        A=analyse(pieces, steps)
        f=A["first"]
        # full-string action probabilities
        ap, flags = {}, {}
        for a in MV_ACTIONS:
            if A["dir_avail"] and A["dir_probs"].get(a) is not None:
                tail,texact=A["tails"].get(a,(1.0,True))
                ap[a]=f["MV"]*A["dir_probs"][a]*tail
                flags[a]="exact" if texact else "tail_assumed"
            elif A["dir_avail"]:
                ap[a]=f["MV"]*(A["dir_minp"] or 0.0)         # below top-20 -> upper bound
                flags[a]="dir_below_top20_ub"
            else:
                ap[a]=None; flags[a]="dir_unavailable"       # greedy left the MV branch at step 0
        ap["GRASP"]=f["GRASP"]*(A["grasp_tail"] if A["grasp_tail_exact"] else 1.0)
        flags["GRASP"]="exact" if A["grasp_tail_exact"] else "asp_tail_assumed_1.0"
        ap["RELEASE"]=f["RELEASE"]; flags["RELEASE"]="exact"
        ap["DONE"]=f["DONE"];       flags["DONE"]="exact"

        eff={a:(0.0 if ap[a] is None else ap[a]) for a in ACTIONS}
        known=[a for a in ACTIONS if ap[a] is not None]
        legal_mass = (sum(eff[a] for a in EVENTS) + (f["MV"] if not A["dir_avail"]
                      else sum(eff[a] for a in MV_ACTIONS)))
        srt=sorted(((a,eff[a]) for a in known), key=lambda kv:-kv[1])
        tot=sum(eff[a] for a in known) or 1e-12
        ent=-sum((eff[a]/tot)*math.log(eff[a]/tot) for a in known if eff[a]>0)
        pred=text.strip()
        rows.append({"i":idx,"label":s["output"],"pred_token":pred,"correct":pred==s["output"],
            "argmax_action":srt[0][0],"argmax_correct":srt[0][0]==s["output"],
            "action_probs":ap,"prob_flags":flags,"legal_mass":legal_mass,
            "first_token_probs":f,"first_token_legal_mass":sum(f.values()),
            "first_token_missing":A["first_missing"],"first_token_minp":A["first_minp"],
            "dir_available":A["dir_avail"],"dir_probs":A["dir_probs"],"dir_minp":A["dir_minp"],
            "dir_legal_mass":(sum(v for v in A["dir_probs"].values() if v is not None) if A["dir_avail"] else None),
            "grasp_tail":A["grasp_tail"],"grasp_tail_exact":A["grasp_tail_exact"],
            "top1_prob":srt[0][1],"top2_prob":srt[1][1] if len(srt)>1 else 0.0,
            "margin":srt[0][1]-(srt[1][1] if len(srt)>1 else 0.0),
            "entropy":ent,"p_label":eff.get(s["output"],0.0),
            "gen_text":text,"gen_steps":steps})
        print(f"[{name}] {idx:>2} lab={s['output']:<8} pred={pred:<10} top1={srt[0][0]}:{srt[0][1]:.4f} legal={legal_mass:.4f}", flush=True)

    # run-to-run variance probe: repeat a fixed subset
    var=[]
    if repeats>1:
        for idx in [0,10,24,30,43,49]:
            if idx>=len(rows): continue
            content=user_content(samples[idx]); tr=[]
            for _ in range(repeats):
                _t,st=generate(base,model,content)
                a2=analyse(pieces,st)
                tr.append({"first":a2["first"],"dir":a2["dir_probs"],"text":_t.strip()})
            var.append({"i":idx,"label":samples[idx]["output"],"trials":tr})
    with open(outp,"w") as fo:
        fo.write(json.dumps({"_meta":meta,"_variance":var})+"\n")
        for r in rows: fo.write(json.dumps(r)+"\n")
    print(f"[{name}] DONE {len(rows)} rows in {time.time()-t0:.0f}s -> {outp}", flush=True)

def main():
    samples=json.loads(EVALSET.read_text())
    limit=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1]!="0" else None
    only=sys.argv[2].split(",") if len(sys.argv)>2 else list(MODELS)
    repeats=int(sys.argv[3]) if len(sys.argv)>3 else 1
    threads=[]; errs={}
    def wrap(n,c):
        try: run_model(n,c,samples,limit,repeats)
        except Exception as e:
            import traceback; errs[n]=traceback.format_exc(); print(f"[{n}] FAILED: {e}", flush=True)
    for n in only:
        t=threading.Thread(target=wrap,args=(n,MODELS[n])); t.start(); threads.append(t); time.sleep(0.3)
    for t in threads: t.join()
    for n,e in errs.items(): print(f"===== {n} traceback =====\n{e}")

if __name__=="__main__": main()
