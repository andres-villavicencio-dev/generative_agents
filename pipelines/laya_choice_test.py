#!/usr/bin/env python3
"""Can laya (typed 'choice' classifier) decide comply / pull / own for the
free-choice prompt? Replays saved prompts, asks laya for a distribution and
compares with a teacher (glm-5.3-flash:cloud) and the recorded 2B answer.
Usage: venv/bin/python pipelines/laya_choice_test.py [N]"""
import json, re, sys, time, statistics as S
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines")
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
from choice_model_replay import load, parse, ask
from persona.prompt_template.laya_scoring import _systemone


def laya(d):
    p = d["prompt"]
    head, _, rest = p.partition("Options for the next while:\n")
    opts_txt = rest.split("\nX. ")[0]
    lines = [l for l in opts_txt.split("\n") if l.strip()]
    crit = {}
    for l in lines:
        k, _, txt = l.partition(". ")
        crit["plan" if k == "0" else f"pull{k}"] = txt.strip()
    crit["own"] = "Do something else of their own choosing instead of the plan"
    ctx = head.strip()[-1500:]
    t0 = time.time()
    ans = _systemone({"context": ctx}, {"c": {
        "type": "choice",
        "instructions": ("Which does this person do for the next while: follow "
                         "their plan, take one of the offered pulls, or do "
                         "something else of their own?"),
        "criteria": crit}})
    a = (ans or {}).get("c") or {}
    legend = a.get("legend") or {}
    probs = {}
    for k, v in (a.get("probabilities") or {}).items():
        if str(k).startswith("__"):
            continue
        lab = str(legend.get(k, k))
        for key, txt in crit.items():
            if lab == key or lab == txt:
                probs[key] = probs.get(key, 0) + float(v)
    tot = sum(probs.values()) or 1
    return {k: v / tot for k, v in probs.items()}, time.time() - t0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    data = load(n)
    rows = []
    for d in data:
        try:
            pr, lat = laya(d)
        except Exception as e:
            pr, lat = {}, 0
            print("laya err", type(e).__name__, e)
        try:
            txt, _ = ask("glm-5.3-flash:cloud", d)
            g = parse(txt)
        except Exception:
            g = None
        r2 = parse(d["content"])
        rows.append({"persona": d["persona"], "laya": pr, "lat": lat,
                     "glm": g[0] if g else None, "b2": r2[0] if r2 else None})
    ok = [r for r in rows if r["laya"]]
    def top(r): return max(r["laya"], key=r["laya"].get)
    def dev(r): return 1 - r["laya"].get("plan", 0)
    print("n", len(rows), "laya answered", len(ok))
    print("laya top: ", {k: sum(1 for r in ok if top(r) == k) for k in set(map(top, ok))})
    print("mean P(deviate):", round(S.mean(dev(r) for r in ok), 3),
          " p50 lat", round(sorted(r["lat"] for r in ok)[len(ok)//2], 2))
    for t in ("glm", "b2"):
        L = [r for r in ok if r[t]]
        agree = sum(1 for r in L if (top(r) == "plan") == (r[t] == "plan"))
        pos = [dev(r) for r in L if r[t] != "plan"]
        neg = [dev(r) for r in L if r[t] == "plan"]
        auc = (sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
               if pos and neg else None)
        print(f"vs {t}: n={len(L)} comply-agree={agree}/{len(L)} "
              f"teacher_deviate={len(pos)} AUC(P_dev)={auc and round(auc,2)}")
    json.dump(rows, open("/home/andus/Projects/generative_agents/laya_choice_test.json", "w"), indent=1)
