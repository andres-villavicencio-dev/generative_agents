#!/usr/bin/env python3
"""Held-out evaluation of laya on the 'choose' task. Usage: laya_choose_eval.py <port>"""
import json, sys, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import persona.prompt_template.laya_scoring as L
L.LAYA_URL = f"http://127.0.0.1:{sys.argv[1]}"
items = [json.loads(l) for l in open("/home/andus/Projects/laya-test/v5_data/test.jsonl")]
items = [i for i in items if i["kind"] == "choose"]
rows = []
for it in items:
    q = it["questions"]["choose"]; keys = list(q["criteria"])
    try:
        a = (L._systemone(it["state"], {"choose": q}) or {}).get("choose") or {}
    except Exception as e:
        continue
    leg = a.get("legend") or {}; pr = {}
    for k, v in (a.get("probabilities") or {}).items():
        if str(k).startswith("__"): continue
        lab = str(leg.get(k, k))
        for key, txt in q["criteria"].items():
            if lab in (key, txt): pr[key] = pr.get(key, 0) + float(v)
    t = sum(pr.values())
    if t <= 0 or "plan" not in pr: continue
    pr = {k: v / t for k, v in pr.items()}
    rows.append((it, keys, pr))
def top(pr): return max(pr, key=pr.get)
n = len(rows)
lab_dev = [it["label"] != 0 for it, _, _ in rows]
pdev = [1 - pr["plan"] for _, _, pr in rows]
pos = [p for p, d in zip(pdev, lab_dev) if d]; neg = [p for p, d in zip(pdev, lab_dev) if not d]
auc = sum((p > q) + .5 * (p == q) for p in pos for q in neg) / max(1, len(pos) * len(neg))
agree = sum(1 for (it, keys, pr) in rows if keys[it["label"]] == top(pr))
dec = sum(1 for (it, keys, pr) in rows if top(pr) != "plan")
print(f"n={n} teacher_deviate={sum(lab_dev)} ({100*sum(lab_dev)/n:.1f}%)  laya_deviate={dec} ({100*dec/n:.1f}%)")
print(f"top-1 agreement {agree}/{n} = {100*agree/n:.1f}%   AUC(P_dev vs teacher deviate) = {auc:.3f}")
tp = sum(1 for (it, keys, pr), d in zip(rows, lab_dev) if d and top(pr) != "plan")
print(f"deviation recall {tp}/{sum(lab_dev)}  precision {tp}/{dec}")
print("mean P(dev) when teacher complies %.3f, deviates %.3f" % (sum(neg)/len(neg), sum(pos)/max(1,len(pos))))
