#!/usr/bin/env python3
"""Held-out 'choose' eval sliced by agenda: due-event cards / not-due agenda cards / legacy (no agenda).
Usage: agenda_eval.py <port> [test.jsonl=laya-test/v11_data/test.jsonl]
Metrics per slice: n, top-1 agreement with the cloud-vote label, teacher-attend recall/precision
(attend = any numbered pull on a due card), mean P(plan)."""
import json, sys, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import persona.prompt_template.laya_scoring as L
L.LAYA_URL = f"http://127.0.0.1:{sys.argv[1]}"
TEST = sys.argv[2] if len(sys.argv) > 2 else "/home/andus/Projects/laya-test/v11_data/test.jsonl"
items = [json.loads(l) for l in open(TEST)]
items = [i for i in items if i["kind"] == "choose"]
S = collections.defaultdict(lambda: collections.Counter())
for it in items:
    sl = "legacy" if not it.get("agenda") else ("agenda_due" if it["meta"]["due"] else "agenda_notdue")
    q = it["questions"]["choose"]; keys = list(q["criteria"])
    try:
        a = (L._systemone(it["state"], {"choose": q}) or {}).get("choose") or {}
    except Exception:
        S[sl]["err"] += 1; continue
    leg = a.get("legend") or {}; pr = {}
    for k, v in (a.get("probabilities") or {}).items():
        if str(k).startswith("__"): continue
        lab = str(leg.get(k, k))
        for key, txt in q["criteria"].items():
            if lab in (key, txt): pr[key] = pr.get(key, 0) + float(v)
    t = sum(pr.values())
    if t <= 0 or "plan" not in pr: S[sl]["err"] += 1; continue
    pr = {k: v / t for k, v in pr.items()}
    top = max(pr, key=pr.get); lab = keys[it["label"]]
    c = S[sl]; c["n"] += 1; c["agree"] += top == lab; c["pplan"] += pr["plan"]
    is_att = lambda k: k.startswith("pull")
    if sl == "agenda_due":
        c["t_att"] += is_att(lab); c["l_att"] += is_att(top); c["tp"] += is_att(lab) and is_att(top)
for sl in ("agenda_due", "agenda_notdue", "legacy"):
    c = S[sl]; n = max(1, c["n"])
    line = f"{sl:14s} n={c['n']:4d} agree={c['agree']/n:.3f} mean P(plan)={c['pplan']/n:.3f} err={c['err']}"
    if sl == "agenda_due":
        line += f"  teacher_attend={c['t_att']} laya_attend={c['l_att']} recall={c['tp']/max(1,c['t_att']):.3f} precision={c['tp']/max(1,c['l_att']):.3f}"
    print(line)
