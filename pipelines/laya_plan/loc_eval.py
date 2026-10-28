#!/usr/bin/env python3
"""Phase 2/3: held-out evaluation of laya 'loc' + gating thresholds.
Usage: loc_eval.py <port> [write_thresholds]
- top-1 agreement with the LLM label per level on persona-disjoint test items
- the LLM's own repeat agreement per level (the ceiling)
- margin thresholds: smallest margin where accepted accuracy >= the LLM's own repeat agreement (floor 0.80 area/room, 0.55 obj),
  and the coverage (share of decisions laya keeps) at that threshold."""
import collections, json, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines/laya_plan")
import persona.prompt_template.laya_scoring as L
L.LAYA_URL = f"http://127.0.0.1:{sys.argv[1]}"
from persona.prompt_template.laya_loc import ask_level
from loc_dataset import load

import os
items = [json.loads(l) for l in open(os.environ.get("LOC_TEST", "/home/andus/Projects/laya-test/v6_data/test.jsonl"))]
items = [i for i in items if i["kind"] == "loc"]
res = collections.defaultdict(list)       # level -> [(margin, correct)]
for it in items:
    q = it["questions"]; opts = list(q["loc"]["criteria"])
    pr = ask_level(it["state"], q, opts)
    if not pr:
        continue
    rk = sorted(pr.items(), key=lambda kv: -kv[1])
    res[it["state"]["level"]].append((rk[0][1] - (rk[1][1] if len(rk) > 1 else 0), rk[0][0] == opts[it["label"]]))
# LLM repeat agreement per level
g = collections.defaultdict(list)
for r in load():
    g[(r["persona"], r["prompt"])].append(r["ans"])
agree = {}
for lvl, idx in (("area", 0), ("room", 1), ("obj", 2)):
    num = den = 0
    for v in g.values():
        if len(v) < 2:
            continue
        if lvl != "area":
            v = [a for a in v if a[:idx] == v[0][:idx]]
        if len(v) < 2:
            continue
        top = collections.Counter(a[idx] for a in v).most_common(1)[0][1]
        num += top; den += len(v)
    agree[lvl] = num / max(1, den)
floors = {"area": 0.80, "room": 0.80, "obj": 0.55}
thr = {}
top1 = {lvl: sum(c for _, c in rows) / len(rows) for lvl, rows in res.items()}
print("RESULT", json.dumps(top1))
for lvl, rows in res.items():
    n = len(rows); acc = sum(c for _, c in rows) / n
    target = max(floors[lvl], min(agree[lvl], 0.95))
    best = None
    for t in [x / 100 for x in range(0, 100, 2)]:
        acc_ok = [c for m, c in rows if m >= t]
        if len(acc_ok) >= 0.1 * n and sum(acc_ok) / len(acc_ok) >= target:
            best = (t, len(acc_ok) / n, sum(acc_ok) / len(acc_ok)); break
    thr[lvl] = best[0] if best else 1.0
    print(f"{lvl:5s} n={n:5d} top1={acc:.3f}  LLM-repeat-agreement={agree[lvl]:.3f}  target={target:.2f}  "
          + (f"thr={best[0]:.2f} coverage={best[1]:.2f} acc@thr={best[2]:.3f}" if best else "no threshold reaches the target"))
if len(sys.argv) > 2:
    json.dump(thr, open("/home/andus/Projects/generative_agents/reverie/backend_server/persona/prompt_template/loc_thresholds.json", "w"))
    print("wrote thresholds", thr)
