#!/usr/bin/env python3
"""v6 data = v5 sets + hierarchical 'loc' items (area / room / object) from action_location traces."""
import json, os, random, shutil, sys, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines/laya_plan")
from loc_dataset import load
V5 = os.environ.get("LOC_BASE", "/home/andus/Projects/laya-test/v5_data")
V6 = os.environ.get("LOC_OUT", "/home/andus/Projects/laya-test/v6_data")
SINCE = float(os.environ.get("LOC_SINCE", "0"))
N_CALLS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
CONF = 0.85
os.makedirs(V6, exist_ok=True)
P = set(json.load(open("/home/andus/Projects/laya-test/v4_data/build_report.json"))["test_personas"])
rng = random.Random(61)

sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
from persona.prompt_template.laya_loc import build_level

def mk(r, level, area=None, room=None):
    bl = build_level(r["persona"], r["parsed"], level, area, room)
    if bl is None:
        return None
    state, qs, opts = bl
    label = {"area": r["ans"][0], "room": r["ans"][1], "obj": r["ans"][2]}[level]
    if label not in opts:
        return None
    k = len(opts)
    tgt = [(1 - CONF) / (k - 1)] * k
    tgt[opts.index(label)] = CONF
    return {"kind": "loc", "persona": r["persona"], "t": r["t"], "state": state, "questions": qs,
            "target": tgt, "label": opts.index(label), "n_votes": 1}

def level_items(r):
    sec, room, obj = r["ans"]
    its = [mk(r, "area"), mk(r, "room", sec), mk(r, "obj", sec, room)]
    return [i for i in its if i]

rows = [r for r in load() if r["valid"] and r["t"] >= SINCE]
rng.shuffle(rows)
# balanced over personas and activities
by_act = collections.Counter()
pick = []
for r in rows:
    a = (r["parsed"]["activity"] or "")[:40]
    if by_act[(r["persona"], a)] >= 2:
        continue
    by_act[(r["persona"], a)] += 1
    pick.append(r)
    if len(pick) >= N_CALLS:
        break
items = [it for r in pick for it in level_items(r)]
tr = [i for i in items if i["persona"] not in P]; te = [i for i in items if i["persona"] in P]
rng.shuffle(tr)
ncal = min(300, len(tr) // 10)
new = {"train": tr[ncal:], "calib": tr[:ncal], "test": te}
for name in ("train", "calib", "test"):
    with open(f"{V6}/{name}.jsonl", "w") as f:
        for l in open(f"{V5}/{name}.jsonl"):
            f.write(l)
        for it in new[name]:
            f.write(json.dumps(it) + "\n")
for name in ("offdiag_persona", "offdiag_time"):
    shutil.copy(f"{V5}/{name}.jsonl", f"{V6}/{name}.jsonl")
print("calls", len(pick), "items", len(items), {k: len(v) for k, v in new.items()},
      collections.Counter(i["state"]["level"] for i in items))
