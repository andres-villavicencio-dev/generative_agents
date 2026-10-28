#!/usr/bin/env python3
"""Build /home/andus/Projects/laya-test/v5_data: v4 sets + labelled 'choose' items.
Persona-disjoint split (same test personas as v4). Usage: laya_choose_build.py"""
import json, os, random, shutil, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
from persona.prompt_template.laya_choose import parse_prompt, vote_target

V4 = "/home/andus/Projects/laya-test/v4_data"
V5 = "/home/andus/Projects/laya-test/v5_data"
os.makedirs(V5, exist_ok=True)
P = set(json.load(open(f"{V4}/build_report.json"))["test_personas"])
rng = random.Random(20261028)
items, bad = [], 0
for l in open(f"{V5}/choose_raw.jsonl"):
    r = json.loads(l)
    pr = parse_prompt(r["prompt"], r["persona"])
    if not pr:
        bad += 1; continue
    state, qs, keys = pr
    tgt = vote_target(r["votes"], keys)
    if not tgt:
        bad += 1; continue
    items.append({"kind": "choose", "persona": r["persona"], "t": r["t"], "state": state,
                  "questions": qs, "target": tgt, "label": max(range(len(tgt)), key=tgt.__getitem__),
                  "n_votes": len(r["votes"])})
tr = [i for i in items if i["persona"] not in P]
te = [i for i in items if i["persona"] in P]
rng.shuffle(tr)
n_cal = min(200, len(tr) // 10)
new = {"train": tr[n_cal:], "calib": tr[:n_cal], "test": te}
for name in ("train", "calib", "test"):
    with open(f"{V5}/{name}.jsonl", "w") as f:
        for l in open(f"{V4}/{name}.jsonl"):
            f.write(l)
        for it in new[name]:
            f.write(json.dumps(it) + "\n")
for name in ("offdiag_persona", "offdiag_time"):
    shutil.copy(f"{V4}/{name}.jsonl", f"{V5}/{name}.jsonl")
import collections
print({k: len(v) for k, v in new.items()}, "bad", bad)
print("label dist (train):", collections.Counter(it["label"] for it in new["train"]),
      "own-share:", round(sum(1 for it in items if it["target"][-1] > 0.5) / max(1, len(items)), 3))
