#!/usr/bin/env python3
"""Build laya-test/v11_data = v10_data + agenda 'choose' cards labelled by cloud teachers only.
Persona-disjoint split (same test personas as v4). A card needs >=2 valid votes of 3 teachers.
Usage: agenda_build.py [base_dir=v10_data] [out_dir=v11_data]"""
import collections, glob, json, os, random, shutil, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
from persona.prompt_template.laya_choose import parse_prompt, vote_target
LT = "/home/andus/Projects/laya-test"
BASE = sys.argv[1] if len(sys.argv) > 1 else f"{LT}/v10_data"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{LT}/v11_data"
D = "/home/andus/Projects/generative_agents/ga_traces/agenda_choice"
P = set(json.load(open(f"{LT}/v4_data/build_report.json"))["test_personas"])
rng = random.Random(20261111)
cards = {}
for l in open(f"{D}/cards.jsonl"):
    r = json.loads(l); cards[r["key"]] = r
votes = collections.defaultdict(list)
for f in glob.glob(f"{D}/*.jsonl"):
    if f.endswith("cards.jsonl"): continue
    for l in open(f):
        r = json.loads(l)
        if r.get("vote") is not None: votes[r["key"]].append(r["vote"])
items, bad, agree = [], 0, []
for k, vs in votes.items():
    c = cards.get(k)
    if not c or len(vs) < 2: bad += 1; continue
    pr = parse_prompt(c["prompt"], c["persona"])
    if not pr: bad += 1; continue
    state, qs, keys = pr
    vs = [v for v in vs if v == "own" or v == 0 or (isinstance(v, int) and f"pull{v}" in keys)]
    tgt = vote_target(vs, keys)
    if not tgt: bad += 1; continue
    agree.append(collections.Counter(map(str, vs)).most_common(1)[0][1] / len(vs))
    items.append({"kind": "choose", "persona": c["persona"], "t": c["t"], "state": state, "questions": qs,
                  "target": tgt, "label": max(range(len(tgt)), key=tgt.__getitem__), "n_votes": len(vs),
                  "agenda": True, "meta": c["meta"]})
tr = [i for i in items if i["persona"] not in P]; te = [i for i in items if i["persona"] in P]
rng.shuffle(tr); n_cal = min(100, len(tr) // 10)
new = {"train": tr[n_cal:], "calib": tr[:n_cal], "test": te}
os.makedirs(OUT, exist_ok=True)
for name in ("train", "calib", "test"):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for l in open(f"{BASE}/{name}.jsonl"): f.write(l)
        for it in new[name]: f.write(json.dumps(it) + "\n")
for extra in os.listdir(BASE):
    if extra not in ("train.jsonl", "calib.jsonl", "test.jsonl") and os.path.isfile(f"{BASE}/{extra}"):
        shutil.copy(f"{BASE}/{extra}", f"{OUT}/{extra}")
lab = collections.Counter(("plan" if i["label"] == 0 else "own" if i["target"][-1] == max(i["target"]) else "pull") for i in items)
print({k: len(v) for k, v in new.items()}, "bad", bad, "mean top-vote share %.2f" % (sum(agree) / max(1, len(agree))))
print("label dist (all agenda cards):", dict(lab))
due = [i for i in items if i["meta"]["due"]]
print("due cards:", len(due), "attend-pull share: %.2f" % (sum(1 for i in due if i["label"] not in (0, len(i["target"]) - 1)) / max(1, len(due))))
