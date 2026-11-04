#!/usr/bin/env python3
"""Extract distinct (action description, location) pairs from a sim's movement files for need-intent labelling.
Stratified by the keyword intent so rare classes (toilet, wash, sleep) are represented.
Usage: needs_extract.py <sim> <first_step> <last_step> [stride] -> ga_traces/needs_intent/rows.jsonl"""
import collections, hashlib, json, os, random, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import needs_env as N
REPO = "/home/andus/Projects/generative_agents"
sim, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 12
CAP = {"none": 1800}; DEF = 450
mv = f"{REPO}/environment/frontend_server/storage/{sim}/movement"
seen = {}
for s in range(a, b, stride):
    try: d = json.load(open(f"{mv}/{s}.json"))
    except Exception: continue
    for n, p in d["persona"].items():
        full = p.get("description") or ""
        if " @ " not in full: continue
        desc, addr = full.split(" @ ", 1)
        k = (desc.strip().lower(), addr.strip().lower())
        if k not in seen:
            seen[k] = {"persona": n, "desc": desc.strip(), "addr": addr.strip(), "step": s}
by = collections.defaultdict(list)
for r in seen.values():
    by[N.keyword_intent(r["desc"])].append(r)
rng = random.Random(7); out = []
for cls, rows in by.items():
    rng.shuffle(rows); out += rows[:CAP.get(cls, DEF)]
rng.shuffle(out)
for r in out:
    r["key"] = hashlib.md5((r["desc"] + "|" + r["addr"]).encode()).hexdigest()
os.makedirs(f"{REPO}/ga_traces/needs_intent", exist_ok=True)
with open(f"{REPO}/ga_traces/needs_intent/rows.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print("distinct pairs", len(seen), "->", len(out), {k: min(len(v), CAP.get(k, DEF)) for k, v in by.items()})
