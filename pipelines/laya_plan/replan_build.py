#!/usr/bin/env python3
"""Build the Laya 'replan' dataset on top of v8_data -> v9_data. Labels: glm+minimax agree -> 0.95; else nemotron tie-break (0.80,
minority 0.20); else dropped. Persona split = v4 test personas (disjoint). Usage: replan_build.py [out_dir]"""
import collections, json, os, random, shutil, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import replan_gate as G
REPO = "/home/andus/Projects/generative_agents"
BASE = "/home/andus/Projects/laya-test/v8_data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/andus/Projects/laya-test/v9_data"
P = set(json.load(open("/home/andus/Projects/laya-test/v4_data/build_report.json"))["test_personas"])
rng = random.Random(91)
rows = {json.loads(l)["key"]: json.loads(l) for l in open(f"{REPO}/ga_traces/replan/rows.jsonl")}
T = {}
for m in ("glm-5.3", "minimax-m3.1-flash", "nemotron-3-ultra"):
    p = f"{REPO}/ga_traces/replan/{m}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l)
            if d["ans"]: T.setdefault(d["key"], {})[m] = d["ans"]
def label(v):
    g, mm, n = v.get("glm-5.3"), v.get("minimax-m3.1-flash"), v.get("nemotron-3-ultra")
    if g and mm and g == mm: return g, 0.95
    if g and mm and n and n in (g, mm): return n, 0.80
    return None
items = []; stats = collections.Counter()
for k, v in T.items():
    r = rows.get(k); lab = label(v) if r else None
    if not lab: stats["dropped"] += 1; continue
    ans, conf = lab; stats[ans] += 1
    tgt = [conf if s == ans else 1 - conf for s in G.SLUGS]
    items.append({"kind": "replan", "persona": r["persona"], "t": 0,
                  "state": {"who": r["persona"], "situation": G.situation(r["persona"], r["window"], r["dur"], r["plan"], r["discussed"])},
                  "questions": {"replan": {"type": "choice", "instructions": G.INSTRUCTIONS, "criteria": G.CRITERIA}},
                  "target": tgt, "label": G.SLUGS.index(ans)})
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(BASE):
    if f.endswith(".jsonl") and not os.path.exists(f"{OUT}/{f}"):
        shutil.copy(f"{BASE}/{f}", f"{OUT}/{f}")
test = [i for i in items if i["persona"] in P]; trainall = [i for i in items if i["persona"] not in P]
rng.shuffle(trainall); ncal = max(20, len(trainall) // 20)
calib, train = trainall[:ncal], trainall[ncal:]
def app(name, its):
    with open(f"{OUT}/{name}.jsonl", "a") as f:
        for i in its: f.write(json.dumps(i) + "\n")
app("train", train); app("calib", calib); app("test", test)
print("replan items", len(items), "train", len(train), "calib", len(calib), "test", len(test), dict(stats))
