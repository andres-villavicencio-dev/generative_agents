#!/usr/bin/env python3
"""Build the Laya 'need' dataset: v7e_data (replay of every existing kind) + need-intent items labelled by cloud teachers.
Labels: glm-5.3 + minimax agree -> 0.95 on the slug; else nemotron as tie-break (2/3 -> 0.80, minority 0.10); else dropped.
Persona split follows the v4 test personas (disjoint). Usage: needs_build.py [out_dir]"""
import collections, json, os, random, shutil, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import needs_env as N
REPO = "/home/andus/Projects/generative_agents"
BASE = "/home/andus/Projects/laya-test/v7e_data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/andus/Projects/laya-test/v8_data"
P = set(json.load(open("/home/andus/Projects/laya-test/v4_data/build_report.json"))["test_personas"])
rng = random.Random(81)
rows = {}
for l in open(f"{REPO}/ga_traces/needs_intent/rows.jsonl"):
    r = json.loads(l); rows[r["key"]] = r
T = {}
for m in ("glm-5.3", "minimax-m3.1-flash", "nemotron-3-ultra"):
    p = f"{REPO}/ga_traces/needs_intent/{m}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            d = json.loads(l)
            if d["ans"]: T.setdefault(d["key"], {})[m] = d["ans"]
def label(v):
    g, mm, n = v.get("glm-5.3"), v.get("minimax-m3.1-flash"), v.get("nemotron-3-ultra")
    if g and mm and g == mm: return g, 0.95, None
    if g and mm and n and n in (g, mm): return n, 0.80, (mm if n == g else g)
    return None
K = len(N.INTENT_SLUGS); items = []; stats = collections.Counter()
for k, v in T.items():
    r = rows.get(k); lab = label(v) if r else None
    if not lab: stats["dropped"] += 1; continue
    ans, conf, minority = lab
    tgt = [(1 - conf) / (K - 1)] * K; tgt[N.INTENT_SLUGS.index(ans)] = conf
    if minority:
        tgt = [0.02] * K; tgt[N.INTENT_SLUGS.index(ans)] = 0.80; tgt[N.INTENT_SLUGS.index(minority)] = 0.10
    stats[ans] += 1
    items.append({"kind": "need", "persona": r["persona"], "t": 0,
                  "state": {"who": r["persona"], "situation": f"{r['persona']} is doing: {r['desc']}\nLocation: {r['addr']}"},
                  "questions": {"need": {"type": "choice",
                      "instructions": "Which basic need is this person acting on right now? Pick the single best match.",
                      "criteria": N.INTENT_CRITERIA}},
                  "target": tgt, "label": N.INTENT_SLUGS.index(ans)})
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
print("need items", len(items), "train", len(train), "calib", len(calib), "test", len(test), dict(stats))
