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

import hashlib
GLM = {}
_gp = "/home/andus/Projects/generative_agents/ga_traces/glm_loc/labels.jsonl"
if os.environ.get("LOC_GLM", "1") == "1" and os.path.exists(_gp):
    for _l in open(_gp):
        _d = json.loads(_l)
        if _d["valid"]:
            GLM[_d["key"]] = _d["ans"]

VOTES = {}   # key -> {teacher: ans}; LOC_VOTES=1 replaces the qwen label with the 3-teacher majority
if os.environ.get("LOC_VOTES") == "1":
    import glob
    for _f in glob.glob("/home/andus/Projects/generative_agents/ga_traces/loc_votes/*.jsonl"):
        _t = os.path.basename(_f)[:-6]
        for _l in open(_f):
            _d = json.loads(_l)
            if _d["valid"]:
                VOTES.setdefault(_d["key"], {})[_t] = tuple(_d["ans"])

def vote_label(r, idx, opts):
    """3-teacher majority at one level (given identical ancestors). Returns (label, target-dict) or None if unusable."""
    v = VOTES.get(hashlib.md5((r["persona"] + "\n" + r["prompt"]).encode()).hexdigest())
    if not v or len(v) < 3:
        return None
    anc = tuple(r["ans"][:idx])
    ans = [a[idx] for a in v.values() if tuple(a[:idx]) == anc and a[idx] in opts]
    if len(ans) < 2:
        return None
    c = collections.Counter(ans).most_common()
    if c[0][1] == 3:
        return c[0][0], {c[0][0]: 0.95}
    if c[0][1] == 2:
        t = {c[0][0]: 0.80}
        rest = [x for x, _ in c[1:]]
        for x in rest:
            t[x] = 0.12
        return c[0][0], t
    q = r["ans"][idx]          # three-way split: only keep it if qwen (the sim's own behaviour) breaks the tie
    return (q, {q: 0.55, **{x: 0.15 for x, _ in c if x != q}}) if q in dict(c) else None

def mk(r, level, area=None, room=None):
    bl = build_level(r["persona"], r["parsed"], level, area, room)
    if bl is None:
        return None
    state, qs, opts = bl
    label = {"area": r["ans"][0], "room": r["ans"][1], "obj": r["ans"][2]}[level]
    if label not in opts:
        return None
    k = len(opts)
    vt = vote_label(r, {"area": 0, "room": 1, "obj": 2}[level], opts) if VOTES else None
    if VOTES and vt is None:
        return None
    if vt is not None:
        label = vt[0]
        rest_p = (1 - sum(vt[1].values())) / max(1, k - len(vt[1]))
        tgt = [vt[1].get(o, rest_p) for o in opts]
        s_ = sum(tgt)
        return {"kind": "loc", "persona": r["persona"], "t": r["t"], "state": state, "questions": qs,
                "target": [x / s_ for x in tgt], "label": opts.index(label), "n_votes": 3}
    g = GLM.get(hashlib.md5((r["persona"] + "\n" + r["prompt"]).encode()).hexdigest())
    idx = {"area": 0, "room": 1, "obj": 2}[level]
    # consensus: GLM agreeing on this level (and every level above it) sharpens the label; disagreeing splits it
    if g is not None and tuple(g[:idx]) == tuple(r["ans"][:idx]):
        if g[idx] == label:
            conf, other = 0.93, None
        elif g[idx] in opts:
            conf, other = 0.55, g[idx]
        else:
            conf, other = CONF, None
    else:
        conf, other = CONF, None
    if other and os.environ.get("LOC_DROP_DISAGREE") == "1":
        return None          # teachers disagree on this level: ambiguous label, leave it out of training
    rest = 1 - conf - (0.30 if other else 0.0)
    tgt = [max(rest, 0) / max(1, k - 1 - (1 if other else 0))] * k if k > 1 else [1.0]
    tgt[opts.index(label)] = conf
    if other:
        tgt[opts.index(other)] = 0.30
    s_ = sum(tgt); tgt = [x / s_ for x in tgt]
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
    if by_act[(r["persona"], a)] >= int(os.environ.get("LOC_CAP", "2")):
        continue
    by_act[(r["persona"], a)] += 1
    pick.append(r)
    if len(pick) >= N_CALLS:
        break
items = [it for r in pick for it in level_items(r)]
_dup = int(os.environ.get("LOC_COLIVING_DUP", "1"))
if _dup > 1:   # the artist co-living space holds 89% of room errors: oversample its clean room/object items
    items += [it for it in items if it["state"]["level"] in ("room", "obj") and "co-living" in it["state"]["situation"]
              and it["target"] and max(it["target"]) >= 0.9] * (_dup - 1)
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
