#!/usr/bin/env python3
"""Per-item loc eval on the v7 test set sliced by label quality: all items, GLM-labelled, qwen==GLM agree, qwen!=GLM disagree.
usage: loc_eval_consensus.py port [port...]  -> accuracy per level per slice for each server."""
import collections, hashlib, json, os, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines/laya_plan")
import persona.prompt_template.laya_scoring as L
from persona.prompt_template.laya_loc import ask_level
from loc_dataset import load
key = {}
for r in load():
    key[(r["persona"], r["t"])] = (hashlib.md5((r["persona"] + "\n" + r["prompt"]).encode()).hexdigest(), r["ans"])
glm = {}
for l in open("/home/andus/Projects/generative_agents/ga_traces/glm_loc/labels.jsonl"):
    d = json.loads(l)
    if d["valid"]:
        glm[d["key"]] = d["ans"]
items = [json.loads(l) for l in open(os.environ.get("EVAL_TEST", "/home/andus/Projects/laya-test/v7_data/test.jsonl"))]
items = [i for i in items if i["kind"] == "loc"]
IDX = {"area": 0, "room": 1, "obj": 2}
for port in sys.argv[1:]:
    L.LAYA_URL = f"http://127.0.0.1:{port}"
    acc = collections.defaultdict(lambda: [0, 0])
    for it in items:
        q = it["questions"]; opts = list(q["loc"]["criteria"]); lvl = it["state"]["level"]
        k, qa = key.get((it["persona"], it["t"]), (None, None))
        g = glm.get(k)
        pr = ask_level(it["state"], q, opts)
        if not pr:
            continue
        pred = max(pr, key=pr.get); qlab = opts[it["label"]]
        slices = ["all"]
        if "co-living" in it["state"]["situation"]:
            slices.append("coliving")
            if g is not None and tuple(g[:IDX[lvl]]) == tuple(qa[:IDX[lvl]]) and g[IDX[lvl]] == qlab:
                slices.append("coliving_agree")
        if g is not None and tuple(g[:IDX[lvl]]) == tuple(qa[:IDX[lvl]]):
            slices.append("glm_scored")
            slices.append("agree" if g[IDX[lvl]] == qlab else "disagree")
        for s in slices:
            acc[(s, lvl)][1] += 1
            acc[(s, lvl)][0] += pred == qlab
        if g is not None and tuple(g[:IDX[lvl]]) == tuple(qa[:IDX[lvl]]):
            acc[("vs_glm", lvl)][1] += 1; acc[("vs_glm", lvl)][0] += pred == g[IDX[lvl]]
    print("PORT", port)
    for k in sorted(acc):
        print(" ", k, f"{acc[k][0]/acc[k][1]:.3f} n={acc[k][1]}")
