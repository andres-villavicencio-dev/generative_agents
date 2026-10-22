#!/usr/bin/env python3
"""Evaluate Laya checkpoints against the blind m3.1 teacher on held-out splits.

Decoding mirrors the sim (laya_scoring.py): argmax of `probabilities`, confidence =
probability of the argmax (no persona temperament shift, no talk yes-bias — raw model).

usage: laya_v4_eval.py <data_dir> <out_json> name=ckpt [name=ckpt ...] [--splits test,offdiag_time]
"""
import json
import os
import sys
import time

import numpy as np
import laya
from laya.common import ece_score

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA, OUT = args[0], args[1]
MODELS = [a.split("=", 1) for a in args[2:]]
SPLITS = ["test"]
for a in sys.argv[1:]:
    if a.startswith("--splits="):
        SPLITS = a.split("=", 1)[1].split(",")


def metrics(rows):
    res = {}
    for kind in ("poignancy", "talk", "react"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        conf = np.array([r["conf"] for r in rs])
        corr = np.array([r["pred"] == r["label"] for r in rs], dtype=float)
        m = {"n": len(rs), "acc": float(corr.mean()), "mean_conf": float(conf.mean()),
             "ece": ece_score(conf, corr)}
        if kind == "poignancy":
            err = np.array([abs(r["pred"] - r["label"]) for r in rs])
            m["mae"] = float(err.mean())
            m["within1"] = float((err <= 1).mean())
            m["exp_mae"] = float(np.mean([abs(r["exp"] - r["label"]) for r in rs]))
            m["mean_pred"] = float(np.mean([r["pred"] for r in rs]) + 1)
            m["mean_teacher"] = float(np.mean([r["label"] for r in rs]) + 1)
        else:
            keys = list(rs[0]["keys"])
            m["teacher_dist"] = {k: sum(1 for r in rs if keys[r["label"]] == k) for k in keys}
            m["pred_dist"] = {k: sum(1 for r in rs if keys[r["pred"]] == k) for k in keys}
            # majority-class baseline for context
            m["majority_baseline"] = max(m["teacher_dist"].values()) / len(rs)
        res[kind] = m
    return res


def run(agent, items):
    rows = []
    for it in items:
        (qid, qdef), = it["questions"].items()
        a = agent.predict(it["state"], it["questions"])["answers"][qid]
        probs = {k: v for k, v in a["probabilities"].items() if not str(k).startswith("__")}
        keys = list(probs.keys()) if qdef["type"] == "choice" else [str(i) for i in range(len(qdef["criteria"]))]
        p = np.array([probs[k] for k in keys])
        pred = int(p.argmax())
        rows.append({"kind": it["kind"], "persona": it["persona"], "label": it["label"], "pred": pred,
                     "conf": float(p[pred] / max(p.sum(), 1e-9)), "keys": keys,
                     "exp": float((np.arange(len(p)) * p).sum() / max(p.sum(), 1e-9))})
    return rows


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for split in SPLITS:
        items = [json.loads(l) for l in open(os.path.join(DATA, f"{split}.jsonl"))]
        for name, ckpt in MODELS:
            agent = laya.load(ckpt, device="cuda")
            t0 = time.time()
            rows = run(agent, items)
            dt = time.time() - t0
            m = metrics(rows)
            m["_latency_ms_per_item"] = 1000 * dt / len(items)
            out.setdefault(split, {})[name] = m
            with open(OUT.replace(".json", f".{split}.{name}.preds.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(split, name, json.dumps(m, indent=1), flush=True)
            del agent
            import torch
            torch.cuda.empty_cache()
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
