#!/usr/bin/env python3
"""Evaluate the replan decision on v9_data/test.jsonl. usage: replan_eval.py <port>...  Reports accuracy, and for keep-conf thresholds
the share of re-plans skipped and the share of REAL commitments wrongly skipped (the costly error)."""
import json, sys
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import replan_gate as G
items = [json.loads(l) for l in open("/home/andus/Projects/laya-test/v9_data/test.jsonl")]
items = [i for i in items if i["kind"] == "replan"]
for port in sys.argv[1:]:
    G.os.environ["GA_REPLAN_LAYA_URL"] = f"http://127.0.0.1:{port}"
    res = []
    for it in items:
        s = it["state"]["situation"]
        import re
        m = re.match(r"(.+?) had planned, between (.+?): (.*?)\.\nThen .+? had a (\d+)-minute conversation. What was discussed: (.*)$", s, re.S)
        slug, keep = G.ask(m.group(1), m.group(2), int(m.group(4)), m.group(3).split("; "), m.group(5))
        if keep is not None: res.append((G.SLUGS[it["label"]], keep))
    acc = sum((("keep" if k >= .5 else "replan") == y) for y, k in res) / len(res)
    print(f"port {port}: n={len(res)} accuracy {acc:.3f}  true keep share {sum(y=='keep' for y,_ in res)/len(res):.2f}")
    for th in (0.5, 0.75, 0.9):
        sk = [(y, k) for y, k in res if k >= th]
        bad = sum(y == "replan" for y, _ in sk)
        print(f"  keep>={th}: skips {len(sk)/len(res):.2f} of replans; wrongly skipped real commitments {bad} ({bad/max(1,sum(y=='replan' for y,_ in res)):.2f} of all commitments)")
