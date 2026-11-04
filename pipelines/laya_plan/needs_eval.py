#!/usr/bin/env python3
"""Evaluate the Laya 'need' decision on v8_data/test.jsonl (persona-disjoint, cloud-teacher labels).
Compares each served model (ports) against the word-boundary keyword baseline. Usage: needs_eval.py <port> [<port>...]"""
import collections, json, sys, urllib.request
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import needs_env as N
items = [json.loads(l) for l in open("/home/andus/Projects/laya-test/v8_data/test.jsonl")]
items = [i for i in items if i["kind"] == "need"]
def ask(port, it):
    body = {"model": "laya-v3", "state": it["state"], "questions": it["questions"]}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    a = (json.loads(urllib.request.urlopen(r).read()).get("answers") or {}).get("need") or {}
    leg = a.get("legend") or {}; mass = {}
    for k, v in (a.get("probabilities") or {}).items():
        if str(k).startswith("__"): continue
        lab = str(leg.get(k, k))
        for sl, d in N.INTENT_CRITERIA.items():
            if lab in (sl, d): mass[sl] = mass.get(sl, 0) + float(v)
    return max(mass, key=mass.get) if mass else None
def report(name, preds):
    per = collections.defaultdict(lambda: [0, 0]); ok = 0
    for it, p in zip(items, preds):
        gold = N.INTENT_SLUGS[it["label"]]; per[gold][1] += 1; per[gold][0] += p == gold; ok += p == gold
    print(f"{name}: accuracy {ok/len(items):.3f} (n={len(items)})  " +
          "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items())), flush=True)
report("keyword baseline", [N.keyword_intent(i["state"]["situation"].split("is doing: ", 1)[1].split("\nLocation")[0]) for i in items])
for port in sys.argv[1:]:
    report(f"laya :{port}", [ask(port, i) for i in items])
