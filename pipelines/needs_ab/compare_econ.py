#!/usr/bin/env python3
"""Matched-sim-time economy comparison of two runs (default control n56 vs treatment n70).

usage: compare_econ.py [control] [treatment] [--until "Feb 14 08:00"]
Reads storage/<sim>/resources/{economy_v2,agent_cards}.json and ga_traces_<sim>/needs_timeseries.jsonl.
Pay-log rows are cut at the common cutoff (default: the earlier of the two sims' curr_time).
"""
import json, re, sys, collections, statistics as st
from datetime import datetime
ROOT = "/home/andus/Projects/generative_agents"
ST = f"{ROOT}/environment/frontend_server/storage"
NEEDS = ("hunger", "energy", "hydration", "hygiene", "bladder", "social", "comfort", "stimulation")

def pt(s):  # "Feb 14 01:07"
    return datetime.strptime("2023 " + s, "%Y %b %d %H:%M")

def curr(sim):
    return datetime.strptime(json.load(open(f"{ST}/{sim}/reverie/meta.json"))["curr_time"], "%B %d, %Y, %H:%M:%S")

def summarize(sim, cut):
    out = {}
    eco = json.load(open(f"{ST}/{sim}/resources/economy_v2.json"))
    pl = [p for p in eco["pay_log"] if pt(p["t"]) <= cut]
    k = collections.defaultdict(lambda: [0, 0.0])
    for p in pl:
        k[p["kind"]][0] += 1; k[p["kind"]][1] += p["amount"]
    out["pay"] = {a: (n, round(s, 1)) for a, (n, s) in sorted(k.items())}
    sales = collections.defaultdict(float)
    items = collections.Counter()
    for p in pl:
        if p["kind"] == "sale":
            sales[p.get("venue")] += p["amount"]
            for part in (p.get("note") or "").split(","):
                part = part.strip()
                m = re.match(r"([\d.]+)x\s+(.*)", part)
                if m: items[m.group(2)] += float(m.group(1))
    out["sales_by_venue"] = {a: round(b, 1) for a, b in sales.items()}
    out["items"] = dict(items.most_common(8))
    out["buyers"] = len({p["who"] for p in pl if p["kind"] == "sale"})
    out["employees"] = sum(len(v) for v in eco["employees"].values())
    out["postings"] = len(eco["postings"]); out["hires"] = len(eco["hire_log"])
    cards = json.load(open(f"{ST}/{sim}/resources/agent_cards.json"))
    w = sorted(c["wallet"] for c in cards.values())
    n = len(w); tot = sum(w)
    gini = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(w)) / (n * tot) if tot else 0
    out["wallet"] = dict(n=n, total=round(tot), mean=round(tot / n, 1), median=w[n // 2], min=w[0], max=w[-1], gini=round(gini, 3))
    out["jobs"] = sum(1 for c in cards.values() if c.get("job"))
    out["fund"] = round(eco["fund"]["balance"], 1)
    # needs at/near cutoff (latest snapshot <= cut)
    last = {}
    try:
        for line in open(f"{ROOT}/ga_traces_{sim}/needs_timeseries.jsonl"):
            r = json.loads(line)
            if datetime.strptime(r["t"], "%Y-%m-%d %H:%M:%S") <= cut: last[r["who"]] = r
    except FileNotFoundError:
        pass
    if last:
        out["needs_mean"] = {x: round(st.mean(r[x] for r in last.values()), 1) for x in NEEDS}
        out["needs_lt20"] = {x: sum(1 for r in last.values() if r[x] < 20) for x in NEEDS}
    return out

def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    ctl, trt = (a + ["n56", "n70"])[:2] if len(a) < 2 else a[:2]
    cut = min(curr(ctl), curr(trt))
    if "--until" in sys.argv:
        cut = min(cut, pt(sys.argv[sys.argv.index("--until") + 1]))
    print(f"matched cutoff: {cut}  (control {ctl} at {curr(ctl)}, treatment {trt} at {curr(trt)})")
    R = {ctl: summarize(ctl, cut), trt: summarize(trt, cut)}
    keys = []
    for s in R.values():
        for key in s:
            if key not in keys: keys.append(key)
    for key in keys:
        print(f"\n## {key}")
        for s in (ctl, trt):
            print(f"  {s:>4}: {R[s].get(key)}")

if __name__ == "__main__":
    main()
