#!/usr/bin/env python3
"""Rows for the Laya 'replan' decision from logged re-plan-after-chat prompts (run_gpt_prompt_new_decomp_schedule).
Row: persona, window, dur, plan (distinct planned activities), discussed. Dedupe on (persona, discussed). -> ga_traces/replan/rows.jsonl"""
import glob, hashlib, json, os, re, collections
REPO = "/home/andus/Projects/generative_agents"
files = sorted(glob.glob(f"{REPO}/ga_traces/calls/qwen2b-ga-chat-v2_latest*.jsonl"))
R1 = re.compile(r"Here was (.+?)'s originally planned schedule from (\d+:\d+ [AP]M) to (\d+:\d+ [AP]M)", re.S)
R2 = re.compile(r"unexpectedly ended up (.*?) for (\d+) minutes\.\s*What was discussed:\s*(.*?)\.\s*If that means", re.S)
LINE = re.compile(r"^\d+:\d+ ~ \d+:\d+ -- (.*)$", re.M)
seen, rows, c = set(), [], collections.Counter()
for f in files:
    for l in open(f):
        if "new_decomp_schedule" not in l[:300]: continue
        try: r = json.loads(l)
        except Exception: continue
        if r.get("caller") != "run_gpt_prompt_new_decomp_schedule": continue
        p = r["prompt"]; a = R1.search(p); b = R2.search(p)
        if not (a and b): c["nomatch"] += 1; continue
        sched = p.split("originally planned schedule", 1)[1].split("But ", 1)[0]
        acts = []
        for x in LINE.findall(sched):
            base = re.sub(r"\s*\(.*$", "", x).strip()
            if base and base not in acts: acts.append(base)
        k = (a.group(1), b.group(3).strip())
        if k in seen: c["dup"] += 1; continue
        seen.add(k)
        rows.append({"key": hashlib.md5(repr(k).encode()).hexdigest()[:16], "persona": a.group(1),
                     "window": f"{a.group(2)} to {a.group(3)}", "dur": int(b.group(2)),
                     "plan": acts[:6], "discussed": b.group(3).strip()[:400]})
os.makedirs(f"{REPO}/ga_traces/replan", exist_ok=True)
with open(f"{REPO}/ga_traces/replan/rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), dict(c), "personas", len({r['persona'] for r in rows}))
