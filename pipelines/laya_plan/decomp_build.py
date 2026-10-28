#!/usr/bin/env python3
"""Seed ga_traces/decomp_cache.json from the call traces and simulate the online hit rate/validity."""
import glob, json, re, sys, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
from persona.prompt_template import decomp_cache as D
REPO = "/home/andus/Projects/generative_agents"
calls = []
for f in sorted(glob.glob(f"{REPO}/ga_traces/calls/qwen2b-ga-chat-v2_latest*.jsonl")):
    for l in open(f, errors="ignore"):
        if "run_gpt_prompt_task_decomp" not in l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("caller") != "run_gpt_prompt_task_decomp" or not d.get("content"):
            continue
        m = re.search(r"Activity to break down: (.*?) from \d+:\d+\w+ ~ \d+:\d+\w+ \(total duration in minutes (\d+)\)", d["prompt"])
        if not m:
            continue
        steps = []
        for line in d["content"].split("\n"):
            k = re.match(r"\s*\d+\)\s*(.*?)\s*\(duration in minutes:\s*(\d+)", line)
            if k:
                steps.append((re.sub(r"\.$", "", k.group(1).strip()), int(k.group(2))))
        calls.append((float(d["ts"]), d["persona"], m.group(1), int(m.group(2)), steps))
calls.sort()
cache = collections.defaultdict(list); hit = bad = 0
for ts, persona, act, dur, steps in calls:
    first = persona.split()[0]
    k = D.key(act, dur)
    if cache.get(k):
        hit += 1
    st = D._clean_steps(steps, first, dur)
    if st:
        g = D.gender_of_text(" ".join(t for t, _ in st))
        if not any(e["steps"] == st for e in cache[k]) and len(cache[k]) < D.MAX_PER_KEY:
            cache[k].append({"g": g, "steps": st})
print("calls", len(calls), "online hit rate", round(hit / len(calls), 3), "keys", len(cache),
      "entries", sum(len(v) for v in cache.values()))
json.dump(cache, open(f"{REPO}/ga_traces/decomp_cache.json", "w"))
