#!/usr/bin/env python3
"""Blind cloud-teacher labels for the re-plan decision. Env: REPLAN_MODEL, REPLAN_N (sample size, default 2500), REPLAN_WORKERS.
Output ga_traces/replan/<model>.jsonl {key, ans(replan|keep|None), raw}. Resumable, no HTTP timeout (project rule)."""
import json, os, random, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
REPO = "/home/andus/Projects/generative_agents"
MODEL = os.environ["REPLAN_MODEL"]; N = int(os.environ.get("REPLAN_N", "2500"))
OUT = f"{REPO}/ga_traces/replan/{MODEL.split(':')[0]}.jsonl"
URL = os.environ.get("GA_OLLAMA_URL", "http://127.0.0.1:11436") + "/api/chat"
SYSTEM = ("You are a simulation assistant. IMPORTANT: Respond in English only. Output ONLY one line in the exact "
          "format: output: <replan or keep>  No explanation, no markdown, no extra lines.")
def prompt(r):
    return (f"{r['persona']} had planned, between {r['window']}: {'; '.join(r['plan'])}.\n"
            f"Then {r['persona']} had a {r['dur']}-minute conversation. What was discussed: {r['discussed']}\n\n"
            "Does this conversation require changing the rest of the plan? Answer replan ONLY if it committed "
            f"{r['persona']} to be somewhere or do something specific at a particular time or soon (an appointment, "
            "a promise, an invitation accepted, a job or task to start, a meeting). Answer keep if it was ordinary "
            "talk, opinions, small talk or gossip that does not change what they must do next.\n"
            "- replan: the conversation created a commitment that changes the plan\n"
            "- keep: no commitment; carry on with the plan\n\nAnswer: output: replan  or  output: keep")
rows = [json.loads(l) for l in open(f"{REPO}/ga_traces/replan/rows.jsonl")]
random.Random(7).shuffle(rows); rows = rows[:N]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
rows = [r for r in rows if r["key"] not in done]
fout = open(OUT, "a"); lock = threading.Lock(); st = {"n": 0, "ok": 0, "err": 0}
def one(r):
    body = {"model": MODEL, "stream": False, "think": False,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(r)}],
            "options": {"num_predict": 4096, "temperature": 0}}
    try:
        c = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                     {"Content-Type": "application/json"})))["message"]["content"]
    except Exception:
        with lock: st["err"] += 1
        time.sleep(5); return
    m = re.findall(r"output:\s*(replan|keep)", c.lower())
    ans = m[-1] if m else None
    with lock:
        fout.write(json.dumps({"key": r["key"], "ans": ans, "raw": c[-60:], "model": MODEL}) + "\n"); fout.flush()
        st["n"] += 1; st["ok"] += ans is not None
        if st["n"] % 100 == 0: print(time.strftime("%H:%M:%S"), st, flush=True)
with ThreadPoolExecutor(int(os.environ.get("REPLAN_WORKERS", "3"))) as ex: list(ex.map(one, rows))
print("DONE", st, flush=True)
