#!/usr/bin/env python3
"""Blind cloud-teacher labels for need-intent rows (ga_traces/needs_intent/rows.jsonl).
Env: NEEDS_MODEL (cloud model), NEEDS_OUT (jsonl), NEEDS_ONLY (file of keys; default all), NEEDS_WORKERS.
Resumable. Output line: {key, ans(slug|None), raw, model}. No HTTP timeout added (project rule)."""
import json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/andus/Projects/generative_agents/reverie/backend_server")
import needs_env as N
REPO = "/home/andus/Projects/generative_agents"
MODEL = os.environ["NEEDS_MODEL"]
OUT = os.environ.get("NEEDS_OUT") or f"{REPO}/ga_traces/needs_intent/{MODEL.split(':')[0]}.jsonl"
URL = os.environ.get("GA_OLLAMA_URL", "http://127.0.0.1:11436") + "/api/chat"
workers = int(os.environ.get("NEEDS_WORKERS", "3"))
SYSTEM = ("You are a simulation assistant. IMPORTANT: Respond in English only. Output ONLY one line in the exact "
          "format: output: <slug>  No explanation, no markdown, no extra lines.")
OPTS = "\n".join(f"- {o}" for o in N.INTENT_OPTIONS)
def prompt(r):
    return (f"{r['persona']} is doing: {r['desc']}\nLocation: {r['addr']}\n\n"
            "Which basic need is this person acting on right now? Pick the single best match "
            "(the purpose of the action, not just the words in it).\n"
            f"{OPTS}\n\nAnswer with the slug before the colon, e.g. output: eat")
rows = [json.loads(l) for l in open(f"{REPO}/ga_traces/needs_intent/rows.jsonl")]
if os.environ.get("NEEDS_ONLY"):
    keep = set(open(os.environ["NEEDS_ONLY"]).read().split()); rows = [r for r in rows if r["key"] in keep]
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
    m = re.findall(r"output:\s*([a-z]+)", c.lower())
    ans = m[-1] if m and m[-1] in N.INTENT_SLUGS else None
    with lock:
        fout.write(json.dumps({"key": r["key"], "ans": ans, "raw": c[-80:], "model": MODEL}) + "\n"); fout.flush()
        st["n"] += 1; st["ok"] += ans is not None
        if st["n"] % 100 == 0: print(time.strftime("%H:%M:%S"), st, flush=True)
with ThreadPoolExecutor(workers) as ex: list(ex.map(one, rows))
print("DONE", st, flush=True)
