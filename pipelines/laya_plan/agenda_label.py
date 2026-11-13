#!/usr/bin/env python3
"""Blind cloud-teacher votes on agenda cards (cards.jsonl). One process per teacher.
Env: AG_MODEL (cloud model, required), AG_LIMIT (cards, due-first), AG_WORKERS, AG_TEMP.
Resumable. Output ga_traces/agenda_choice/<model>.jsonl {key, vote (0|k|'own'|None), raw, model}.
Neutral-honest tail (as laya_choose_label.py); no HTTP timeout added (project rule)."""
import json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines")
import re
from choice_prompt_variants import TAILS
def retail(p, name, tail):
    first = re.search(r"\n([^\n]+?) planned this morning", p).group(1).split()[0]   # tolerant: "Soo-jin"
    i = p.index("Default to option 0")
    return p[:i] + tail.format(f=first)
REPO = "/home/andus/Projects/generative_agents"
D = f"{REPO}/ga_traces/agenda_choice"
MODEL = os.environ["AG_MODEL"]
OUT = f"{D}/{MODEL.split(':')[0]}.jsonl"
URL = os.environ.get("GA_OLLAMA_URL", "http://127.0.0.1:11436") + "/api/chat"
SYSTEM = "You are a simulation assistant. IMPORTANT: Respond in English only."
cards = [json.loads(l) for l in open(f"{D}/cards.jsonl")]
cards.sort(key=lambda r: not r["meta"]["due"])
cards = cards[:int(os.environ.get("AG_LIMIT", len(cards)))]
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
cards = [r for r in cards if r["key"] not in done]
fout = open(OUT, "a"); lock = threading.Lock(); st = {"n": 0, "ok": 0, "err": 0}
def one(r):
    p = retail(r["prompt"], "neutral+honest", TAILS["neutral+honest"])
    body = {"model": MODEL, "stream": False, "think": False,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            "options": {"num_ctx": 4096, "temperature": float(os.environ.get("AG_TEMP", "0.4")), "num_predict": 2048}}
    try:
        c = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                     {"Content-Type": "application/json"})))["message"]["content"]
    except Exception:
        with lock: st["err"] += 1
        time.sleep(5); return
    m = re.search(r"choice\s*[:=]\s*(\w+)", c, re.I)
    tok = m.group(1).upper() if m else ""
    vote = "own" if tok == "X" else (int(tok) if tok.isdigit() else None)
    with lock:
        fout.write(json.dumps({"key": r["key"], "vote": vote, "raw": c[-160:], "model": MODEL}) + "\n"); fout.flush()
        st["n"] += 1; st["ok"] += vote is not None
        if st["n"] % 100 == 0: print(time.strftime("%H:%M:%S"), MODEL, st, flush=True)
with ThreadPoolExecutor(int(os.environ.get("AG_WORKERS", "4"))) as ex: list(ex.map(one, cards))
print("DONE", MODEL, st, flush=True)
