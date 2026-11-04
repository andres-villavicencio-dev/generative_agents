#!/usr/bin/env python3
"""Blind second-teacher labels for action_location prompts (area | room | object).

Replays every distinct recorded action_location prompt through glm-5.3-flash:cloud with the sim's own system
message and appends {key, persona, ans, valid, latency} to ga_traces/glm_loc/labels.jsonl. The teacher never sees
the qwen answer. Resumable (skips keys already written). Usage: glm_loc_label.py [max_calls] [workers]
Paced: each worker sleeps GAP_S between calls to stay under the cloud's per-minute quota."""
import hashlib, json, os, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loc_dataset import load, parse_answer, REPO

URL = os.environ.get("GA_OLLAMA_URL", "http://127.0.0.1:11436") + "/api/chat"
MODEL = os.environ.get("GLM_MODEL", "glm-5.3-flash:cloud")
GAP_S = float(os.environ.get("GLM_GAP_S", "0.5"))
SYSTEM = ("You are a simulation assistant. IMPORTANT: Respond in English only. Output ONLY one line in the exact "
          "YAML format shown in the prompt: output: <value> No explanation, no markdown, no extra keys or lines.")
OUT = os.environ.get("LOC_LABEL_OUT") or f"{REPO}/ga_traces/glm_loc/labels.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def key(r):
    return hashlib.md5((r["persona"] + "\n" + r["prompt"]).encode()).hexdigest()

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            done.add(json.loads(l)["key"])
        except Exception:
            pass
rows, seen = [], set()
for r in load():
    k = key(r)
    if k not in seen and k not in done:
        seen.add(k); rows.append(r)
random.Random(7).shuffle(rows)
_only = os.environ.get("LOC_ONLY_KEYS")  # file of keys (one per line): label only these prompts
if _only:
    _ok = set(open(_only).read().split())
    rows = [r for r in rows if key(r) in _ok]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(rows)
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rows = rows[:limit]
lock = threading.Lock()
fout = open(OUT, "a")
stats = {"n": 0, "valid": 0, "err": 0}

def one(r):
    body = {"model": MODEL, "stream": False, "think": False,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompt"]}],
            "options": {"num_predict": int(os.environ.get("GLM_NUM_PREDICT", "2048")), "temperature": 0}}
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                             {"Content-Type": "application/json"}), timeout=180))
        c = d["message"]["content"]
    except Exception:
        with lock:
            stats["err"] += 1
        time.sleep(5)
        return
    ans = parse_answer("output: " + c.rsplit("output:", 1)[-1].strip().split("\n")[0]) if "output:" in c else None
    tree = r["parsed"]["tree"]
    valid = bool(ans) and ans[0] in tree and ans[1] in tree[ans[0]] and (ans[2] in tree[ans[0]][ans[1]] or not tree[ans[0]][ans[1]])
    with lock:
        fout.write(json.dumps({"key": key(r), "persona": r["persona"], "t": r["t"], "ans": ans, "valid": valid,
                               "raw": c[-200:], "latency_s": round(time.time() - t, 2), "model": MODEL}) + "\n")
        fout.flush()
        stats["n"] += 1; stats["valid"] += valid
        if stats["n"] % 200 == 0:
            print(time.strftime("%H:%M:%S"), stats, flush=True)
    time.sleep(GAP_S)

with ThreadPoolExecutor(workers) as ex:
    list(ex.map(one, rows))
print("DONE", stats, flush=True)
