#!/usr/bin/env python3
"""Label every distinct free-choice decision context in the call traces with a
teacher (glm-5.3-flash:cloud, two votes, neutral+honest tail) -> v5 'choose'
records for laya. Output: /home/andus/Projects/laya-test/v5_data/choose_raw.jsonl
Usage: laya_choose_label.py [max_items]"""
import glob, json, os, re, sys, threading, queue, time
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines")
from choice_model_replay import parse
from choice_prompt_variants import TAILS, retail
import urllib.request

REPO = "/home/andus/Projects/generative_agents"
OUT = "/home/andus/Projects/laya-test/v5_data"
TEACHER = "glm-5.3-flash:cloud"
os.makedirs(OUT, exist_ok=True)


def contexts(limit):
    seen, out = set(), []
    for f in sorted(glob.glob(f"{REPO}/ga_traces/calls/*.jsonl")):
        for l in open(f, errors="ignore"):
            if "Options for the next while" not in l or "choose_activity" not in l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            p = d["prompt"]
            if "\nX. " not in p or "Default to option 0" not in p and "Prefer option 0" not in p:
                continue
            head, _, rest = p.partition("Options for the next while:\n")
            k = (d["persona"], head[-700:], rest.split("\nX. ")[0])
            if k in seen:
                continue
            seen.add(k)
            out.append({"persona": d["persona"], "t": float(d["ts"]), "prompt": p,
                        "system": d.get("system", "")})
    return out[:limit] if limit else out


def vote(rec, temp):
    p = rec["prompt"]
    if "Default to option 0" in p:
        p = retail(p, "neutral+honest", TAILS["neutral+honest"])
    else:   # first-run wording ("Prefer option 0 ...")
        i = p.index("Prefer option 0")
        first = re.search(r"\n(\w+) planned this morning", p).group(1)
        p = p[:i] + TAILS["neutral+honest"].format(f=first)
    body = {"model": TEACHER, "stream": False, "think": False,
            "messages": [{"role": "system", "content": rec["system"]},
                         {"role": "user", "content": p}],
            "options": {"num_ctx": 4096, "temperature": temp, "num_predict": 1024}}
    r = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(r))["message"]["content"]
    m = re.search(r"choice\s*[:=]\s*(\w+)", t, re.I)
    if not m:
        return None
    tok = m.group(1).upper()
    return "own" if tok == "X" else (int(tok) if tok.isdigit() else None)


def worker(q, lock, fh, stats):
    while True:
        try:
            rec = q.get_nowait()
        except queue.Empty:
            return
        votes = []
        for temp in (0.3, 0.8):
            for _ in range(2):
                try:
                    v = vote(rec, temp)
                    if v is not None:
                        votes.append(v); break
                except Exception:
                    time.sleep(2)
        if votes:
            rec = dict(rec); rec["votes"] = votes
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                stats["done"] += 1
        else:
            stats["bad"] += 1


if __name__ == "__main__":
    items = contexts(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print("distinct contexts:", len(items), flush=True)
    done_path = f"{OUT}/choose_raw.jsonl"
    have = set()
    if os.path.exists(done_path):
        for l in open(done_path):
            r = json.loads(l); have.add((r["persona"], r["prompt"]))
    items = [r for r in items if (r["persona"], r["prompt"]) not in have]
    q = queue.Queue()
    for r in items:
        q.put(r)
    lock, stats = threading.Lock(), {"done": 0, "bad": 0}
    fh = open(done_path, "a")
    ths = [threading.Thread(target=worker, args=(q, lock, fh, stats)) for _ in range(4)]
    [t.start() for t in ths]
    while any(t.is_alive() for t in ths):
        time.sleep(30); print(stats, q.qsize(), flush=True)
    print("finished", stats, flush=True)
