#!/usr/bin/env python3
"""Replay saved free-choice prompts (new wording) through candidate models.
Usage: venv/bin/python pipelines/choice_model_replay.py [N] model1 model2 ...
Reports per model: parse rate, share following the plan (0), pulls, own (X),
filler share among own choices, latency. Baseline = the recorded 2B answers."""
import glob, json, re, sys, time, collections, urllib.request

REPO = "/home/andus/Projects/generative_agents"
FILLER = re.compile(r"\b(walk|stroll|coffee|tea|break|rest|relax|review|check|read|read(ing)?|unwind)\b", re.I)


def load(n):
    seen, out = set(), []
    for f in sorted(glob.glob(f"{REPO}/ga_traces/calls/*.jsonl")):
        for l in open(f, errors="ignore"):
            if "Default to option 0" not in l or "Options for the next while" not in l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            k = (d["persona"], d["prompt"])
            if d.get("caller") == "run_gpt_prompt_choose_activity" and k not in seen:
                seen.add(k); out.append(d)
    step = max(1, len(out) // n)
    return out[::step][:n]


def parse(t):
    m = re.search(r"choice\s*[:=]\s*(\w+)", t or "", re.I)
    if not m:
        return None
    tok = m.group(1).upper()
    a = re.search(r"activity\s*[:=]\s*(.+)", t, re.I)
    w = re.search(r"why\s*[:=]\s*(.+)", t, re.I)
    if tok == "X":
        return ("own", (a.group(1) if a else "").strip(), (w.group(1) if w else "").strip())
    if tok.isdigit():
        return ("plan" if tok == "0" else "pull", "", (w.group(1) if w else "").strip())
    return None


def ask(model, d):
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": d.get("system", "")},
                         {"role": "user", "content": d["prompt"]}],
            "options": {"num_ctx": 4096, "temperature": 0.3, "num_predict": 1024}}
    r = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(r))
    return resp["message"]["content"], time.time() - t0


def summarize(name, rows, lat):
    c = collections.Counter(r[0] if r else "unparsed" for r in rows)
    own = [r for r in rows if r and r[0] == "own"]
    filler = sum(1 for r in own if FILLER.search(r[1]))
    n = len(rows)
    return {"model": name, "n": n, **{k: f"{100*v/n:.0f}%" for k, v in c.items()},
            "own_filler": f"{filler}/{len(own)}",
            "lat_p50": round(sorted(lat)[len(lat)//2], 1) if lat else None}


if __name__ == "__main__":
    n = int(sys.argv[1]); models = sys.argv[2:]
    data = load(n)
    print("prompts:", len(data), flush=True)
    res = [summarize("recorded 2B (baseline)", [parse(d["content"]) for d in data], [])]
    print(json.dumps(res[-1]), flush=True)
    samples = {}
    for m in models:
        rows, lat = [], []
        for d in data:
            try:
                txt, dt = ask(m, d)
            except Exception as e:
                txt, dt = "", 0
            rows.append(parse(txt)); lat.append(dt)
        samples[m] = [(d["persona"], r) for d, r in zip(data, rows) if r and r[0] == "own"][:5]
        res.append(summarize(m, rows, lat)); print(json.dumps(res[-1]), flush=True)
    json.dump({"results": res, "own_samples": samples}, open(f"{REPO}/choice_model_replay.json", "w"), indent=1)
