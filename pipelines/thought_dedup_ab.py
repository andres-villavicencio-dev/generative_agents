#!/usr/bin/env python3
"""A/B validation for the thought-dedup gate (GA_THOUGHT_DEDUP), Nov 2026.

Part A — gate replay: sample real n76 utterance prompts, extract the memory
block the model actually saw, and replay it through the gate's Jaccard filter
(0.45, first-kept-wins, plan markers protected). Reports the suppression rate
and spot examples.

Part B — model A/B: for each sampled prompt, build a 'gated' version by
removing the lines the gate would have suppressed, then ask the live chat
model (GA ollama lane) for an utterance with BOTH the original and the gated
prompt. Compares output similarity (difflib), format compliance and prompt
token counts. High similarity + same format + ~55% fewer memory lines means
the gate changes the memory, not the conversation.

Usage: thought_dedup_ab.py [n_prompts=40] [seed=20261116]
"""
import datetime, difflib, glob, json, random, re, sys, time, urllib.request

REPO = "/home/andus/Projects/generative_agents"
sys.path.insert(0, f"{REPO}/reverie/backend_server")
from persona.memory_structures import associative_memory as am

CALLS = sorted(glob.glob(f"{REPO}/ga_traces/calls/qwen2b-ga-chat-v2_latest*.jsonl"))
LO = datetime.datetime(2026, 11, 14, 2, 44).timestamp()
HI = datetime.datetime(2026, 11, 15, 3, 18).timestamp()
URL = "http://127.0.0.1:11436/api/chat"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 20261116)

MEM_RE = re.compile(r"(Here is the memory that is in .+?'s head: \n)(.*?)(Here is the|PART \d|$)", re.S)
LINE_RE = re.compile(r"^(?:- )?(.+)$")


def gate_lines(block):
    """Replay a rendered memory block through the gate: keep the first of each
    near-dup cluster; plan markers are protected on both sides (never dropped,
    never matched against). Returns (kept, suppressed)."""
    kept, sup = [], []
    for raw in block.split("\n"):
        line = raw[2:] if raw.startswith("- ") else raw
        if not line.strip():
            continue
        if am._is_plan_marker(line):
            kept.append(raw)
            continue
        new = am._dedup_words(line)
        if not new:
            kept.append(raw)
            continue
        if any((lambda o: bool(o) and len(new & o) / len(new | o) >= am.THOUGHT_DEDUP_JACCARD)
               (am._dedup_words(k[2:] if k.startswith("- ") else k)) for k in kept):
            sup.append(raw)
        else:
            kept.append(raw)
    return kept, sup


def sample_utterances(n):
    utts = []
    for f in CALLS:
        for l in open(f, errors="ignore"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("caller") != "generate_one_utterance":
                continue
            try:
                ts = float(r.get("ts") or 0)
            except Exception:
                continue
            if LO <= ts <= HI:
                utts.append(r)
    return rng.sample(utts, min(n, len(utts)))


def ask(rec, prompt):
    body = {"model": "qwen2b-ga-chat-v2:latest", "stream": False, "think": False,
            "messages": [{"role": "system", "content": rec.get("system") or ""},
                         {"role": "user", "content": prompt}],
            "options": {"num_ctx": 8192, "temperature": 0.3, "num_predict": 256}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return {"text": r.get("message", {}).get("content", ""),
            "prompt_tok": r.get("prompt_eval_count", 0),
            "out_tok": r.get("eval_count", 0), "wall": round(time.time() - t0, 1)}


def main():
    recs = sample_utterances(N)
    print(f"sampled {len(recs)} n76 utterance prompts")
    # ---- Part A: gate replay over every sampled memory block ----
    tot_lines = tot_sup = 0
    examples = []
    gated_recs = []
    for r in recs:
        p = r["prompt"]
        m = MEM_RE.search(p)
        if not m:
            continue
        kept, sup = gate_lines(m.group(2))
        tot_lines += len(kept) + len(sup)
        tot_sup += len(sup)
        if sup and len(examples) < 6:
            examples.append((r.get("persona"), sup[0][:100]))
        gated_recs.append((r, m, kept, sup))
    print(f"\n[A] gate replay: {tot_lines:,} memory lines -> {tot_lines - tot_sup:,} kept "
          f"({tot_sup / max(1, tot_lines) * 100:.0f}% suppressed)")
    print("[A] sample suppressed lines:")
    for who, s in examples:
        print(f"    {who}: {s}")
    # ---- Part B: model A/B on the gated prompts ----
    sims, fmt = [], [0, 0]
    tok_old = tok_new = 0
    shown = 0
    for r, m, kept, sup in gated_recs:
        if not sup:
            continue
        gated_prompt = r["prompt"][:m.start(2)] + "\n".join(kept) + r["prompt"][m.end(2):]
        o = ask(r, r["prompt"])
        g = ask(r, gated_prompt)
        tok_old += o["prompt_tok"]
        tok_new += g["prompt_tok"]
        sims.append(difflib.SequenceMatcher(None, o["text"], g["text"]).ratio())
        for i, res in enumerate((o, g)):
            if res["text"].lstrip().lower().startswith("utterance"):
                fmt[i] += 1
        if shown < 3:
            shown += 1
            print(f"\n[B] {r.get('persona')} | {o['prompt_tok']} -> {g['prompt_tok']} prompt tok, "
                  f"sim {sims[-1]:.2f}")
            print(f"    OLD: {o['text'][:110]!r}")
            print(f"    GATE: {g['text'][:110]!r}")
    n = max(1, len(sims))
    print(f"\n[B] pairs: {len(sims)} | mean output similarity: {sum(sims)/n:.3f} "
          f"| min: {min(sims):.2f}")
    print(f"[B] format 'utterance:': old {fmt[0]}/{len(sims)}, gated {fmt[1]}/{len(sims)}")
    print(f"[B] prompt tokens: {tok_old:,} -> {tok_new:,} "
          f"({(1 - tok_new/max(1,tok_old))*100:.0f}% cut)")


if __name__ == "__main__":
    main()
