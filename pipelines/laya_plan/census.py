#!/usr/bin/env python3
"""Phase 0: which LLM calls could laya take over? Per caller: call count, LLM time share,
output size, answer-space shape and self-consistency (identical prompts -> same answer?).
Usage: census.py [part_glob ...] -> pipelines/laya_plan/census.{json,md}"""
import collections, glob, hashlib, json, re, sys, statistics as S
REPO = "/home/andus/Projects/generative_agents"
pats = sys.argv[1:] or ["qwen2b-ga-chat-v2_latest.part02[6-9].jsonl", "qwen2b-ga-chat-v2_latest.part03*.jsonl",
                        "qwen2b-ga-chat-v2_latest.jsonl"]
files = sorted({f for p in pats for f in glob.glob(f"{REPO}/ga_traces/calls/{p}")})
st = collections.defaultdict(lambda: {"n": 0, "lat": 0.0, "out": [], "inp": [], "short": 0, "numeric": 0,
                                      "yesno": 0, "answers": collections.Counter(), "by_prompt": collections.defaultdict(list)})
for f in files:
    for l in open(f, errors="ignore"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        c = str(d.get("caller") or "?")
        s = st[c]; s["n"] += 1
        try:
            s["lat"] += float(d.get("latency_s") or 0)
            s["out"].append(float(d.get("eval_count") or 0)); s["inp"].append(float(d.get("prompt_eval_count") or 0))
        except Exception:
            pass
        a = (d.get("content") or "").strip()
        if not a:
            continue
        s["short"] += len(a) <= 60
        s["numeric"] += bool(re.fullmatch(r"[\W\d.]*\d[\W\d.]*", a)) and len(a) < 40
        s["yesno"] += a.lower().strip(" .!\n") in ("yes", "no", "yes.", "no.")
        if len(s["answers"]) < 3000:
            s["answers"][a[:80]] += 1
        h = hashlib.md5((d.get("prompt") or "").encode()).hexdigest()
        if len(s["by_prompt"][h]) < 6:
            s["by_prompt"][h].append(a[:200])
T = sum(s["lat"] for s in st.values()); N = sum(s["n"] for s in st.values())
rows = []
for c, s in st.items():
    rep = [v for v in s["by_prompt"].values() if len(v) >= 2]
    cons = (sum(1 for v in rep if len(set(v)) == 1) / len(rep)) if rep else None
    rows.append({"caller": c, "calls": s["n"], "call_share": s["n"] / N, "time_share": s["lat"] / T,
                 "mean_s": s["lat"] / max(1, s["n"]), "out_tok": S.mean(s["out"]) if s["out"] else 0,
                 "in_tok": S.mean(s["inp"]) if s["inp"] else 0,
                 "short_frac": s["short"] / s["n"], "numeric_frac": s["numeric"] / s["n"], "yesno_frac": s["yesno"] / s["n"],
                 "distinct_answers": len(s["answers"]), "repeat_prompts": len(rep), "self_consistency": cons})
rows.sort(key=lambda r: -r["time_share"])
json.dump({"files": len(files), "calls": N, "llm_minutes": T / 60, "rows": rows}, open(f"{REPO}/pipelines/laya_plan/census.json", "w"), indent=1)
with open(f"{REPO}/pipelines/laya_plan/census.md", "w") as f:
    f.write(f"# LLM call census ({N} calls, {T/60:.0f} LLM-minutes, {len(files)} trace files)\n\n")
    f.write("| caller | calls | time % | mean s | out tok | short | numeric | self-consist | distinct |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in rows[:30]:
        sc = "-" if r["self_consistency"] is None else f"{r['self_consistency']:.2f}"
        f.write(f"| {r['caller'][:46]} | {r['calls']} | {100*r['time_share']:.1f} | {r['mean_s']:.1f} | {r['out_tok']:.0f} | "
                f"{r['short_frac']:.2f} | {r['numeric_frac']:.2f} | {sc} | {r['distinct_answers']} |\n")
print(open(f"{REPO}/pipelines/laya_plan/census.md").read())
