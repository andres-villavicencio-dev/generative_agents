#!/usr/bin/env python3
"""Numbers for pipelines/model_ab.py: control adaptctl vs treatment boltchat.

Per-call figures come from ga_traces/calls/<model>.jsonl sliced by each arm's
wall-clock window (adapt_ab_runs.json / model_ab_runs.json). Lane callers:
  chat     generate_one_utterance, agent_chat_summarize_relationship,
           memo_on_convo, planning_thought_on_convo, summarize_conversation
  artifact create_artifact, artifact_reaction, generate_meal_description,
           generate_written_content, generate_painting_with_image
Usage: venv/bin/python pipelines/model_ab_metrics.py
"""
import collections, glob, json, os, re, statistics as S, sys

REPO = "/home/andus/Projects/generative_agents"
sys.path.insert(0, f"{REPO}/pipelines")
LANES = {
    "chat": ("generate_one_utterance", "run_gpt_prompt_agent_chat_summarize_relationship",
             "run_gpt_prompt_memo_on_convo", "run_gpt_prompt_planning_thought_on_convo",
             "run_gpt_prompt_summarize_conversation"),
    "artifact": ("run_gpt_prompt_create_artifact", "run_gpt_prompt_artifact_reaction",
                 "generate_meal_description", "generate_written_content",
                 "generate_painting_with_image"),
}


def windows():
    a = json.load(open(f"{REPO}/adapt_ab_runs.json"))
    ctl = next(x for x in a if x["sim"] == "adaptctl")
    exp = json.load(open(f"{REPO}/model_ab_runs.json"))
    return {"control (qwen2b)": ctl, "treatment (bolt)": exp}


def calls(win):
    out = []
    for f in glob.glob(f"{REPO}/ga_traces/calls/*.jsonl"):
        for line in open(f, errors="ignore"):
            if '"ts"' not in line:
                continue
            try:
                d = json.loads(line)
                ts = float(d["ts"])
            except Exception:
                continue
            if win["boot"] <= ts <= win["stop"]:
                out.append(d)
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def ngram_distinct(texts, n=3):
    grams = []
    for t in texts:
        w = re.findall(r"\w+", t.lower())
        grams += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return round(len(set(grams)) / max(len(grams), 1), 3)


def lane_stats(cs, callers):
    rows = [d for d in cs if d.get("caller") in callers]
    if not rows:
        return {"calls": 0}
    lat = [float(d["latency_s"]) for d in rows if d.get("latency_s")]
    empty = sum(1 for d in rows if not (d.get("content") or "").strip())
    overflow = sum(1 for d in rows if d.get("done_reason") == "length")
    retried = sum(1 for d in rows if str(d.get("attempt", "0")) not in ("0", "None"))
    ev = [float(d["eval_count"]) for d in rows if d.get("eval_count")]
    good = [d["content"] for d in rows if (d.get("content") or "").strip()]
    return {"calls": len(rows), "lat_p50_s": round(pct(lat, .5), 2), "lat_p90_s": round(pct(lat, .9), 2),
            "total_lat_min": round(sum(lat) / 60, 1), "empty_rate": round(empty / len(rows), 3),
            "overflow_rate": round(overflow / len(rows), 3), "retry_rate": round(retried / len(rows), 3),
            "mean_eval_tok": round(S.mean(ev), 1) if ev else None,
            "mean_chars": round(S.mean(len(g) for g in good), 0) if good else None,
            "distinct_3gram": ngram_distinct(good)}


def sim_stats(sim):
    from adapt_metrics import chats, steps
    share, longest = chats(sim, 17280)
    fr = steps(sim, 17280)
    return {"steps": len(fr), "chat_share": round(share, 3), "chat_longest_min": longest}


if __name__ == "__main__":
    w = windows()
    res = {}
    for label, win in w.items():
        cs = calls(win)
        wall = (win["stop"] - win["boot"] - win.get("paused_s", 0)) / 60
        r = {"wall_min": round(wall, 1), "all_calls": len(cs)}
        r.update(sim_stats(win["sim"]))
        r["steps_per_min"] = round(r["steps"] / wall, 1)
        for lane, callers in LANES.items():
            r[lane] = lane_stats(cs, callers)
        res[label] = r
    print(json.dumps(res, indent=1))
