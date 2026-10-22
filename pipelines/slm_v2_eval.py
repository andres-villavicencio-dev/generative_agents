#!/usr/bin/env python3
"""Held-out eval of chat-lane models via ollama /api/chat (same request shape as the sim).

For each test record: messages=[system,user], think=false, the record's own sampling
temperature (fixed seed), top_k 20, top_p 0.95, presence_penalty 0, num_ctx 8192,
num_predict 1536. Scores: format validity per family (ga_chat_formats.validate), closed-form
agreement with the teacher answer, and saves raw generations for qualitative review.

usage: slm_v2_eval.py <test.jsonl> <out_dir> <ollama_model> [<ollama_model> ...]
env: OLLAMA_URL (default http://127.0.0.1:11434), EVAL_KEEP_ALIVE (default 2m),
     EVAL_NUM_GPU (0 = CPU only), EVAL_NUM_THREAD
"""
import collections
import difflib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_chat_formats import CLOSED_FORM, clean, extract, validate  # noqa: E402

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
KEEP = os.environ.get("EVAL_KEEP_ALIVE", "2m")
TEST, OUT = sys.argv[1], sys.argv[2]
MODELS = sys.argv[3:]


def chat(model, msgs, temp):
    body = {"model": model, "messages": msgs, "stream": False, "think": False, "keep_alive": KEEP,
            "options": {"temperature": temp, "top_k": 20, "top_p": 0.95, "presence_penalty": 0.0,
                        "num_ctx": 8192, "num_predict": 1536, "seed": 1234}}
    if os.environ.get("EVAL_NUM_GPU") is not None:
        body["options"]["num_gpu"] = int(os.environ["EVAL_NUM_GPU"])
    if os.environ.get("EVAL_NUM_THREAD"):
        body["options"]["num_thread"] = int(os.environ["EVAL_NUM_THREAD"])
    req = urllib.request.Request(URL + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return (d.get("message") or {}).get("content") or "", d.get("done_reason"), d.get("eval_count", 0), time.time() - t0


def agree(fam, a, b):
    if a is None or b is None:
        return None
    if fam in ("run_gpt_prompt_event_poignancy", "run_gpt_prompt_chat_poignancy"):
        return {"exact": a == b, "within1": abs(a - b) <= 1}
    if fam == "run_gpt_prompt_wake_up_hour":
        return {"exact": a == b, "within1": abs(a - b) <= 60}
    if fam == "run_gpt_prompt_action_location":
        return {"exact": a == b, "area": a[:1] == b[:1], "area_room": a[:2] == b[:2]}
    if fam in ("run_gpt_prompt_event_triple", "run_gpt_prompt_act_obj_event_triple", "run_gpt_prompt_act_obj_desc"):
        return {"exact": a == b, "fuzzy": difflib.SequenceMatcher(None, a, b).ratio() >= 0.8}
    return {"exact": a == b}


def main():
    os.makedirs(OUT, exist_ok=True)
    test = [json.loads(l) for l in open(TEST)]
    summary = json.load(open(os.path.join(OUT, "summary.json"))) if os.path.exists(os.path.join(OUT, "summary.json")) else {}
    for model in MODELS:
        tag = model.replace(":", "_").replace("/", "_")
        gens = []
        per = collections.defaultdict(lambda: {"n": 0, "valid": 0, "agree": collections.Counter(), "agree_n": 0,
                                               "length_stop": 0, "tokens": 0, "secs": 0.0})
        for i, r in enumerate(test):
            msgs = r["messages"][:-1]
            teacher = r["messages"][-1]["content"]
            prompt = msgs[-1]["content"]
            temp = r.get("temperature", 0.6)
            try:
                out, dr, n, dt = chat(model, msgs, temp)
            except Exception as e:  # noqa: BLE001
                out, dr, n, dt = "", "error:%s" % type(e).__name__, 0, 0.0
            fam = r["family"]
            ok, why = validate(fam, prompt, out)
            p = per[fam]
            p["n"] += 1
            p["valid"] += int(ok)
            p["length_stop"] += int(dr == "length")
            p["tokens"] += n
            p["secs"] += dt
            ag = None
            if fam in CLOSED_FORM:
                ag = agree(fam, extract(fam, prompt, out), extract(fam, prompt, teacher))
                if ag is not None:
                    p["agree_n"] += 1
                    for k, v in ag.items():
                        p["agree"][k] += int(v)
            gens.append({"family": fam, "valid": ok, "why": why, "agree": ag, "done_reason": dr,
                         "output": out, "teacher": teacher, "prompt_tail": prompt[-600:]})
            if (i + 1) % 50 == 0:
                print(f"{model}: {i+1}/{len(test)}", flush=True)
        with open(os.path.join(OUT, f"gens.{tag}.jsonl"), "w") as f:
            for g in gens:
                f.write(json.dumps(g) + "\n")
        fams = {}
        for fam, p in sorted(per.items()):
            fams[fam] = {"n": p["n"], "valid": p["valid"] / p["n"], "length_stop": p["length_stop"],
                         "tok_per_s": p["tokens"] / max(p["secs"], 1e-6), "avg_tokens": p["tokens"] / p["n"]}
            if p["agree_n"]:
                fams[fam]["agree"] = {k: v / p["agree_n"] for k, v in p["agree"].items()}
                fams[fam]["agree_n"] = p["agree_n"]
        tot = sum(p["n"] for p in per.values())
        summary[model] = {"overall_valid": sum(p["valid"] for p in per.values()) / tot,
                          "macro_valid": sum(f["valid"] for f in fams.values()) / len(fams),
                          "families": fams}
        print(model, "overall valid %.3f macro %.3f" % (summary[model]["overall_valid"], summary[model]["macro_valid"]), flush=True)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
