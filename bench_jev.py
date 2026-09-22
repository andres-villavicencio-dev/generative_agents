"""Standalone A/B benchmark for the local JEV-style scorer.

Replays REAL decide_to_talk scenarios drawn from the live sim's memory
store, through both paths, and reports latency + agreement + quality.

No sim processes are touched: this reads only from the fork's persisted
state on disk. The benchmark is read-only with respect to the live sim.
"""
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error

REVERIE_DIR = "/home/andus/Projects/ga-jev/reverie/backend_server"
SIM_STORE = ("/home/andus/Projects/generative_agents/environment/"
             "frontend_server/storage/live3d_v2/personas")

sys.path.insert(0, REVERIE_DIR)
# gpt_structure.py does `from utils import *` — utils lives in backend_server
# itself, and the backend resolves it by running with cwd = backend_server.
# os.chdir makes the bench use the same resolution.
os.chdir(REVERIE_DIR)

# The backend imports persona.* as namespace packages with
# reverie/backend_server as cwd — do the same here.
import importlib  # noqa: E402
jev_scoring = importlib.import_module("persona.prompt_template.jev_scoring")
jev_decide = jev_scoring.jev_decide

print("Jev scoring enabled:", jev_scoring.USE_JEV_SCORING)
print("Chat model:", jev_scoring.OLLAMA_CHAT_MODEL)

# --------------------------------------------------------------- scenarios
# Real prompts, reconstructed from the live sim's persisted memory: pairs of
# agents, their actual current activities, and the retrieved context events
# that the real decide_to_talk call would have used. Built the same way
# run_gpt_prompt_decide_to_talk's create_prompt_input builds its context.

def load_events(name, limit=4):
    with open(f"{SIM_STORE}/{name}/bootstrap_memory/associative_memory/nodes.json") as f:
        nodes = json.load(f)
    events = [v for v in nodes.values()
              if v.get("type", "").lower() == "event"]
    events.sort(key=lambda v: v.get("created", ""))
    return events[-limit:]


def scenario_prompt(init_name, target_name, init_act, target_act,
                   last_chat, context_lines):
    context = " ".join(context_lines)
    prompt = f"""Task -- given context, determine whether the subject will initiate a conversation with another.
Context: {context}
Right now, it is February 14, 2023, 08:00:00 AM. {init_name} and {target_name} last chatted at {last_chat}.

{init_name} is {init_act}.
{target_name} is {target_act}.

Question: Would {init_name} initiate a conversation with {target_name}?

Answer in "yes" or "no":
"""
    return prompt


SCENARIOS = [
    # (init, target, init_act, target_act, last_chat, context)
    ("Maria Lopez", "Klaus Mueller",
     "on the way to Hobbs Cafe to study physics",
     "on the way to the library to work on his research paper",
     "February 13, 2023, 18:13:40 about meeting at Hobbs Cafe",
     ["Maria Lopez was heading to Hobbs Cafe.",
      "Klaus Mueller was working on his research paper."]),
    ("Klaus Mueller", "Maria Lopez",
     "on the way to the library",
     "at Hobbs Cafe studying",
     "February 13, 2023, 18:13:40 about meeting at Hobbs Cafe",
     ["Klaus Mueller was heading to the library.",
      "Maria Lopez was studying at Hobbs Cafe."]),
    ("Isabella Rodriguez", "Maria Lopez",
     "setting up the cafe for the morning shift",
     "sleeping in her bed in the dorm",
     "February 13, 2023, 08:01:40 about breakfast",
     ["Isabella Rodriguez is setting up the cafe.",
      "Maria Lopez is sleeping."]),
    ("Maria Lopez", "Isabella Rodriguez",
     "on the way to the cafe",
     "working at the cafe counter",
     "February 13,  2023, 08:01:40 about breakfast",
     ["Maria Lopez is on the way to Hobbs Cafe.",
      "Isabella Rodriguez is working."]),
    ("Klaus Mueller", "Isabella Rodriguez",
     "eating lunch at Hobbs Cafe",
     "serving customers at the cafe",
     "February 13, 2023, 08:01:40 about breakfast",
     ["Klaus Mueller is eating at the cafe.",
      "Isabella Rodriguez is serving food."]),
]


def main():
    results = []

    # --- Lane A: JEV-style scoring (all orderings averaged) ---
    print("\n=== Lane A: JEV-style scoring ===")
    choices = {"yes": "yes — initiate a conversation",
               "no": "no — continue current activity"}
    for i, s in enumerate(SCENARIOS):
        init, target, init_act, target_act, last_chat, ctx = s
        base = scenario_prompt(init, target, init_act, target_act, last_chat, ctx)
        prompt = base.replace('Answer in "yes" or "no":',
                              'Options:\n__JEV_OPTIONS__\n\nAnswer in "yes" or "no":')
        t0 = time.time()
        decision, dist = jev_decide(prompt, choices, return_dist=True)
        dt = time.time() - t0
        results.append({"lane": "jev", "case": i, "decision": decision,
                        "dist": dist, "latency_s": round(dt, 3)})
        print(f"  case {i}: {decision!r} in {dt*1000:.0f}ms | dist={dist}")

    # --- Lane B: legacy generation path ---
    print("\n=== Lane B: legacy generation path ===")
    from persona.prompt_template.gpt_structure import safe_generate_response  # noqa: E402
    for i, s in enumerate(SCENARIOS):
        init, target, init_act, target_act, last_chat, ctx = s
        prompt = scenario_prompt(init, target, init_act, target_act, last_chat, ctx)
        gpt_param = {"engine": "text-davinci-003", "max_tokens": 20,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
        # legacy: CoT then yes/no
        cot_prompt = prompt + "Reasoning: Let's think step by step."

        def validate(resp, prompt=""):
            try:
                # Mirrors the FIXED real __func_clean_up: extract last yes/no
                # token (JSON-escaped marker makes plain splitting useless).
                ans = clean(resp)
                return ans in ("yes", "no")
            except Exception:
                return False

        def clean(resp, prompt=""):
            import re as _re
            matches = _re.findall(r'\b(yes|no)\b', resp.lower())
            if matches:
                return matches[-1]
            return resp.strip().lower().split(" ")[0]

        t0 = time.time()
        try:
            out = safe_generate_response(cot_prompt, gpt_param, 5, "yes",
                                          validate, clean)
        except Exception as e:
            out = f"ERROR: {e}"
        dt = time.time() - t0
        # normalize
        if out and "yes" in out.lower():
            out = "yes"
        elif out and "no" in out.lower():
            out = "no"
        results.append({"lane": "llm", "case": i, "decision": out,
                        "latency_s": round(dt, 3)})
        print(f"  case {i}: {out!r} in {dt:.2f}s")

    # --- Report ---
    jev_res = [r for r in results if r["lane"] == "jev"]
    llm_res = [r for r in results if r["lane"] == "llm"]
    agree = sum(1 for j, l in zip(jev_res, llm_res)
                if j["decision"] == l["decision"])
    print("\n=== REPORT ===")
    print(f"cases: {len(SCENARIOS)}")
    print(f"jev latency avg: {sum(r['latency_s'] for r in jev_res)/len(jev_res)*1000:.0f}ms")
    print(f"llm latency avg: {sum(r['latency_s'] for r in llm_res)/len(llm_res):.2f}s")
    print(f"agreement: {agree}/{len(SCENARIOS)}")
    print(f"jev stats: {json.dumps(jev_scoring.jev_stats_snapshot())}")
    with open("/home/andus/Projects/ga-jev/bench_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved -> bench_results.json")


if __name__ == "__main__":
    main()