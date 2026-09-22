"""Diagnose: does the REAL legacy decide_to_talk path ever pass validation?

Uses the byte-exact real prompt (generate_prompt + v2 template) and the
verbatim real validate/clean, then prints raw model output and whether
validation passes. If the fail-safe fires, decide_to_talk is a constant
"yes" in the live sim — a real bug.
"""
import os
import sys
import json

REVERIE_DIR = "/home/andus/Projects/ga-jev/reverie/backend_server"
sys.path.insert(0, REVERIE_DIR)
os.chdir(REVERIE_DIR)

import importlib
gpt_structure = importlib.import_module("persona.prompt_template.gpt_structure")
generate_prompt = gpt_structure.generate_prompt
safe_generate_response = gpt_structure.safe_generate_response

# Same scenario as bench case 0: Maria -> Klaus, both en route
prompt_input = [
    "Maria Lopez was on the way to Hobbs Cafe to study physics. Klaus Mueller was working on his research paper.",  # context
    "February 14, 2023, 08:00:00 AM",       # curr_time
    "Maria Lopez",                          # init name
    "Klaus Mueller",                        # target name
    "February 13, 2023, 18:13:40",          # last chatted time
    "conversing about meeting at Hobbs Cafe",  # last chat about
    "Maria Lopez is on the way to Hobbs Cafe to study physics",   # init desc
    "Klaus Mueller is on the way to the library to work on his research paper",  # target desc
    "Maria Lopez", "Klaus Mueller",
]
prompt = generate_prompt(prompt_input, "persona/prompt_template/v2/decide_to_talk_v2.txt")
print("=== REAL PROMPT (tail) ===")
print(prompt[-400:])
print()

# Verbatim from run_gpt_prompt_decide_to_talk
def real_validate(gpt_response, prompt=""):
    try:
        if gpt_response.split("Answer in yes or no:")[-1].strip().lower() in ["yes", "no"]:
            return True
        return False
    except Exception:
        return False

def real_clean(gpt_response, prompt=""):
    return gpt_response.split("Answer in yes or no:")[-1].strip().lower()

gpt_param = {"engine": "text-davinci-003", "max_tokens": 20,
             "temperature": 0, "top_p": 1, "stream": False,
             "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

# Grab the RAW response first (bypassing validation) to see what the model writes
raw = gpt_structure._ollama_generate(prompt)
print("=== RAW MODEL OUTPUT ===")
print(repr(raw))
print()
print("=== REAL VALIDATION ===")
print("passes:", real_validate(raw))
print("cleaned:", repr(real_clean(raw)))
print()

# Now the full legacy path with retries, to see the end result
import time
t0 = time.time()
out = safe_generate_response(prompt, gpt_param, 5, "yes", real_validate, real_clean)
print("=== LEGACY PATH RESULT ===")
print(f"output: {out!r} in {time.time()-t0:.2f}s (fail-safe is 'yes')")
print()
print("VERDICT:", "validation passes normally" if real_validate(raw)
      else "BROKEN: real sim runs on fail-safe 'yes' -> agents chat on EVERY encounter")