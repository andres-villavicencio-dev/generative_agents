#!/usr/bin/env python3
"""
flag_decomp_leak.py — flag task decompositions contaminated by the v3
few-shot example, so they can be excluded from sLM fine-tuning sets.

task_decomp_v3.txt shipped a single worked example (Kelly planning a
kindergarten lesson). minimax copied it into ~27% of decompositions of
UNRELATED activities ("eat breakfast" -> curriculum / worksheet steps).
task_decomp_v4.txt fixes the prompt; this script marks what v3 already
produced in ga_traces/calls/*.jsonl.

A record is flagged when its activity is not teaching/study-related but
the answer uses lesson-plan vocabulary. Output: one JSON line per flagged
record (ts, persona, activity, matched terms) in
ga_traces/calls/flags/task_decomp_example_leak.jsonl — join on (ts, persona).

Usage:
    python pipelines/flag_decomp_leak.py
"""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALLS = os.path.join(REPO, "ga_traces", "calls")
OUT = os.path.join(CALLS, "flags", "task_decomp_example_leak.jsonl")

LEAK = re.compile(r"\b(lesson|kindergarten|curriculum|worksheet|printable|"
                  r"differentiation|learning[- ]center|students?)\b", re.I)
TEACH = re.compile(r"\b(teach|teaching|lesson|class|lecture|tutor|school|"
                   r"students?|course|professor|study|studying|homework)\b", re.I)
ACTIVITY = re.compile(r"complete the following activity: (.*?) from ")


def main():
    flagged, total = [], 0
    for path in sorted(glob.glob(os.path.join(CALLS, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("caller") != "run_gpt_prompt_task_decomp":
                    continue
                content = r.get("content") or ""
                acts = ACTIVITY.findall(r.get("prompt") or "")
                if not content.strip() or not acts:
                    continue
                total += 1
                # v4 prompts carry the format-only example; never flag them
                if "FORMAT EXAMPLE ONLY" in (r.get("prompt") or ""):
                    continue
                act = acts[-1]
                hits = LEAK.findall(content)
                if hits and not TEACH.search(act):
                    flagged.append({"ts": r.get("ts"), "persona": r.get("persona"),
                                    "activity": act,
                                    "terms": sorted({h.lower() for h in hits}),
                                    "source": os.path.basename(path)})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for rec in flagged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"flagged {len(flagged)}/{total} task decompositions -> {OUT}")


if __name__ == "__main__":
    main()
