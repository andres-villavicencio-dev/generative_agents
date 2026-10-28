#!/usr/bin/env python3
"""Same 80 contexts, three prompt tails, one model: how does wording move the
deviation rate? Usage: choice_prompt_variants.py [model] [N]"""
import re, sys, json, collections
sys.path.insert(0, "/home/andus/Projects/generative_agents/pipelines")
from choice_model_replay import load, parse, ask, FILLER

FMT = ("choice: <0, a number, or X>\nactivity: <only if X: a specific action>\n"
       "minutes: <15 to 120>\nwhy: <one short sentence>")
TAILS = {
 "tight(current)": None,
 "neutral": ("Think about what {f} would actually do right now, given {f}'s state, "
             "money, relationships and these options. The plan is what {f} "
             "intended this morning; people also change course. There is no "
             "right answer. Answer in this exact format:\n" + FMT),
 "neutral+honest": ("Decide as {f} really would, not as an assistant would. {f} "
             "has moods, habits, obligations and impulses; sometimes a plan "
             "wins, sometimes something else does. Answer in this exact "
             "format:\n" + FMT),
}

def retail(p, name, tail):
    first = re.search(r"\n(\w+) planned this morning", p).group(1)
    i = p.index("Default to option 0")
    return p[:i] + tail.format(f=first)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "glm-5.3-flash:cloud"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    data = load(n)
    out = {}
    for name, tail in TAILS.items():
        rows = []
        for d in data:
            dd = dict(d)
            if tail:
                dd["prompt"] = retail(d["prompt"], name, tail)
            try:
                txt, _ = ask(model, dd); r = parse(txt)
            except Exception:
                r = None
            rows.append(r)
        c = collections.Counter(r[0] if r else "unparsed" for r in rows)
        own = [r for r in rows if r and r[0] == "own"]
        out[name] = {k: f"{100*v/len(rows):.0f}%" for k, v in c.items()}
        out[name]["filler"] = f"{sum(1 for r in own if FILLER.search(r[1]))}/{len(own)}"
        out[name]["samples"] = [(r[1][:70], r[2][:80]) for r in own[:3]]
        print(name, json.dumps(out[name]), flush=True)
