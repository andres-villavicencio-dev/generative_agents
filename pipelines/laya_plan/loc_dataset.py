#!/usr/bin/env python3
"""Parse action_location LLM calls into hierarchical laya examples.
Each call -> up to 3 'choice' items: loc_area, loc_room (given area), loc_obj (given area+room).
Label = the LLM's answer (validated against the options in the prompt)."""
import collections, glob, hashlib, json, re, sys
REPO = "/home/andus/Projects/generative_agents"

def parse_prompt(p):
    m = re.search(r"Places .*? can go[^\n]*\n(.*?)\n\n\*", p, re.S)
    if not m:
        return None
    tree = collections.OrderedDict()
    for line in m.group(1).split("\n"):
        mm = re.match(r"- (.+?): (.*)$", line)
        if not mm:
            continue
        sec, rest = mm.group(1), mm.group(2)
        rooms = collections.OrderedDict()
        for part in rest.split("; "):
            r = re.match(r"(.+?)(?: \[(.*)\])?$", part.strip())
            if r:
                rooms[r.group(1)] = [o.strip() for o in (r.group(2) or "").split(", ") if o.strip()]
        tree[sec] = rooms
    head = p[:p.index("Places ")]
    act = re.search(r"Activity: (.*)", head)
    return {"head": head.strip(), "activity": act.group(1).strip() if act else "", "tree": tree}

def parse_answer(a):
    m = re.search(r"output:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$", (a or "").strip().split("\n")[0])
    return tuple(x.strip() for x in m.groups()) if m else None

def load(files=None, limit=None):
    files = files or sorted(glob.glob(f"{REPO}/ga_traces/calls/qwen2b-ga-chat-v2_latest*.jsonl"))
    out, seen = [], set()
    for f in files:
        for l in open(f, errors="ignore"):
            if "run_gpt_prompt_action_location" not in l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get("caller") != "run_gpt_prompt_action_location":
                continue
            pr = parse_prompt(d["prompt"]); ans = parse_answer(d.get("content"))
            if not pr or not ans:
                continue
            k = hashlib.md5((d["persona"] + d["prompt"] + d["content"]).encode()).hexdigest()
            if k in seen:
                continue
            seen.add(k)
            sec, room, obj = ans
            t = pr["tree"]
            valid = sec in t and room in t[sec] and (obj in t[sec][room] or not t[sec][room])
            out.append({"persona": d["persona"], "t": float(d["ts"]), "prompt": d["prompt"], "parsed": pr,
                        "ans": ans, "valid": valid})
    return out

if __name__ == "__main__":
    rows = load()
    v = [r for r in rows if r["valid"]]
    print("distinct calls", len(rows), "valid", len(v))
    # self-consistency ceiling: same persona+prompt asked multiple times
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["persona"], r["prompt"])].append(r["ans"])
    rep = [a for a in g.values() if len(a) >= 2]
    maj = sum(collections.Counter(a).most_common(1)[0][1] for a in rep) / max(1, sum(len(a) for a in rep))
    print("repeat groups", len(rep), "majority-agreement ceiling", round(maj, 3))
    sizes = [len(r["parsed"]["tree"]) for r in v]
    print("area options mean", sum(sizes) / len(sizes), "max", max(sizes))
    stay = sum(1 for r in v if r["ans"][0] in r["parsed"]["head"].split("currently in ")[-1][:60])
    print("stay-in-current-area share", round(stay / len(v), 3))
