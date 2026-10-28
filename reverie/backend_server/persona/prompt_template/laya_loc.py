"""Laya 'loc' task: area -> room -> object for an activity, three typed `choice`
calls over the options the LLM prompt itself lists. Shared by the v6 training
builder (pipelines/laya_plan/loc_build.py) and the live sim, so the model sees
exactly what it was trained on. Modes (GA_LOC_LAYA): off | shadow | live."""
import collections, json, os, re, threading, time

LEVEL_INS = {"area": "Which area does this person go to for the activity? Stay in the current area if the activity can be done there.",
             "room": "Which room of the chosen area does this person use for the activity?",
             "obj": "Which object in the chosen room does this person use for the activity?"}
_THR_FILE = os.path.join(os.path.dirname(__file__), "loc_thresholds.json")


def parse_prompt(p):
    m = re.search(r"Places .*? can go[^\n]*\n(.*?)\n\n\*", p, re.S)
    if not m:
        return None
    tree = collections.OrderedDict()
    for line in m.group(1).split("\n"):
        mm = re.match(r"- (.+?): (.*)$", line)
        if not mm:
            continue
        rooms = collections.OrderedDict()
        for part in mm.group(2).split("; "):
            r = re.match(r"(.+?)(?: \[(.*)\])?$", part.strip())
            if r:
                rooms[r.group(1)] = [o.strip() for o in (r.group(2) or "").split(", ") if o.strip()]
        tree[mm.group(1)] = rooms
    head = p[:p.index("Places ")]
    act = re.search(r"Activity: (.*)", head)
    return {"head": head.strip(), "activity": act.group(1).strip() if act else "", "tree": tree}


def build_level(who, parsed, level, area=None, room=None):
    """(state, questions, options) for one level, or None when there is nothing to choose."""
    t = parsed["tree"]
    extra = ""
    if level == "area":
        opts = list(t)
        desc = {a: (a + ": " + ", ".join(t[a]))[:110] for a in opts}
    elif level == "room":
        opts = list(t[area])
        desc = {x: (x + ": " + ", ".join(t[area][x]))[:110] for x in opts}
        extra = f"\nChosen area: {area}"
    else:
        opts = list(t[area][room])
        desc = {o: o for o in opts}
        extra = f"\nChosen area: {area}; room: {room}"
    if len(opts) < 2:
        return None
    state = {"who": who, "situation": (parsed["head"] + extra)[-1300:], "level": level}
    return state, {"loc": {"type": "choice", "instructions": LEVEL_INS[level], "criteria": desc}}, opts


def ask_level(state, questions, opts):
    """{option: prob} from laya, or None."""
    from persona.prompt_template.laya_scoring import _systemone
    try:
        a = (_systemone(state, questions) or {}).get("loc") or {}
    except Exception:
        return None
    leg = a.get("legend") or {}
    crit = questions["loc"]["criteria"]
    pr = {}
    for k, v in (a.get("probabilities") or {}).items():
        if str(k).startswith("__"):
            continue
        lab = str(leg.get(k, k))
        for o in opts:
            if lab in (o, crit[o]):
                pr[o] = pr.get(o, 0.0) + float(v)
    t = sum(pr.values())
    return {k: v / t for k, v in pr.items()} if t > 0 else None


def thresholds():
    d = {"area": 0.0, "room": 0.0, "obj": 0.0}
    try:
        d.update(json.load(open(_THR_FILE)))
    except Exception:
        pass
    for k in d:
        if os.environ.get(f"GA_LOC_THR_{k.upper()}") is not None:
            d[k] = float(os.environ[f"GA_LOC_THR_{k.upper()}"])
    return d


def decide(prompt, who, gate=True):
    """{"sector","arena","obj","margins":{level: top1-top2}} or None (fall back to the LLM).
    With gate=True a level whose margin is under its threshold aborts the decision."""
    parsed = parse_prompt(prompt)
    if not parsed or not parsed["tree"]:
        return None
    thr = thresholds()
    margins, sel = {}, {}
    area = room = None
    for level in ("area", "room", "obj"):
        b = build_level(who, parsed, level, area, room)
        if b is None:
            only = list(parsed["tree"][area][room]) if level == "obj" else (
                list(parsed["tree"][area]) if level == "room" else list(parsed["tree"]))
            pick = only[0] if only else None
            margins[level] = 1.0
        else:
            pr = ask_level(*b)
            if not pr:
                return None
            ranked = sorted(pr.items(), key=lambda kv: -kv[1])
            pick = ranked[0][0]
            margins[level] = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            if gate and margins[level] < thr[level]:
                return None
        sel[level] = pick
        if level == "area":
            area = pick
        elif level == "room":
            room = pick
    obj = sel["obj"] or "<random>"
    return {"sector": area, "arena": room, "obj": obj, "margins": margins}


def shadow(prompt, who, llm_triple):
    """Off-thread: laya's ungated answer next to the LLM's -> ga_traces/loc_shadow.jsonl."""
    def run():
        try:
            r = decide(prompt, who, gate=False)
            if not r:
                return
            d = os.environ.get("GA_TRACE_DIR") or os.path.join(os.path.dirname(__file__), "../../../../ga_traces")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "loc_shadow.jsonl"), "a") as f:
                f.write(json.dumps({"ts": time.time(), "who": who, "llm": list(llm_triple),
                                    "laya": [r["sector"], r["arena"], r["obj"]], "margins": r["margins"]}) + "\n")
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()
