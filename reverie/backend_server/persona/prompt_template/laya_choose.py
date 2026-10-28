"""Laya 'choose' task: comply with the schedule, take a pull, or do something
of one's own. The state/question builder is shared by the v5 training-set
builder (pipelines/laya_choose_build.py) and the live sim, so the model sees
exactly what it was trained on. Labels: glm-5.3-flash votes on the same
context."""
import re

INSTRUCTIONS = ("Which does this person do for the next while: follow their plan, "
                "take one of the offered pulls, or do something else of their own?")
CONTEXT_CHARS = 1500


def parse_prompt(prompt, persona):
    """(state, questions, keys) from a free-choice prompt string, or None."""
    head, sep, rest = prompt.partition("Options for the next while:\n")
    if not sep or "\nX. " not in rest:
        return None
    lines = [l for l in rest.split("\nX. ")[0].split("\n") if l.strip()]
    crit = {}
    for l in lines:
        k, _, txt = l.partition(". ")
        if not k.isdigit():
            return None
        crit["plan" if k == "0" else f"pull{k}"] = txt.strip()
    crit["own"] = "Do something else of their own choosing instead of the plan"
    state = {"who": persona, "situation": head.strip()[-CONTEXT_CHARS:]}
    questions = {"choose": {"type": "choice", "instructions": INSTRUCTIONS,
                            "criteria": crit}}
    return state, questions, list(crit)


def vote_target(votes, keys, conf=0.85):
    """Soft target over keys from teacher votes (0 | pull index | 'own')."""
    k = len(keys)
    tgt = [0.0] * k
    for v in votes:
        key = "own" if v == "own" else ("plan" if v == 0 else f"pull{v}")
        if key not in keys:
            continue
        d = [(1 - conf) / (k - 1)] * k
        d[keys.index(key)] = conf
        tgt = [a + b for a, b in zip(tgt, d)]
    n = sum(tgt)
    return [x / n for x in tgt] if n else None


def decide(prompt, persona):
    """Laya's call on a free-choice prompt. Returns None (caller falls back to
    the LLM) or {"choice": 0 | k | "own", "p_dev": float, "probs": {...}}.
    The most likely option wins (no target rebellion rate)."""
    import os
    pr = parse_prompt(prompt, persona)
    if not pr:
        return None
    state, qs, keys = pr
    try:
        from persona.prompt_template.laya_scoring import _systemone
        ans = (_systemone(state, qs) or {}).get("choose") or {}
    except Exception:
        return None
    legend = ans.get("legend") or {}
    crit = qs["choose"]["criteria"]
    probs = {}
    for k, v in (ans.get("probabilities") or {}).items():
        if str(k).startswith("__"):
            continue
        lab = str(legend.get(k, k))
        for key, txt in crit.items():
            if lab == key or lab == txt:
                probs[key] = probs.get(key, 0.0) + float(v)
    tot = sum(probs.values())
    if tot <= 0 or "plan" not in probs:
        return None
    probs = {k: v / tot for k, v in probs.items()}
    p_dev = 1.0 - probs["plan"]
    # No fixed rebellion rate: laya's own most likely option wins. An optional
    # GA_CHOICE_LAYA_THR (deviate only when P(deviate) >= thr) stays as a dial.
    thr = os.environ.get("GA_CHOICE_LAYA_THR")
    if thr is not None and p_dev < float(thr):
        return {"choice": 0, "p_dev": p_dev, "probs": probs}
    best = max(probs, key=probs.get)
    if best == "plan":
        return {"choice": 0, "p_dev": p_dev, "probs": probs}
    choice = "own" if best == "own" else int(best[4:])
    return {"choice": choice, "p_dev": p_dev, "probs": probs}
