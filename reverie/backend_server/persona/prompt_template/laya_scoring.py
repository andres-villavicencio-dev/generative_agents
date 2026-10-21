"""
Laya v3 scoring client for generative_agents (production poignancy, Sep 2026).

Same interception + wire pattern as rizzo_scoring.py: ga-laya.service serves
the fine-tuned laya checkpoint on 127.0.0.1:8018 (/v1/systemone, identical
request/response shape). v3 was distilled from cloud Jev (typesafe/jev-1.13
harvested live in-sim, n42+n43) + 85 gold-anchored hard negatives.

Eval (85-event blind gold): laya-v3 MAE 0.53 / within±1 87% / 44ms vs
rizzo MAE 0.85 / within±1 75% / ~130ms. Promoted ahead of rizzo in the
poignancy cascade; rizzo remains the fail-open fallback.
"""
import json
import os
import threading
import time
import urllib.request


def _persona_temperament(who):
    """Layer 3: per-persona temperament biases from town_model_settings.json.
    Returns {'poignancy_shift': 0, 'talk_bias': 0.0, 'patience': 0.0} with
    persona-file defaults for absent keys. Never raises."""
    out = {"poignancy_shift": 0, "talk_bias": 0.0, "patience": 0.0}
    try:
        import json as _json
        _path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "town_model_settings.json")
        with open(_path) as f:
            data = _json.load(f)
        p = data.get(who) or {}
        out["poignancy_shift"] = int(p.get("poignancy_shift", 0))
        out["talk_bias"] = float(p.get("talk_bias", 0.0))
        out["patience"] = float(p.get("patience", 0.0))
    except Exception:
        pass
    return out

def _talk_key_bias(who):
    """Probability-mass bias for talk decisions (2026-10-05): folds the
    GLOBAL multiplier (GA_TALK_YES_BIAS, default 1.0) with the persona's
    temperament talk_bias from town_model_settings.json. Only 'yes' is
    ever lifted; negatives shrink it. Returns {} when neutral.
    Note: react decisions (wait/continue) are untouched — only
    conversation INITIATION gets the sociability bias."""
    try:
        mult = float(os.environ.get("GA_TALK_YES_BIAS", "1.0"))
    except Exception:
        mult = 1.0
    if mult == 1.0:
        return {}
    bias = _persona_temperament(who).get("talk_bias", 0.0)
    mult = mult * (1.0 + bias) if bias else mult
    if mult <= 0.0:
        return {}
    return {"yes": mult}


USE_LAYA = os.environ.get("GA_LAYA_POIGNANCY", "0") == "1"
LAYA_URL = os.environ.get("GA_LAYA_URL", "http://127.0.0.1:8018").rstrip("/")
LAYA_TIMEOUT = float(os.environ.get("GA_LAYA_TIMEOUT", "10"))

_RUBRIC = [
    '1 = purely mundane routine (brushing teeth, making bed, idle objects, napping)',
    '2 = everyday activity with slight personal meaning (a meal, a routine errand)',
    '3 = mildly notable (a pleasant chat, working on a hobby, a minor inconvenience)',
    '4 = a small personal event (a new idea, finishing a small task, meeting someone)',
    '5 = moderately meaningful (creative work progress, an invitation, a small success)',
    '6 = personally significant (an achievement, an important conversation, a discovery)',
    '7 = clearly poignant (a milestone, a meaningful social moment, strong emotion)',
    '8 = highly poignant (creating/publishing a work, a major personal breakthrough)',
    '9 = very poignant (a breakup, a loss, winning an award, a life-changing event)',
    '10 = extremely poignant (the most emotionally significant events in a life)',
]

_stats = {"calls": 0, "answered": 0, "errors": 0, "latency_samples": []}
_lock = threading.Lock()


def laya_stats():
    lat = _stats["latency_samples"]
    out = dict(_stats)
    out.pop("latency_samples")
    if lat:
        out["latency_p50_ms"] = sorted(lat)[len(lat) // 2] * 1000
    return out


def _systemone(state, questions):
    """POST /v1/systemone to ga-laya.service. Returns answers dict or None."""
    body = {"model": "laya-v3", "state": state, "questions": questions}
    req = urllib.request.Request(
        f"{LAYA_URL}/v1/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=LAYA_TIMEOUT) as resp:
        return json.loads(resp.read()).get("answers") or None


def laya_score_poignancy(agent, iss, description):
    """Poignancy 1-10 typed decision from laya v3.
    Returns (score:int, confidence:float) or (None, None) to signal fallback."""
    if os.environ.get("GA_LAYA_POIGNANCY", "0") != "1":
        return None, None
    _iss = (iss or "").strip()
    if len(_iss) > 500:
        _iss = _iss[:500]
    state = {"who": agent, "identity": _iss, "event": description}
    questions = {"poignancy": {
        "type": "score",
        "instructions": ("Rate the likely poignancy of the event for this "
                         "person, considering their identity and circumstances."),
        "criteria": _RUBRIC}}
    with _lock:
        _stats["calls"] += 1
    t0 = time.time()
    try:
        answers = _systemone(state, questions)
    except Exception as e:
        _stats["errors"] += 1
        print(f"[LAYA] poignancy request failed ({type(e).__name__}: {e}); falling back")
        return None, None
    dt = time.time() - t0
    _stats["latency_samples"].append(dt)
    if len(_stats["latency_samples"]) > 200:
        _stats["latency_samples"] = _stats["latency_samples"][-200:]
    answer = (answers or {}).get("poignancy")
    if not isinstance(answer, dict):
        _stats["errors"] += 1
        return None, None
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not str(k).startswith("__")}
    if not probs:
        _stats["errors"] += 1
        return None, None
    best = max(probs, key=probs.get)
    label = str((answer.get("legend") or {}).get(best, best))
    num = label.split("=")[0].strip() if "=" in label else label
    try:
        score = int(float(num))
    except (ValueError, TypeError):
        _stats["errors"] += 1
        return None, None
    if not (1 <= score <= 10):
        _stats["errors"] += 1
        return None, None
    # Layer 3: per-persona poignancy temperament (Carlos amplifies everything;
    # Giorgio the mathematician deflates it). Applied AFTER validity gates.
    score = score + _persona_temperament(agent)["poignancy_shift"]
    score = max(1, min(10, score))
    _stats["answered"] += 1
    _score_conf = float(probs[best])
    try:
        from persona.prompt_template.ga_trace import record
        record("poignancy", agent,
               {"conf": _score_conf, "score": score,
                "state": state,
                "questions": questions,
                "description": (description or "")[:200]})
    except Exception:
        pass
    return score, _score_conf


# ============================================================================
# Reflex layer: react / talk typed decisions (rizzo handover, Sep 30 2026)
# ============================================================================

def _typed_choice(question_key, state_fields, instructions, options,
                  key_bias=None):
    """Shared choice-type decision via laya. Returns (option_key, conf)
    or (None, None) to signal fallback. Never raises.
    key_bias: optional {option_key: multiplier} applied to the resolved
    probability mass before argmax — flips close calls toward the biased
    option without overriding clear decisions (GA_TALK_YES_BIAS, 2026-10-05)."""
    if os.environ.get("GA_LAYA_POIGNANCY", "0") != "1":
        return None, None
    state = {k: v[:600] for k, v in state_fields.items() if isinstance(v, str)}
    questions = {question_key: {
        "type": "choice",
        "instructions": instructions,
        "criteria": options,
    }}
    with _lock:
        _stats["calls"] += 1
    t0 = time.time()
    try:
        answers = _systemone(state, questions)
    except Exception as e:
        _stats["errors"] += 1
        print(f"[LAYA] {question_key} request failed ({type(e).__name__}: {e}); falling back")
        return None, None
    dt = time.time() - t0
    _stats["latency_samples"].append(dt)
    if len(_stats["latency_samples"]) > 200:
        _stats["latency_samples"] = _stats["latency_samples"][-200:]
    answer = (answers or {}).get(question_key)
    if not isinstance(answer, dict):
        _stats["errors"] += 1
        return None, None
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not str(k).startswith("__")}
    legend = answer.get("legend") or {}
    if not probs:
        _stats["errors"] += 1
        return None, None
    # Resolve every option's probability mass through the legend before
    # biasing/argmax (options may be split across raw label variants).
    mass = {}
    for opt in options:
        mass[opt] = sum(v for k, v in probs.items()
                        if str(legend.get(k, k)) == opt)
    if key_bias:
        for bkey, mult in key_bias.items():
            if bkey in mass and mult:
                mass[bkey] = max(0.0, mass[bkey] * mult)
    best = max(mass, key=mass.get)
    key = best
    if key not in options:
        _stats["errors"] += 1
        return None, None
    _stats["answered"] += 1
    return key, float(mass[key] / max(sum(mass.values()), 1e-9))


def laya_decide_react(who, who_act, target, target_act, venue,
                      relation=None, min_conf=None, exclusive=False):
    """decide_to_react as a typed choice: wait vs continue.
    Returns ('1'|'2', conf) matching the legacy answer codes, or (None, None)."""
    # Layer 0 hard gate: a capacity-1 facility in use means "continue" is
    # physically wrong — two people cannot occupy it. Deterministic, no model
    # call. (Both rizzo and laya-v3 fumble this case; the harvest had zero
    # real exclusive collisions to learn from. Sep 30 2026.)
    if exclusive:
        return ("1", 0.99)
    options = {
        "1": (f"Wait: {who} should wait until {target} is done with "
              f"{target_act} before doing {who_act}"
              + (" — the venue is a single-person facility already in use"
                 if exclusive else "")),
        "2": (f"Continue: {who} should continue with {who_act} now, "
              f"even though {target} is doing {target_act} at the same venue"
              + (" — two people cannot use this single-person facility at once"
                 if exclusive else "")),
    }
    instructions = ("Two agents are heading toward the same venue in a small "
                    "town. Should the first agent wait for the second agent to "
                    "finish their activity, or continue with their own plan?")
    state = {
        "who": who, "who_activity": who_act,
        "other": target, "other_activity": target_act,
        "venue": venue, "relationship": relation or "acquaintances in town",
        "facility": ("a single-person facility (only one person can use it at "
                     "a time)" if exclusive else "a shared public space"),
    }
    key, conf = _typed_choice("react", state, instructions, options)
    # Layer 3: per-persona patience — Tom Moreno barges on (conf of "continue"
    # inflated), Jane Moreno waits (conf of "wait" inflated). Applied after the
    # deterministic physics gates; exclusive collisions stay gated above.
    if key is not None:
        bias = _persona_temperament(who)["patience"]
        if bias and key == "2":          # "continue" won — patience<0 helps it
            conf = min(0.99, conf * (1.0 + abs(bias))) if bias < 0 else \
                   max(0.10, conf * (1.0 - bias)) if bias > 0 else conf
    try:
        from persona.prompt_template.ga_trace import record
        record("react", who,
               {"conf": conf, "choice": key,
                "state": state, "options": options,
                "who_act": (who_act or "")[:150],
                "target": target, "target_act": (target_act or "")[:150],
                "venue": (venue or "")[:120]})
    except Exception:
        pass
    return key, conf


def laya_decide_talk(who, who_act, target, target_act, last_chat_about=None,
                     min_conf=None, needs=None):
    """decide_to_talk as a typed yes/no choice. Returns ('yes'|'no', conf)
    or (None, None). Keeps rizzo's deterministic critical-need hard gate:
    a critical non-social need means no socializing — we don't even ask."""
    if needs:
        for k, v in needs.items():
            if v < 20 and k not in ("social", "stimulation", "status"):
                try:
                    from persona.prompt_template.ga_trace import record
                    record("talk", who,
                           {"conf": 0.99, "choice": "no",
                            "state": (state if "state" in dir() else
                                      {"who": who, "critical_need": k}),
                            "questions": {"talk": {"type": "choice",
                                                   "gate": "critical_need"}},
                            "hard_gate": f"critical {k}",
                            "who_act": (who_act or "")[:150],
                            "target": target,
                            "target_act": (target_act or "")[:150]})
                except Exception:
                    pass
                return ("no", 0.99)
    state = {
        "who": who, "who_activity": who_act,
        "other": target, "other_activity": target_act,
        "last_chat": last_chat_about or "no recent conversation",
    }
    if needs:
        parts = []
        s = needs.get("social", 50)
        parts.append("desperately lonely, starved for company" if s < 20
                     else "lonely, wanting company" if s < 50
                     else "socially satisfied" if s > 70
                     else "socially content")
        if "hunger" in needs:
            parts.append("hungry" if needs["hunger"] < 20
                         else "well-fed" if needs["hunger"] > 70
                         else "moderately hungry")
        if "energy" in needs:
            parts.append("exhausted" if needs["energy"] < 20
                         else "energetic" if needs["energy"] > 70
                         else "moderately tired")
        if "bladder" in needs:
            parts.append("urgently needs a bathroom" if needs["bladder"] < 20
                         else "bathroom needs fine")
        state["needs"] = "; ".join(parts)
        crit = [k for k, v in needs.items()
                if v < 20 and k not in ("social", "stimulation", "status")]
        state["urgent_needs"] = (", ".join(crit) if crit else "none critical")
    options = {
        "no": (f"No: {who} stays with their current activity ({who_act}) "
               f"and does not start a conversation with {target}"),
        "yes": (f"Yes: {who} initiates a friendly conversation with {target} "
                f"about {last_chat_about or 'a shared interest'} instead of "
                f"continuing {who_act}"),
    }
    instructions = ("Should the first agent initiate a conversation with the "
                    "other agent right now, or continue their own activity? "
                    "Weigh the person's current needs as stated.")
    key, conf = _typed_choice("talk", state, instructions, options,
                              key_bias=_talk_key_bias(who))
    try:
        from persona.prompt_template.ga_trace import record
        record("talk", who,
               {"conf": conf, "choice": key,
                "state": state, "options": options,
                "who_act": (who_act or "")[:150],
                "target": target, "target_act": (target_act or "")[:150]})
    except Exception:
        pass
    # Layer 3 sociability, 2026-10-05 fix: the legacy code only multiplied
    # the *confidence* of an already-chosen yes (caller ignores conf — a
    # dead knob). talk decisions now bias the actual probability mass via
    # _talk_key_bias: GA_TALK_YES_BIAS multiplier folded in with the
    # per-persona temperament talk_bias from town_model_settings.json.
    return key, conf