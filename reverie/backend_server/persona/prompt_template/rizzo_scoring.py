"""
Rizzo Flow (local Jev) scoring for generative_agents.

Replaces expensive full-generation decisions with single-forward-pass typed
decisions from a local fine-tuned 4B (rizzo-flow, llama.cpp, port 8017).
Zero generated tokens, ~130ms per decision, calibrated-ish probabilities.

Design (mirrors jev_scoring.py):
- Master flag: GA_RIZZO_POIGNANCY=1 enables the rizzo fast path.
- Fail-open: any error -> (None, None) -> caller falls back to the legacy path.
- Confidence gate: rizzo's self-reported confidence is meaningful (measured on
  85 real sim events: conf >= 0.7 -> MAE 0.08; < 0.5 -> ~1.1). Decisions below
  GA_RIZZO_MIN_CONF are punted to the legacy path.
- IMPORTANT (measured 2026-09-27): the fine-tune expects the canonical
  /v1/systemone format with per-level `criteria` strings. Bare numeric levels
  on /v1/decisions cause middle-of-scale regression (MAE 0.64 vs 0.85 here).
"""
import json
import os
import time
import urllib.request

USE_RIZZO_POIGNANCY = os.environ.get("GA_RIZZO_POIGNANCY", "0") == "1"
RIZZO_URL = os.environ.get("GA_RIZZO_URL", "http://127.0.0.1:8017").rstrip("/")
RIZZO_MIN_CONF = float(os.environ.get("GA_RIZZO_MIN_CONF", "0.5"))
RIZZO_TIMEOUT = float(os.environ.get("GA_RIZZO_TIMEOUT", "20"))

# Rubric calibrated in the Sep 2026 A/B (see ga-sim-ops references/rizzo-flow-ab.md).
# Order = level 1..10; each string MUST start with "N = ".
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

_rizzo_stats = {
    "calls": 0, "answered": 0, "low_conf": 0, "errors": 0,
    "latency_samples": [],
}


def rizzo_stats():
    lat = _rizzo_stats["latency_samples"]
    out = dict(_rizzo_stats)
    out.pop("latency_samples")
    if lat:
        out["latency_p50_ms"] = sorted(lat)[len(lat) // 2]
        out["latency_max_ms"] = max(lat)
    return out


def rizzo_score_poignancy(agent, iss, description):
    """
    One typed decision: poignancy 1-10 of `description` for `agent`.
    Returns (score:int, confidence:float) or (None, None) to signal fallback.
    """
    if not USE_RIZZO_POIGNANCY:
        return None, None
    _rizzo_stats["calls"] += 1
    t0 = time.time()
    # Identity gist, not the full ISS: poignancy needs persona context, not the
    # whole bio. Keeps prefill lean and leaves room for long reflection
    # thoughts inside --ctx (a fresh insight scored pre-storage hit 2189
    # tokens on 2026-09-29 and 422'd against --ctx 2048).
    _iss = (iss or "").strip()
    if len(_iss) > 500:
        _iss = _iss[:500]
    body = {
        "model": "rizzo-latest",
        "state": {"who": agent, "identity": _iss, "event": description},
        "questions": {"poignancy": {
            "type": "score",
            "instructions": ("Rate the likely poignancy of the event for this "
                             "person, considering their identity and circumstances."),
            "criteria": _RUBRIC,
        }},
    }
    req = urllib.request.Request(
        f"{RIZZO_URL}/v1/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=RIZZO_TIMEOUT) as resp:
            answer = json.loads(resp.read())["answers"]["poignancy"]
    except Exception as e:
        _rizzo_stats["errors"] += 1
        print(f"[RIZZO] poignancy request failed ({type(e).__name__}: {e}); falling back")
        return None, None
    dt = time.time() - t0
    _rizzo_stats["latency_samples"].append(dt)
    if len(_rizzo_stats["latency_samples"]) > 200:
        _rizzo_stats["latency_samples"] = _rizzo_stats["latency_samples"][-200:]

    legend = answer.get("legend") or {}
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not k.startswith("__")}
    if not probs:
        _rizzo_stats["errors"] += 1
        return None, None
    best = max(probs, key=probs.get)
    label = str(legend.get(best, best))
    num = label.split("=")[0].strip() if "=" in label else label
    try:
        score = int(float(num))
    except (ValueError, TypeError):
        _rizzo_stats["errors"] += 1
        return None, None
    conf = probs[best]

    if not (1 <= score <= 10) or conf < RIZZO_MIN_CONF:
        _rizzo_stats["low_conf"] += 1
        return None, None
    _rizzo_stats["answered"] += 1
    return score, conf

# ============================================================================
# Reflex layer: react / talk typed decisions (Sep 2026 redesign)
# ============================================================================

USE_RIZZO_REFLEX = os.environ.get("GA_RIZZO_REFLEX", "1") == "1"


def _typed_choice(question_key, state_fields, instructions, options,
                  min_conf=None):
    """Shared implementation for choice-type decisions. Returns
    (option_key, confidence) or (None, None) to signal fallback. Never raises."""
    gate = RIZZO_MIN_CONF if min_conf is None else min_conf
    if not (USE_RIZZO_POIGNANCY and USE_RIZZO_REFLEX):
        return None, None
    _rizzo_stats["calls"] += 1
    t0 = time.time()
    body = {
        "model": "rizzo-latest",
        "state": {k: v[:600] for k, v in state_fields.items()
                  if isinstance(v, str)},
        "questions": {question_key: {
            "type": "choice",
            "instructions": instructions,
            "criteria": options,
        }},
    }
    req = urllib.request.Request(
        f"{RIZZO_URL}/v1/systemone",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=RIZZO_TIMEOUT) as resp:
            answer = json.loads(resp.read())["answers"][question_key]
    except Exception as e:
        _rizzo_stats["errors"] += 1
        print(f"[RIZZO] {question_key} request failed ({type(e).__name__}: {e}); "
              f"falling back")
        return None, None
    dt = time.time() - t0
    _rizzo_stats["latency_samples"].append(dt)
    if len(_rizzo_stats["latency_samples"]) > 200:
        _rizzo_stats["latency_samples"] = _rizzo_stats["latency_samples"][-200:]

    # The legend maps slot -> the option KEY we sent, so the answer is
    # directly usable as a dict key.
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not k.startswith("__")}
    legend = answer.get("legend") or {}
    if not probs:
        _rizzo_stats["errors"] += 1
        return None, None
    best = max(probs, key=probs.get)
    key = str(legend.get(best, best))
    if key not in options:
        _rizzo_stats["errors"] += 1
        return None, None
    conf = probs[best]
    if conf < gate:
        _rizzo_stats["low_conf"] += 1
        return None, None
    _rizzo_stats["answered"] += 1
    return key, conf


def rizzo_decide_react(who, who_act, target, target_act, venue,
                       relation=None, min_conf=None, exclusive=False):
    """decide_to_react as a typed choice: wait vs continue.
    Returns ('1'|'2', conf) matching the legacy answer codes, or (None, None).
    `exclusive=True` marks a capacity-1 resource (bathroom, shower) — the
    two activities genuinely conflict and waiting is the cooperative move."""
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
    return _typed_choice("react", state, instructions, options, min_conf)


def rizzo_decide_talk(who, who_act, target, target_act, last_chat_about=None,
                      min_conf=None, needs=None):
    """decide_to_talk as a typed yes/no choice.
    Returns ('yes'|'no', conf) or (None, None).
    `needs` = persona.scratch.needs dict (0-100, 100=satisfied). Social
    starvation should push toward talking — the reflex was needs-blind and
    the town froze at social 0."""
    # Layer 0 hard gate: a critical non-social need (<20: bathroom, hunger,
    # exhaustion) means no socializing right now — deterministic, no model
    # call. (In the sim these states usually trigger emergency replanning
    # before talk is even considered; this is defense-in-depth. A/B showed
    # rizzo over-weights loneliness vs a critical bladder, so we don't ask.)
    if needs:
        for k, v in needs.items():
            if v < 20 and k not in ("social", "stimulation"):
                return ("no", 0.99)
    state = {
        "who": who, "who_activity": who_act,
        "other": target, "other_activity": target_act,
        "last_chat": last_chat_about or "no recent conversation",
    }
    # Needs as human-readable facts in state (A/B v3: terse "social 0/100"
    # stats barely moved rizzo; human phrasing flips marginal cases with the
    # control still at 12/12). Instructions stay neutral — no witness-leading.
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
                if v < 20 and k not in ("social", "stimulation")]
        state["urgent_needs"] = (", ".join(crit) if crit
                                 else "none critical")
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
    return _typed_choice("talk", state, instructions, options, min_conf)
