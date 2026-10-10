"""
Cloud Jev (OpenRouter typesafe/jev-1.13) scoring for generative_agents.

Same interception pattern as rizzo_scoring.py, but the teacher is a cloud
System One decision model. EVERY answered call is appended to a harvest JSONL
(state + questions + full answer) — that log IS the training dataset for the
next local fine-tune (the whole point of this run).

Design:
- Master flag: GA_CLOUDJEV=1 enables the cloud path (checked BEFORE rizzo at
  each call site; rizzo stays as the local fallback cascade).
- Key: read once from GA_CLOUDJEV_KEY_FILE (default ~/.openrouter_key),
  chmod 600. Never logged, never printed.
- Fail-open: any error -> (None, None) -> caller falls back to rizzo/legacy.
- Harvest: GA_CLOUDJEV_HARVEST (default
  <repo>/jev_harvest/decisions.jsonl) — one JSON line per answered call:
  {"ts", "q", "state", "questions", "answer"} — plus stats counters.
- Rate/cost guard: GA_CLOUDJEV_MAX_CALLS (default 100000) hard-stops the
  cloud path if exceeded, so a wedged retry loop can't drain an account.
"""
import json
import os
import threading
import time
import urllib.request

USE_CLOUDJEV = os.environ.get("GA_CLOUDJEV", "0") == "1"
_KEY_FILE = os.environ.get("GA_CLOUDJEV_KEY_FILE",
                           os.path.expanduser("~/.openrouter_key"))
_MODEL = os.environ.get("GA_CLOUDJEV_MODEL", "typesafe/jev-1.13")
_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
_TIMEOUT = float(os.environ.get("GA_CLOUDJEV_TIMEOUT", "30"))
_MAX_CALLS = int(os.environ.get("GA_CLOUDJEV_MAX_CALLS", "100000"))
_HARVEST = os.environ.get(
    "GA_CLOUDJEV_HARVEST",
    "/home/andus/Projects/generative_agents/jev_harvest/decisions.jsonl")

# Same rubric as rizzo_scoring — the teacher must see the identical scale.
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

_key_cache = {"key": None, "checked": False}
_stats = {"calls": 0, "answered": 0, "errors": 0,
          "latency_samples": [], "cost_usd": 0.0}
_lock = threading.Lock()
_harvest_f = None


def _get_key():
    if not _key_cache["checked"]:
        _key_cache["checked"] = True
        try:
            with open(_KEY_FILE) as f:
                _key_cache["key"] = f.read().strip()
        except OSError:
            _key_cache["key"] = None
    return _key_cache["key"]


def _harvest_fh():
    global _harvest_f
    if _harvest_f is None:
        os.makedirs(os.path.dirname(_HARVEST), exist_ok=True)
        _harvest_f = open(_HARVEST, "a")
    return _harvest_f


def cloudjev_stats():
    lat = _stats["latency_samples"]
    out = dict(_stats)
    out.pop("latency_samples")
    if lat:
        out["latency_p50_ms"] = sorted(lat)[len(lat) // 2] * 1000
        out["latency_max_ms"] = max(lat) * 1000
    return out


def _decide(question_key, state, questions, min_conf=0.0):
    """One typed decision against the cloud teacher. Returns
    (answer_dict, None) on success or (None, None) to signal fallback.
    Harvests state + questions + full answer on success. Never raises."""
    # import-time binding of USE_CLOUDJEV is fine for the sim (env set before
    # boot), but a test or repl that flips the env post-import would silently
    # no-op — re-check here so the gate is always live.
    if os.environ.get("GA_CLOUDJEV", "0") != "1":
        return None, None
    key = _get_key()
    if not key:
        _stats["errors"] += 1
        return None, None
    with _lock:
        if _stats["answered"] >= _MAX_CALLS:
            return None, None
        _stats["calls"] += 1
    body = {"model": _MODEL, "state": state, "questions": questions}
    req = urllib.request.Request(
        _ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read())
    except Exception as e:
        _stats["errors"] += 1
        print(f"[CLOUDJEV] {question_key} request failed "
              f"({type(e).__name__}: {e}); falling back")
        return None, None
    dt = time.time() - t0
    answer = (payload.get("answers") or {}).get(question_key)
    if not isinstance(answer, dict):
        _stats["errors"] += 1
        return None, None
    _stats["latency_samples"].append(dt)
    if len(_stats["latency_samples"]) > 200:
        _stats["latency_samples"] = _stats["latency_samples"][-200:]
    try:
        _stats["cost_usd"] += float((payload.get("usage") or {}).get("cost", 0))
    except (TypeError, ValueError):
        pass
    # harvest the full exchange — this is the training data
    try:
        line = json.dumps({"ts": time.time(), "q": question_key,
                           "state": state, "questions": questions,
                           "answer": answer})
        with _lock:
            fh = _harvest_fh()
            fh.write(line + "\n")
            fh.flush()
    except Exception as e:
        print(f"[CLOUDJEV] harvest write failed ({e}) — answer still used")
    _stats["answered"] += 1
    return answer, None


def cloudjev_score_poignancy(agent, iss, description):
    """Poignancy 1-10 typed decision from the cloud teacher.
    Returns (score:int, confidence:float) or (None, None)."""
    _iss = (iss or "").strip()
    if len(_iss) > 500:
        _iss = _iss[:500]
    state = {"who": agent, "identity": _iss, "event": description}
    questions = {"poignancy": {
        "type": "score",
        "instructions": ("Rate the likely poignancy of the event for this "
                         "person, considering their identity and circumstances."),
        "criteria": _RUBRIC}}
    answer, _ = _decide("poignancy", state, questions)
    if answer is None:
        return None, None
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not str(k).startswith("__")}
    if not probs:
        return None, None
    best = max(probs, key=probs.get)
    # rubric strings start with "N = " — the level is the leading number
    label = str((answer.get("legend") or {}).get(best, best))
    num = label.split("=")[0].strip() if "=" in label else label
    try:
        score = int(float(num))
    except (ValueError, TypeError):
        return None, None
    if not (1 <= score <= 10):
        return None, None
    return score, float(probs[best])


def cloudjev_decide_react(who, who_act, target, target_act, venue,
                         relation=None, exclusive=False):
    """decide_to_react as a cloud typed choice. Returns ('1'|'2', conf)."""
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
    state = {"who": who, "who_activity": who_act,
             "other": target, "other_activity": target_act,
             "venue": venue, "relationship": relation or "acquaintances in town",
             "facility": ("a single-person facility (only one person can use it at "
                          "a time)" if exclusive else "a shared public space")}
    questions = {"react": {
        "type": "choice",
        "instructions": ("Two agents are heading toward the same venue in a small "
                         "town. Should the first agent wait for the second agent "
                         "to finish their activity, or continue with their own plan?"),
        "criteria": options}}
    answer, _ = _decide("react", state, questions)
    if answer is None:
        return None, None
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not str(k).startswith("__")}
    if not probs:
        return None, None
    best = max(probs, key=probs.get)
    key = str((answer.get("legend") or {}).get(best, best))
    if key not in options:
        return None, None
    return key, float(probs[best])


def cloudjev_decide_talk(who, who_act, target, target_act, last_chat_about=None,
                         needs=None):
    """decide_to_talk as a cloud typed choice. Returns ('yes'|'no', conf).
    Mirrors rizzo's needs-in-state phrasing so the teacher sees the same facts."""
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
    options = {
        "no": (f"No: {who} stays with their current activity ({who_act}) "
               f"and does not start a conversation with {target}"),
        "yes": (f"Yes: {who} initiates a friendly conversation with {target} "
                f"about {last_chat_about or 'a shared interest'} instead of "
                f"continuing {who_act}"),
    }
    questions = {"talk": {
        "type": "choice",
        "instructions": ("Should the first agent initiate a conversation with the "
                         "other agent right now, or continue their own activity? "
                         "Weigh the person's current needs as stated."),
        "criteria": options}}
    answer, _ = _decide("talk", state, questions)
    if answer is None:
        return None, None
    probs = {k: v for k, v in (answer.get("probabilities") or {}).items()
             if not str(k).startswith("__")}
    if not probs:
        return None, None
    best = max(probs, key=probs.get)
    key = str((answer.get("legend") or {}).get(best, best))
    if key not in options:
        return None, None
    return key, float(probs[best])