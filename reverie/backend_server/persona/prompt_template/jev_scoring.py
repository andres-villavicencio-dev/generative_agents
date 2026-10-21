"""
Local JEV-style decision scoring for generative_agents.

Implements the "System One" pattern from TypeSafe AI's Jev locally, using the
same Ollama daemon that already serves the simulation, with zero additional
infrastructure: for a decision with K known answers, we run ONE forward pass
of the prompt (ending exactly at the decision point) and read the top-N
next-token logprobs at the answer position. The K answer tokens' probabilities
form the distribution. No autoregressive loop, no JSON parsing, no retries.

Why not SGLang /v1/score: Ollama (>= 0.12.x, commit 59241c5) exposes
`logprobs` + `top_logprobs` on /api/generate, which lets us implement the
exact same restricted-softmax-over-known-choices on the model the sim
already runs. One daemon, one GPU allocation, zero VRAM contention.

Design notes
------------
- Labels are single tokens. "yes"/"no" are single tokens in gemma/qwen
  tokenizers; we verify at runtime via /api/generate's own logprobs response
  (the returned token strings are the tokenizer's rendering) and fall back
  to generation on any mismatch.
- Position-bias mitigation (order sensitivity is a known footgun of
  next-token scoring): for small K we score ALL orderings of the options
  list and average the distributions. For K=2 that's 2 prompts; the cost
  is still one forward pass each.
- Calibration: raw logprob mass is NOT a validated confidence. The
  distribution is returned to the caller so thresholds can be tuned; a
  `min_margin` gate keeps low-margin decisions on the legacy path.
- Fail-open: any error, missing capability, or low-margin decision falls
  back to the legacy generate path. The simulation must never stall on
  this experiment.
"""
import json
import math
import os
import time
import urllib.request
import urllib.error
import itertools

from persona.prompt_template.gpt_structure import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL

# Master flag: set GA_JEV_SCORING=1 in the backend's environment to enable.
# Backend: laya typed decisions by default (see JEV_BACKEND below).
# Default off: everything routes through the legacy generation path.
USE_JEV_SCORING = os.environ.get("GA_JEV_SCORING", "0") == "1"

# Per-call-type margin gates. A decision that isn't sharply separated from
# its runner-up defers to the legacy CoT path. Poignancy is a 10-way score
# with a forgiving downstream (importance recency-weighting), spatial choice
# picks a literal building — wrong pick = visible pathing weirdness, so it
# needs a higher bar.
JEV_MIN_MARGIN = 0.05
JEV_SPATIAL_MIN_MARGIN = 0.10

# How many candidate tokens to request per position. 20 is Ollama's cap.
JEV_TOP_LOGPROBS = 20

_jev_stats = {
    "calls": 0,
    "scored": 0,
    "fallback_margin": 0,
    "fallback_error": 0,
    "latency_samples": [],
}


# Capability breaker (Oct 21 2026, n68): cloud models proxied by ollama
# (minimax-m3.1-flash:cloud) return NO logprobs and an empty first token —
# every JEV call was a guaranteed 1-2s miss before the legacy path ran.
# After this many consecutive logprob-less responses, disable JEV for the
# process. Callers re-import USE_JEV_SCORING per call, so this takes effect.
JEV_NO_LOGPROBS_TRIP = 3


def _note_no_logprobs():
    global USE_JEV_SCORING
    _jev_stats["no_logprobs_streak"] = _jev_stats.get("no_logprobs_streak", 0) + 1
    if USE_JEV_SCORING and _jev_stats["no_logprobs_streak"] >= JEV_NO_LOGPROBS_TRIP:
        USE_JEV_SCORING = False
        print(f"[JEV] {OLLAMA_CHAT_MODEL} returned no logprobs "
              f"{JEV_NO_LOGPROBS_TRIP}x in a row — JEV scoring disabled "
              f"for this process (legacy path only).")


# Backend (Oct 21 2026): JEV's typed decisions belong on laya — the
# fine-tuned local System-One decision model (ga-laya.service :8018) — not
# on the chat model's next-token logprobs. GA_JEV_BACKEND=ollama restores
# the logprob path (local completion models only; :cloud has no logprobs).
JEV_BACKEND = os.environ.get("GA_JEV_BACKEND", "laya")
JEV_LAYA_CONTEXT_CHARS = 1500
# GA_JEV_LAYA_MODE: "shadow" (default) asks laya on a background thread,
# traces its answer, and returns None so the cloud teacher still decides;
# "live" lets laya's answer drive the sim. Measured Oct 21 2026 (n68 call
# log, laya v3 vs minimax picks): sector 6/11 agreement with CONFIDENT misses
# (library -> Johnson Park @ margin 0.95), objects 6-7/8, arenas 2/2 —
# laya-v3 was never trained on spatial choice, and poignancy/talk/react only
# reach JEV after laya already deferred. Shadow traces are the paired
# (laya pick, teacher pick) data to fine-tune that in.
JEV_LAYA_MODE = os.environ.get("GA_JEV_LAYA_MODE", "shadow")
# Per-kind live override for spatial choices on the legacy per-level path
# ("sector,arena,object"): those kinds let laya decide even in shadow mode.
# Default empty — the unified location call (one teacher call for all three
# levels) already removed these calls, and laya-v3's spatial picks are not
# yet trustworthy (sector 6/11, Oct 21 eval).
JEV_LAYA_LIVE_KINDS = {k.strip() for k in
                       os.environ.get("GA_JEV_LAYA_LIVE_KINDS", "").split(",")
                       if k.strip()}


def _shadow(fn, *args):
    """Run a laya decision off the critical path; its trace is the output."""
    import threading as _t
    from persona.prompt_template.gpt_structure import _PERSONA_CTX
    who = getattr(_PERSONA_CTX, "name", None)

    def _run():
        _PERSONA_CTX.name = who
        try:
            fn(*args)
        except Exception:
            pass
    _t.Thread(target=_run, daemon=True).start()


def _laya_typed(kind, context, instructions, criteria, trace_payload=None):
    """One laya /v1/systemone typed question. kind: 'choice' (criteria =
    {key: description}) or 'score' (criteria = rubric list). Returns
    {resolved_label: prob} (normalized) or None on any failure."""
    from persona.prompt_template.laya_scoring import _systemone
    state = {"context": (context or "")[-JEV_LAYA_CONTEXT_CHARS:]}
    questions = {"jev": {"type": kind, "instructions": instructions,
                         "criteria": criteria}}
    try:
        answers = _systemone(state, questions)
    except Exception as e:
        _jev_stats["fallback_error"] += 1
        print(f"[JEV] laya request failed ({type(e).__name__}: {e}); falling back")
        return None
    answer = (answers or {}).get("jev")
    if not isinstance(answer, dict):
        _jev_stats["fallback_error"] += 1
        return None
    legend = answer.get("legend") or {}
    probs = {}
    for k, v in (answer.get("probabilities") or {}).items():
        if str(k).startswith("__"):
            continue
        label = str(legend.get(k, k))
        probs[label] = probs.get(label, 0.0) + float(v)
    total = sum(probs.values())
    if total <= 0:
        _jev_stats["fallback_error"] += 1
        return None
    probs = {k: v / total for k, v in probs.items()}
    if trace_payload is not None:
        try:
            from persona.prompt_template.ga_trace import record
            from persona.prompt_template.gpt_structure import _PERSONA_CTX
            best = max(probs, key=probs.get)
            record(trace_payload, getattr(_PERSONA_CTX, "name", None),
                   {"conf": probs[best], "choice": best,
                    "mode": JEV_LAYA_MODE, "dist": probs,
                    "state": state, "questions": questions})
        except Exception:
            pass
    return probs


def _rank(dist):
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    winner, winner_p = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, winner_p - runner_p


def _laya_decide(question_prompt, choices, return_dist):
    block = "\n".join(f"- {m}" for m in choices.values())
    context = question_prompt.replace("__JEV_OPTIONS__", block)
    probs = _laya_typed(
        "choice", context,
        "Choose the answer the context's question calls for.",
        {k: str(v) for k, v in choices.items()}, trace_payload="jev_decide")
    if not probs:
        return None, None
    dist = {k: p for k, p in probs.items() if k in choices}
    if not dist:
        _jev_stats["fallback_error"] += 1
        return None, None
    winner, margin = _rank(dist)
    _jev_stats["scored"] += 1
    if margin < JEV_MIN_MARGIN:
        _jev_stats["fallback_margin"] += 1
        return (None, dist) if return_dist else (None, None)
    return choices[winner], (dist if return_dist else None)


def _laya_score_digit(prompt, lo, hi, min_margin):
    if (lo, hi) == (1, 10):
        from persona.prompt_template.laya_scoring import _RUBRIC
        rubric = list(_RUBRIC)
    else:
        rubric = [str(d) for d in range(lo, hi + 1)]
    probs = _laya_typed("score", prompt,
                        "Answer the rating the context asks for.", rubric,
                        trace_payload="jev_digit")
    if not probs:
        return None
    dist = {}
    for label, p in probs.items():
        try:
            d = int(str(label).split("=")[0].strip())
        except ValueError:
            continue
        if lo <= d <= hi:
            dist[d] = dist.get(d, 0.0) + p
    if not dist:
        _jev_stats["fallback_error"] += 1
        return None
    winner, margin = _rank(dist)
    _jev_stats["scored"] += 1
    if margin < min_margin:
        _jev_stats["fallback_margin"] += 1
        return None
    return winner


def _laya_score_choice(prompt, options, min_margin, return_dist):
    block = "\n".join(f"- {o}" for o in options)
    context = (prompt.replace("__JEV_OPTIONS__", block)
                     .replace("Answer with the letter only:", "").strip())
    criteria = {o: o for o in options}
    probs = _laya_typed("choice", context,
                        "Pick the option the context's question calls for.",
                        criteria, trace_payload="spatial")
    if not probs:
        return None
    dist = {o: probs.get(o, 0.0) for o in options}
    if sum(dist.values()) <= 0:
        _jev_stats["fallback_error"] += 1
        return None
    winner, margin = _rank(dist)
    _jev_stats["scored"] += 1
    if margin < min_margin:
        _jev_stats["fallback_margin"] += 1
        return (None, dist) if return_dist else None
    return (winner, dist) if return_dist else winner


def _post(url_path, body, timeout=300):
    req = urllib.request.Request(
        url=f"{OLLAMA_BASE_URL}{url_path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _score_one_ordering(prompt, answer_tokens):
    """
    Run one forward pass ending at the decision point and extract
    probabilities for the given answer tokens from top-N logprobs.
    Returns dict {token_text: prob} or None on any failure.
    """
    body = {
        "model": OLLAMA_CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        # Modern-model compatibility: suppress reasoning so the answer token
        # lands at position 1 where the restricted-softmax scoring expects it.
        # Verified 2026-09-23 on qwen3.5:2b — clean digit logprobs at top.
        "think": False,
        "options": {"num_ctx": 8192, "temperature": 0, "num_predict": 1},
        "logprobs": True,
        "top_logprobs": JEV_TOP_LOGPROBS,
    }
    try:
        result = _post("/api/generate", body)
    except Exception as e:
        _jev_stats["fallback_error"] += 1
        print(f"[JEV] scoring request failed: {e}")
        return None

    logprob_list = result.get("logprobs")
    if not logprob_list:
        _jev_stats["fallback_error"] += 1
        _note_no_logprobs()
        return None
    _jev_stats["no_logprobs_streak"] = 0
    # Ollama returns logprobs as a list of per-generated-token entries; with
    # num_predict=1 there is exactly one entry.
    entry = logprob_list[0]
    top = entry.get("top_logprobs") or []
    if not top:
        _jev_stats["fallback_error"] += 1
        return None

    # Map: normalized token text -> probability. Restricted softmax happens
    # implicitly: we renormalize over the answer tokens only.
    probs = {}
    for alt in top:
        tok_text = (alt.get("token") or "").strip().lower()
        if tok_text in answer_tokens:
            probs[tok_text] = math.exp(alt.get("logprob", -100.0))
    if not probs:
        return None
    return probs


def _average_distributions(dists):
    """Average a list of {token: prob} dicts into one normalized dict."""
    tokens = set()
    for d in dists:
        tokens.update(d.keys())
    avg = {t: sum(d.get(t, 0.0) for d in dists) / len(dists) for t in tokens}
    total = sum(avg.values())
    if total <= 0:
        return None
    return {t: p / total for t, p in avg.items()}


def jev_decide(question_prompt, choices, return_dist=False):
    """
    Make a bounded decision: given a prompt ending at the decision point
    and a dict {answer_token: meaning}, return the chosen meaning and the
    full averaged distribution.

    question_prompt must end EXACTLY at the position where the answer
    token would be generated. The caller controls the prompt template.

    Returns (decision, distribution) or (None, None) to signal fallback.
    """
    if not USE_JEV_SCORING:
        return None, None
    _jev_stats["calls"] += 1
    t0 = time.time()
    if JEV_BACKEND == "laya":
        if JEV_LAYA_MODE != "live":
            _shadow(_laya_decide, question_prompt, choices, return_dist)
            return None, None
        return _laya_decide(question_prompt, choices, return_dist)

    answer_tokens = list(choices.keys())
    # Order-sensitivity mitigation: score every ordering (K! is small here).
    orderings = list(itertools.permutations(answer_tokens))
    if len(orderings) > 6:  # safety valve for large K
        orderings = orderings[:6]

    dists = []
    for ordering in orderings:
        # Rebuild the options block in THIS ordering inside the prompt.
        options_block = "\n".join(
            f"- {meaning}" for meaning in (choices[t] for t in ordering)
        )
        prompt = question_prompt.replace("__JEV_OPTIONS__", options_block)
        d = _score_one_ordering(prompt, set(ordering))
        if d is None:
            return None, None  # hard fail -> fallback
        dists.append(d)

    avg = _average_distributions(dists)
    if not avg:
        return None, None

    ranked = sorted(avg.items(), key=lambda kv: -kv[1])
    winner, winner_p = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = winner_p - runner_p

    _jev_stats["scored"] += 1
    _jev_stats["latency_samples"].append(time.time() - t0)
    if len(_jev_stats["latency_samples"]) > 200:
        _jev_stats["latency_samples"] = _jev_stats["latency_samples"][-200:]

    if margin < JEV_MIN_MARGIN:
        _jev_stats["fallback_margin"] += 1
        if return_dist:
            return None, avg
        return None, None

    return choices[winner], avg if return_dist else None


def jev_score_digit(prompt, lo=1, hi=10, min_margin=JEV_MIN_MARGIN):
    """
    Score an integer answer (poignancy 1-10) from ONE forward pass.
    Prompt must end exactly at the answer position (e.g. 'Rate (return a
    number between 1 to 10):'). Returns int or None (caller falls back to
    legacy path).
    """
    if not USE_JEV_SCORING:
        return None
    _jev_stats["calls"] += 1
    t0 = time.time()
    if JEV_BACKEND == "laya":
        if JEV_LAYA_MODE != "live":
            _shadow(_laya_score_digit, prompt, lo, hi, min_margin)
            return None
        return _laya_score_digit(prompt, lo, hi, min_margin)

    digit_tokens = {str(d): d for d in range(lo, hi + 1)}
    probs = _score_one_ordering(prompt, set(digit_tokens.keys()))
    if not probs:
        _jev_stats["fallback_error"] += 1
        return None

    total = sum(probs.values())
    if total <= 0:
        _jev_stats["fallback_error"] += 1
        return None
    dist = {digit_tokens[t]: p / total for t, p in probs.items()}

    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    winner, winner_p = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0

    _jev_stats["scored"] += 1
    _jev_stats["latency_samples"].append(time.time() - t0)
    if len(_jev_stats["latency_samples"]) > 200:
        _jev_stats["latency_samples"] = _jev_stats["latency_samples"][-200:]

    if winner_p - runner_p < min_margin:
        _jev_stats["fallback_margin"] += 1
        return None

    return winner


def jev_score_choice(prompt, options, min_margin=JEV_SPATIAL_MIN_MARGIN,
                     return_dist=False, kind=None):
    """
    Score a choice over enumerated multi-token options (sector/arena/
    game_object names). Uses letter labels (A/B/C/...) so the FIRST token
    after the decision point identifies the pick; permuting the label
    assignment mitigates order bias. Returns the chosen option STRING
    (verbatim) or None.
    """
    if not USE_JEV_SCORING:
        return None
    if not options:
        return None
    _jev_stats["calls"] += 1
    t0 = time.time()
    if JEV_BACKEND == "laya":
        if JEV_LAYA_MODE != "live" and kind not in JEV_LAYA_LIVE_KINDS:
            _shadow(_laya_score_choice, prompt, options, min_margin, return_dist)
            return None
        return _laya_score_choice(prompt, options, min_margin, return_dist)

    labels = [chr(ord("A") + i) for i in range(len(options))]
    # Order-bias mitigation: try 2 label permutations (forward + reversed).
    # K! for sectors is large; 2 samples bound the cost at 2 passes.
    permutations_to_try = [list(range(len(options)))]
    if len(options) > 2:
        permutations_to_try.append(list(reversed(range(len(options)))))

    dists = []
    option_by_label = None
    for perm in permutations_to_try:
        # perm[i] = index into options for label i
        option_by_label = {labels[i]: options[perm[i]] for i in range(len(labels))}
        options_block = "\n".join(
            f"{labels[i]}) {options[perm[i]]}" for i in range(len(labels))
        )
        p = prompt.replace("__JEV_OPTIONS__", options_block)
        d = _score_one_ordering(p, set(l.lower() for l in labels))
        if d is None:
            return None  # hard fail -> fallback
        # Map label probs onto option strings (scores come back lowercase)
        d_opts = {}
        for t, prob in d.items():
            if t.upper() in option_by_label:
                d_opts[option_by_label[t.upper()]] = prob
        dists.append(d_opts)

    # Drop empty perm-dist maps (soft miss: label not in top-20 logprobs) —
    # but require at least one non-empty map to proceed.
    dists = [d for d in dists if d]
    if not dists:
        _jev_stats["fallback_error"] += 1
        return None

    avg = _average_distributions(dists)
    if not avg:
        _jev_stats["fallback_error"] += 1
        return None

    ranked = sorted(avg.items(), key=lambda kv: -kv[1])
    winner, winner_p = ranked[0]
    runner_p = ranked[1][1] if len(ranked) > 1 else 0.0

    _jev_stats["scored"] += 1
    _jev_stats["latency_samples"].append(time.time() - t0)
    if len(_jev_stats["latency_samples"]) > 200:
        _jev_stats["latency_samples"] = _jev_stats["latency_samples"][-200:]

    if winner_p - runner_p < min_margin:
        _jev_stats["fallback_margin"] += 1
        if return_dist:
            return None, avg
        return None

    return (winner, avg) if return_dist else winner


def jev_stats_reset():
    _jev_stats.update({
        "calls": 0, "scored": 0, "fallback_margin": 0,
        "fallback_error": 0, "latency_samples": [],
    })


def jev_stats_snapshot():
    lat = _jev_stats["latency_samples"]
    return {
        **{k: v for k, v in _jev_stats.items() if k != "latency_samples"},
        "latency_avg_s": (sum(lat) / len(lat)) if lat else None,
    }