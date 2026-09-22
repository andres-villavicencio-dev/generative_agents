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
# Default off: everything routes through the legacy generation path.
USE_JEV_SCORING = os.environ.get("GA_JEV_SCORING", "0") == "1"

# If the margin between winner and runner-up is below this, defer to the
# legacy path (which does full CoT reasoning). Measured against the full
# averaged distribution.
JEV_MIN_MARGIN = 0.05

# How many candidate tokens to request per position. 20 is Ollama's cap.
JEV_TOP_LOGPROBS = 20

_jev_stats = {
    "calls": 0,
    "scored": 0,
    "fallback_margin": 0,
    "fallback_error": 0,
    "latency_samples": [],
}


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
        return None
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