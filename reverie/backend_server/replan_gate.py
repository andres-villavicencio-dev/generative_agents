"""Laya gate for the re-plan after a chat: skip the (slow) schedule-rewriting LLM call when the conversation did not
commit the agent to anything. Modes (GA_REPLAN_LAYA): off (default) | shadow (ask + trace, always re-plan) | live
(skip the LLM call when Laya says keep with prob >= GA_REPLAN_KEEP_CONF, default 0.75; ties/errors re-plan).
Server: GA_REPLAN_LAYA_URL (default GA_NEEDS_LAYA_URL, then the sim's Laya)."""
import json, os, re, time, urllib.request

MODE = os.environ.get("GA_REPLAN_LAYA", "off").lower()
KEEP_CONF = float(os.environ.get("GA_REPLAN_KEEP_CONF", "0.75"))
CRITERIA = {
    "replan": "the conversation created a commitment (appointment, promise, invitation, task to start) that changes the plan",
    "keep": "no commitment: ordinary talk, opinions or gossip; carry on with the plan",
}
SLUGS = list(CRITERIA)
INSTRUCTIONS = "Does this conversation require changing the rest of the person's plan? Pick replan only for a real commitment."


def situation(persona, window, dur, plan, discussed):
    window = re.sub(r"\s*[AP]M", "", window or "")
    return (f"{persona} had planned, between {window}: {'; '.join(plan[:6])}.\n"
            f"Then {persona} had a {dur}-minute conversation. What was discussed: {(discussed or '')[:400]}")


def _trace(d):
    try:
        base = os.environ.get("GA_TRACE_DIR") or os.path.join(os.path.dirname(__file__), "../../ga_traces")
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "replan_gate.jsonl"), "a") as f:
            f.write(json.dumps(d) + "\n")
    except Exception:
        pass


def ask(persona, window, dur, plan, discussed):
    """-> (slug, keep_prob) or (None, None). Never raises."""
    try:
        from persona.prompt_template import laya_scoring as L
        url = (os.environ.get("GA_REPLAN_LAYA_URL") or os.environ.get("GA_NEEDS_LAYA_URL") or L.LAYA_URL).rstrip("/")
        body = {"model": "laya-v3",
                "state": {"who": persona, "situation": situation(persona, window, dur, plan, discussed)},
                "questions": {"replan": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}}
        req = urllib.request.Request(f"{url}/v1/systemone", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=L.LAYA_TIMEOUT) as resp:
            ans = (json.loads(resp.read()).get("answers") or {}).get("replan")
        probs = {k: v for k, v in (ans.get("probabilities") or {}).items() if not str(k).startswith("__")}
        legend = ans.get("legend") or {}
        mass = {s: 0.0 for s in SLUGS}
        for k, v in probs.items():
            lab = str(legend.get(k, k))
            for s in SLUGS:
                if lab in (s, CRITERIA[s]):
                    mass[s] += float(v)
        tot = sum(mass.values())
        if tot <= 0:
            return None, None
        keep = mass["keep"] / tot
        return ("keep" if keep >= 0.5 else "replan"), keep
    except Exception:
        return None, None


def should_skip(persona, window, dur, plan, discussed):
    """True = skip the re-plan LLM call. Always False unless mode is live."""
    if MODE not in ("shadow", "live"):
        return False
    t0 = time.time()
    slug, keep = ask(persona, window, dur, plan, discussed)
    skip = MODE == "live" and slug == "keep" and keep is not None and keep >= KEEP_CONF
    _trace({"t": time.time(), "who": persona, "window": window, "dur": dur, "plan": plan[:6],
            "discussed": (discussed or "")[:300], "slug": slug, "keep": keep, "skip": skip, "mode": MODE,
            "ms": int((time.time() - t0) * 1000)})
    return skip
