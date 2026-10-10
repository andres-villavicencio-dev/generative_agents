# Copyright (c) 2026 Andus VU — MIT license
"""Agentic drives (PIANO-style) for generative agents.

Layer 1 of the agency stack: continuous drive weights (hunger/energy/social/
curiosity/purpose) that ALWAYS compete for behavior control, arbitrated at
the action-selection seam — replacing the threshold-reflex model where needs
only mattered when they crossed a critical line.

Design notes:
- Drives are the single source of behavioral pressure; the legacy needs
  meters keep decaying as the *somatic layer* underneath. Drive salience is
  a monotone function of the needs + context (crowd density, last social
  contact, artifact progress, novelty starvation).
- Arbitration runs at EVERY action decision as a typed decision via the
  laya-style /v1/systemone wire when GA_DRIVES_LLM=1 (server side), or a
  deterministic softmax when not — both env-gated, default OFF.

This module is imported by plan.py (deterministic part) so anything that
crashes here must fail open — every public function wraps its body.
"""
import os
import random
import math

DRIVES = ["hunger", "energy", "social", "curiosity", "purpose"]

# Legacy needs -> drive mapping (somatic layer)
_SOMATIC = {
    "hunger": "hunger",
    "energy": "energy",
    "bladder": "hunger",     # discomfort folds into hunger-drive salience
    "hygiene": "energy",
    "hydration": "hunger",
}

# Drive -> resource_goal string (the EXISTING pursuit machinery executes
# these; the drive layer only decides WHICH goal to inject).
DRIVE_GOALS = {
    "hunger": "eating a meal or getting food from Hobbs Cafe",
    "energy": "going home to take a rest",
    "social": "heading somewhere lively to spend time with people",
    "curiosity": "exploring a part of the ville they rarely visit",
    "purpose": "spending focused time on their personal project",
}

SOCIAL_VENUES = ["Hobbs Cafe", "Johnson Park", "The Rose and Crown Pub",
                 "Oak Hill College", "Harvey Oak Supply Store"]


def _env_on(flag="GA_DRIVES"):
    return os.environ.get(flag, "0") == "1"


def _recent_event_history(persona, k=15):
    """(arena, created) for the persona's k most recent event nodes — read
    from a_mem (checkpoint-persistent, no extra recording seams)."""
    try:
        nodes = [n for n in persona.a_mem.id_to_node.values() if n.type == "event"]
        nodes.sort(key=lambda n: n.created or "", reverse=True)
        out = []
        for n in nodes[:k]:
            arena = n.description.split("@")[-1].strip() if "@" in n.description else "unknown"
            out.append(arena)
        return out
    except Exception:
        return []


def _last_chat_time(persona):
    """Created-time of the persona's most recent chat memory node (None if
    never). Cached on the node count so the O(N) scan is amortized."""
    try:
        am = persona.a_mem
        cache_key = getattr(am, "node_count", None)
        cached = getattr(persona.scratch, "_last_chat_cache", None)
        if cached and cached[0] == cache_key:
            return cached[1]
        best = None
        for n in am.id_to_node.values():
            if n.type == "chat" and n.created is not None:
                if best is None or n.created > best:
                    best = n.created
        persona.scratch._last_chat_cache = (cache_key, best)
        return best
    except Exception:
        return None


def drive_salience(persona, maze=None):
    """Compute {drive: salience in [0,1]}. Sources are the agent's OWN
    recorded state (needs dict) + its own memory graph (a_mem events/chats)
    — checkpoint-persistent, no parallel recording. Deterministic, no LLM.
    Safe under partial scratch (old checkpoints)."""
    out = {d: 0.0 for d in DRIVES}
    try:
        sc = persona.scratch
        needs = getattr(sc, "needs", None) or {}
        critical = getattr(sc, "needs_critical", 20)
        for need, drv in _SOMATIC.items():
            val = needs.get(need, 100)
            deficit = max(0.0, (100 - val) / 100.0)
            out[drv] = max(out[drv], deficit)
        for need, drv in _SOMATIC.items():
            if needs.get(need, 100) < critical:
                out[drv] = max(out[drv], 0.95)
        # social drive: hours since the last REAL conversation (chat memory
        # node). GA_SOCIAL_DRIVE_MULT (default 1.6, 2026-10-05): quiet-era
        # towns under-weight social pull vs curiosity's solo wandering; the
        # multiplier makes shared-venue goals competitive again.
        last_chat = _last_chat_time(persona)
        now = getattr(sc, "curr_time", None)
        _soc_mult = float(os.environ.get("GA_SOCIAL_DRIVE_MULT", "1.6"))
        if last_chat is not None and now is not None:
            hours = (now - last_chat).total_seconds() / 3600.0
            out["social"] = max(out["social"],
                                min(1.0, _soc_mult * hours / 8.0))
        else:
            out["social"] = max(out["social"], min(1.0, 0.5 * _soc_mult))
        # curiosity: venue-novelty of recent own history
        hist = _recent_event_history(persona)
        recent = hist[-12:]
        if recent:
            uniq = len(set(recent)) / max(len(recent), 1)
            out["curiosity"] = max(out["curiosity"], 1.0 - uniq)
        else:
            out["curiosity"] = max(out["curiosity"], 0.4)
        # purpose: unserved projects/artifact progress + somatic room
        prog = getattr(sc, "artifact_progress", {}) or {}
        projs = getattr(sc, "active_projects", []) or []
        somatic_room = 1.0 - max(out["hunger"], out["energy"])
        base = 0.6 if (prog or projs) else 0.5
        out["purpose"] = max(0.0, somatic_room * base)
        return out
    except Exception:
        return out


# Deterministic softmax arbitration (no-LLM fallback path)
def arbitrate(persona, maze=None, curiosity_boost=0.0):
    """Returns the winning drive name (str), or None when GA_DRIVES=0.
    Deterministic softmax over salience with per-persona taste jitter.
    curiosity_boost: extra salience from learning-progress/boredom (L3)."""
    try:
        if not _env_on():
            return None
        sal = drive_salience(persona, maze)
        if curiosity_boost:
            sal["curiosity"] = min(1.0, sal["curiosity"] + curiosity_boost)
        rng = random.Random(f"{getattr(persona.scratch, 'name', '?')}-drive")
        tastes = {d: rng.uniform(0.85, 1.15) for d in DRIVES}
        scores = {d: sal[d] * tastes[d] for d in DRIVES}
        # curiosity only wins when the body is basically fine
        if scores["hunger"] >= 0.45 or scores["energy"] >= 0.45:
            scores["curiosity"] *= 0.8
        return max(scores, key=scores.get)
    except Exception:
        return None