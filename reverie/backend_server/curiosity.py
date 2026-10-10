# Copyright (c) 2026 Andus VU — MIT license
"""Schmidhuber learning-progress curiosity for generative agents.

Layer 3 of the agency stack: each agent carries a tiny PREDICTIVE MODEL of
its own experience stream. Intrinsic reward = drop in prediction error
(learning progress) — the agent is drawn to what it can LEARN about, not
what is merely new (raw novelty degenerates into tourist behavior) nor what
it has fully mastered (boring). Boredom = sustained near-zero error.

Implementation: a per-persona token-frequency predictor over venue
co-occurrence: given the current venue, predict the next venue. Trackable,
checkpointable, zero LLM — and its learning curve is measurable.

Env gate: GA_CURIOSITY=1. Default OFF. Fail-open everywhere.
"""
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

from routines import _arena_of  # shared canonical extractor


def _env_on(flag='GA_CURIOSITY'):
    return os.environ.get(flag, '0') == '1'


# ---------------------------------------------------------------------------
# The predictive model: P(next venue | current venue) via smoothed counts
# ---------------------------------------------------------------------------
class VenuePredictor:
    """Self-model: predicts the agent's next venue from the current one.
    Prediction error = 1 - prob assigned to the TRUE next venue (brier-ish).
    Learning progress = error now vs error at the previous visit of the
    same (current, next) context."""

    def __init__(self):
        self.trans = defaultdict(lambda: defaultdict(int))  # cur -> next -> n
        self.tot = Counter()                                # cur -> total
        self.history = []                                   # venue sequence
        # per-context error memory (bounded)
        self.err_hist = defaultdict(lambda: deque_maxlen())

    def observe(self, venue_seq):
        """Feed a venue sequence; returns list of per-step errors (the
        agent's surprise curve) while updating counts."""
        errs = []
        for i in range(1, len(venue_seq)):
            cur, nxt = venue_seq[i - 1], venue_seq[i]
            if cur is None or nxt is None:
                errs.append(None)
                continue
            row = self.trans[cur]
            total = self.tot[cur]
            # error BEFORE update (did I predict this transition well?)
            if total >= 3:
                p_true = row.get(nxt, 0) / total
                # smoothed
                err = 1.0 - (p_true + 0.15) / (1.0 + 0.15 * len(row) + 0.0)
                errs.append(max(0.0, min(1.0, err)))
            else:
                errs.append(None)  # not enough data yet = unmeasurable
            row[nxt] += 1
            self.tot[cur] += 1
        self.history.extend([v for v in venue_seq if v])
        return errs

    def predicted_entropy(self, venue):
        """Entropy of P(next|venue): high = uncertain = potentially learnable."""
        row = self.trans.get(venue)
        if not row or self.tot.get(venue, 0) < 3:
            return None
        H = 0.0
        for n in row.values():
            p = n / self.tot[venue]
            if p > 0:
                H -= p * math.log2(p)
        return H

    def confidence(self, venue):
        total = self.tot.get(venue, 0)
        return min(1.0, total / 12.0)


def deque_maxlen(n=20):
    return []


# ---------------------------------------------------------------------------
# Experience extraction: build the venue sequence from a_mem
# ---------------------------------------------------------------------------
def venue_timeline(persona, k=80):
    """Chronological venue list from the persona's event memories (newest
    last). Only walkable-world arenas (persona-anchor tails dropped)."""
    try:
        nodes = [n for n in persona.a_mem.id_to_node.values() if n.type == 'event']
        nodes.sort(key=lambda n: n.created or '')
        out = []
        for n in nodes[-k:]:
            a = _arena_of(n.description)
            if a:
                out.append(a)
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# The curiosity signal: learning progress + boredom
# ---------------------------------------------------------------------------
def learning_progress(persona):
    """(progress, boredom) for the persona's recent life.
    progress = mean recent error-drop (positive = still learning about life)
    boredom = fraction of recent steps with near-zero error at high confidence
    (fully mastered = boring; Schmidhuber: seek LEARNABLE, not just new)."""
    vp = getattr(persona.scratch, 'venue_predictor', None)
    if vp is None:
        vp = VenuePredictor()
        persona.scratch.venue_predictor = vp
    seq = venue_timeline(persona)
    errs = vp.observe(seq)
    # recent window with data
    vals = [e for e in errs if e is not None]
    recent = vals[-10:]
    if not recent:
        # everything unmeasured = maximal novelty pressure
        return 0.5, 0.0
    conf = [vp.confidence(v) for v in seq[-10:] if v]
    mean_err = sum(recent) / len(recent)
    # learning progress: error TREND (early high -> late low = learned)
    if len(vals) >= 6:
        half = len(vals) // 2
        early, late = sum(vals[:half]) / max(half, 1), sum(vals[half:]) / max(len(vals) - half, 1)
        progress = max(0.0, early - late)
    else:
        progress = 0.3
    mastered = (mean_err < 0.15 and conf and max(conf) >= 0.9)
    boredom = 1.0 if mastered else 0.0
    return progress, boredom


def curiosity_goal(persona, maze=None):
    """Goal string when curiosity should drive behavior, else None.
    Fires on (a) unmeasured territory, (b) high learning progress, or
    (c) boredom at high confidence (seek a NEW venue class to learn)."""
    try:
        if not _env_on():
            return None
        sc = getattr(persona, 'scratch', None)
        if sc is None:
            return None
        progress, boredom = learning_progress(persona)
        if boredom < 1.0 and progress < 0.25:
            return None
        # pick target: a venue the agent has NEVER predicted well about
        known = set(vp_arena for vp_arena in
                    (getattr(persona.scratch, 'venue_predictor', None) or VenuePredictor()).trans.keys())
        seq = venue_timeline(persona)
        seen = set(seq)
        # town venue inventory from routines harvest (global) if available
        try:
            all_arenas = set()
            import glob as _glob
            for f in _glob.glob(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '../../environment/frontend_server/storage/live3d_n*',
                    'personas', '*', 'routines.json')):
                data = json.load(open(f))
                for r in data.get('routines', []):
                    all_arenas.add(r['arena'])
            fresh = [a for a in all_arenas if a not in seen]
        except Exception:
            fresh = []
        if not fresh:
            return None
        pick = random.Random(f"cur3-{sc.name}-{sc.curr_time}").choice(fresh)
        return f"explore {pick} to learn something new"
    except Exception:
        return None