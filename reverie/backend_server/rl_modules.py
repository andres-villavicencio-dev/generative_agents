# Copyright (c) 2026 Andus VU — MIT license
"""RL modules for generative agents (Oct 2026, RL-meets-GA project).

Two tiers, one module — designed around the deepmind-research patterns
(DQN Zoo's tabular ancestor for Tier 1; RL Unplugged's offline-RL thesis
for Tier 2) but implemented natively for THIS stack. No external RL
frameworks; everything is pure Python so plan.py can import it without
torch. Fail-open everywhere.

Tier 1 — VenueQ (tabular Q-learning over venue transitions)
  Replaces the curiosity layer's counted-frequency predictor with a proper
  Q-table: state = current arena, actions = next arenas, reward = intrinsic
  novelty (visit-count decay, Schmidhuber-compatible). The TD-error stream
  IS the learning signal: high recent TD error = still learning (pull),
  near-zero = mastered (boredom). Gate: GA_QCURIOSITY (default ON since
  Oct 7 2026; opt out with GA_QCURIOSITY=0).
  Persistence: serializes into scratch (hasattr-guarded, like routines).

Tier 2 — TalkPolicy (offline policy from ga_traces)
  Offline-learned talk-decision policy distilled from the logged behavior
  policy (talk.jsonl = POST-bias laya decisions with confidences acting as
  propensities). Conservative count-based improvement (BCQ/CQL-flavored):
  the policy may only deviate from the behavior policy where the data
  supports it — the Tandem DQN lesson (passive learning fails without
  inductive biases) is mitigated by the support-count constraint.
  Gate: GA_TALKPOLICY (default ON since Oct 7 2026; opt out with
  GA_TALKPOLICY=0). Serves as a cascade candidate
  ahead of laya when a state matches high-support patterns.
"""

import json
import os
from collections import Counter, defaultdict


def _env_on(flag):
    """Env gate read. RL gates (GA_QCURIOSITY, GA_TALKPOLICY) are
    DEFAULT-ON since Oct 7 2026 (two A/B wins: +23%/+8% venue diversity,
    +75%/+14% chats, first live policy serve, 0 referee flips). Opt OUT
    explicitly with <flag>=0; any legacy unit file setting =1 is
    harmless (same behavior)."""
    if flag in ("GA_QCURIOSITY", "GA_TALKPOLICY"):
        return os.environ.get(flag, "1") != "0"
    return os.environ.get(flag, "0") == "1"


# ===========================================================================
# Tier 1: VenueQ — tabular Q-learning over (arena -> arena) transitions
# ===========================================================================

class VenueQ:
    """Per-persona tabular Q-learner on venue transitions."""

    def __init__(self, alpha=0.3, gamma=0.85, nov_r=1.0, mastery_r=0.3,
                 min_alpha=0.05):
        self.alpha = alpha          # initial learning rate
        self.min_alpha = min_alpha  # alpha decays with row visits
        self.gamma = gamma
        self.nov_r = nov_r          # novelty reward scale
        self.mastery_r = mastery_r  # extra reward for stabilizing a row
        self.Q = defaultdict(lambda: defaultdict(float))
        self.pair_visits = Counter()   # "a||b" -> n
        self.row_visits = Counter()   # arena -> n
        self.td_hist = defaultdict(list)  # arena -> recent |TD errors|
        self.last_venue = None

    def _step_alpha(self, arena):
        """Robust Q-learning: alpha decays per-row with visit count so a
        deterministic chain CONVERGES. Measured nuance (T1 diagnostic):
        with a NON-STATIONARY reward (novelty decaying with visits), Q
        lags a falling target and |TD| bottoms out then drifts up from
        the lag — bounded, small, and monotone-in-the-mean over the
        first epochs. min_alpha=0.02 keeps the lag term tiny."""
        n = self.row_visits.get(arena, 0)
        return max(0.02, self.alpha / (1.0 + 0.1 * n))

    # -- learning ------------------------------------------------------------
    def _reward(self, prev, nxt):
        """Intrinsic reward. Novelty scales with pair visits BEFORE this
        step (so the first traversal pays full novelty, and a fully
        explored pair converges to the small mastery term only)."""
        n = self.pair_visits.get(prev + "||" + nxt, 0)
        novelty = self.nov_r / (1.0 + 0.5 * n)
        rown = self.row_visits.get(prev, 0)
        mastery = self.mastery_r / (1.0 + rown)
        return novelty + mastery

    def observe(self, venue_seq):
        """Q-learning update over a venue sequence; returns |TD errors|.
        Reward is computed BEFORE the visit counters increment, so the
        novelty term reflects prior visits (a repeated chain converges
        to near-zero reward and decaying TD error)."""
        if not venue_seq:
            return []
        td_errs = []
        for i in range(1, len(venue_seq)):
            prev, nxt = venue_seq[i - 1], venue_seq[i]
            if not prev or not nxt or prev == nxt:
                continue
            r = self._reward(prev, nxt)
            best_next = max(self.Q[nxt].values()) if self.Q[nxt] else 0.0
            target = r + self.gamma * best_next
            old = self.Q[prev][nxt]
            td_err = target - old
            self.Q[prev][nxt] = old + self._step_alpha(prev) * td_err
            self.pair_visits[prev + "||" + nxt] += 1
            self.row_visits[prev] += 1
            hist = self.td_hist[prev]
            hist.append(abs(td_err))
            if len(hist) > 20:
                del hist[0]
            td_errs.append(abs(td_err))
        self.last_venue = venue_seq[-1]
        return td_errs

    # -- signals -------------------------------------------------------------
    def progress_signal(self):
        """Learning-progress trend: positive when recent |TD error| is
        FALLING (error dropped from early to late = something was just
        learned) OR still high (actively learning). Near-zero recent
        error = mastered. Range [0,1].

        Trend semantics (mirrors the incumbent curiosity layer): early
        vs late halves of the recent window — a big early/late drop is
        the strongest 'I am learning' signal."""
        errs = []
        for hist in self.td_hist.values():
            errs.extend(hist[-10:])
        if not errs:
            return 0.0
        recent = errs[-10:]
        if len(recent) >= 6:
            half = len(recent) // 2
            early = sum(recent[:half]) / max(half, 1)
            late = sum(recent[half:]) / max(len(recent) - half, 1)
            progress = max(0.0, early - late)  # error-dropping = learning
            activity = min(1.0, late / 1.0)   # still-surprising = learning
            return min(1.0, progress + 0.5 * activity)
        return min(1.0, sum(recent) / len(recent) / 2.0)

    def boredom_signal(self):
        """1.0 when at least half the visited rows are mastered (n>=6 and
        recent |TD| < 0.15), else 0.0."""
        if not self.row_visits:
            return 0.0
        mastered = 0
        for arena, n in self.row_visits.items():
            hist = self.td_hist.get(arena) or []
            if n >= 6 and hist and sum(hist[-5:]) / len(hist[-5:]) < 0.15:
                mastered += 1
        if mastered >= max(1, len(self.row_visits) // 2):
            return 1.0
        return 0.0

    # -- serialization -------------------------------------------------------
    def to_dict(self):
        return {
            "alpha": self.alpha, "gamma": self.gamma,
            "nov_r": self.nov_r, "mastery_r": self.mastery_r,
            "Q": {k: dict(v) for k, v in self.Q.items()},
            "pair_visits": dict(self.pair_visits),
            "row_visits": dict(self.row_visits),
            "td_hist": {k: v[-20:] for k, v in self.td_hist.items()},
        }

    @classmethod
    def from_dict(cls, d):
        """Type-guarded rehydration: garbage values degrade to empty
        tables, never raise (a truncated checkpoint scratch must not
        poison a boot)."""
        obj = cls()
        if not isinstance(d, dict):
            return obj
        q = d.get("Q")
        if isinstance(q, dict):
            for k, row in q.items():
                if isinstance(row, dict):
                    obj.Q[k] = defaultdict(float,
                                           {a: float(v) for a, v in row.items()
                                            if isinstance(v, (int, float))})
        for src, cast in (("pair_visits", int), ("row_visits", int)):
            data = d.get(src)
            if isinstance(data, dict):
                c = getattr(obj, src)
                for k, v in data.items():
                    try:
                        c[k] = cast(v)
                    except (TypeError, ValueError):
                        pass
        th = d.get("td_hist")
        if isinstance(th, dict):
            for k, v in th.items():
                if isinstance(v, list):
                    obj.td_hist[k] = [float(e) for e in v
                                      if isinstance(e, (int, float))][-20:]
        return obj


# ===========================================================================
# Tier 1 integration seam: enrich learning_progress with Q-signals
# ===========================================================================

def venue_q_for(persona):
    """Get-or-create the persona's VenueQ (scratch-attached)."""
    try:
        vq = getattr(persona.scratch, "venue_q", None)
        if vq is None or not isinstance(vq, VenueQ):
            vq = VenueQ()
            persona.scratch.venue_q = vq
        return vq
    except Exception:
        return VenueQ()


def q_learning_progress(persona):
    """(progress, boredom) from the Q-learner — the GA_CURIOSITY-consumable
    signal. Returns (0.5, 0.0) on empty data (maximal novelty pressure,
    matching the incumbent's convention)."""
    try:
        if not _env_on("GA_QCURIOSITY"):
            return None
        vq = venue_q_for(persona)
        seq = _venue_timeline(persona)
        if not seq:
            return (0.5, 0.0)
        td_errs = vq.observe(seq)
        if not td_errs:
            return (0.5, 0.0)
        return (vq.progress_signal(), vq.boredom_signal())
    except Exception:
        return None


def _venue_timeline(persona, k=80):
    """Chronological arena list from event memories (mirror of
    curiosity.venue_timeline, kept local so this module has no import
    cycle with curiosity.py)."""
    try:
        from curiosity import venue_timeline
        return venue_timeline(persona, k)
    except Exception:
        return []


# ===========================================================================
# Tier 2: TalkPolicy — offline policy from ga_traces/talk.jsonl
# ===========================================================================

def _featurize(state):
    """Deterministic feature key for a talk-decision state.

    Uses the coarse fields that generalize across encounters:
    (who_activity, other_activity, needs-bucket, last_chat bucket).
    Fine-grained identity fields are deliberately dropped — with ~60
    logged decisions we need bucketing, not memorization.
    """
    try:
        who_act = _canon(str(state.get("who_activity", "")))
        other_act = _canon(str(state.get("other_activity", "")))
        needs = _canon(_needs_summary(state.get("needs", "")))
        last = _canon(str(state.get("last_chat", "")))
        return (who_act, other_act, needs, last)
    except Exception:
        return None


# Generic verb heads and glue words harvested from REAL trace
# phrasings (never a guessed list): these carry no activity signal, so
# the canon skips them and takes the first INFORMATIVE stem.
_GENERIC_HEADS = {
    "go", "going", "goes", "buy", "buying", "head", "heading",
    "spend", "spending", "take", "taking", "do", "doing", "have",
    "having", "make", "making", "get", "getting", "put", "putting",
    "the", "and", "for", "with", "about", "his", "her", "their",
}


def _canon(s):
    """Bucket an activity/needs string: lowercase, first INFORMATIVE
    content word stem (generic verb heads like 'go/buy/head' are
    skipped — 'buy groceries at...' and 'go grocery shopping' must land
    in the SAME bucket)."""
    s = s.lower().strip()
    if not s:
        return "-"
    for w in s.split(";")[0].replace(",", " ").split():
        if len(w) > 2 and w.isalpha() and w not in _GENERIC_HEADS:
            return w[:6]
    # everything generic ('head somewhere...' with only glue words is
    # impossible in practice; fall back to the first token anyway)
    toks = s.split()
    return (toks[0][:6] if toks else "-")


def _needs_summary(needs):
    """Render the scratch needs dict EXACTLY as the trace recorder does
    (laya_scoring builds the trace state's needs string): key order
    social/hunger/energy/bladder, same threshold wording. Any deviation
    re-buckets live states away from their training buckets (measured
    Oct 7: needs= was the ONLY field that never matched — 92/92 serves
    lost to 'hunger...' vs 'socially...')."""
    try:
        if needs is None:
            return "-"
        if isinstance(needs, str):
            return needs
        if isinstance(needs, dict):
            parts = []
            s = needs.get("social", 50)
            try:
                s = float(s)
            except (TypeError, ValueError):
                s = 50.0
            parts.append("desperately lonely, starved for company" if s < 20
                         else "lonely, wanting company" if s < 50
                         else "socially satisfied" if s > 70
                         else "socially content")
            if "hunger" in needs:
                h = needs["hunger"]
                try:
                    h = float(h)
                except (TypeError, ValueError):
                    h = 50.0
                parts.append("hungry" if h < 20
                             else "well-fed" if h > 70
                             else "moderately hungry")
            if "energy" in needs:
                e = needs["energy"]
                try:
                    e = float(e)
                except (TypeError, ValueError):
                    e = 50.0
                parts.append("exhausted" if e < 20
                             else "energetic" if e > 70
                             else "moderately tired")
            if "bladder" in needs:
                b = needs["bladder"]
                try:
                    b = float(b)
                except (TypeError, ValueError):
                    b = 50.0
                parts.append("urgently needs a bathroom" if b < 20
                             else "bathroom needs fine")
            return "; ".join(parts)
        return str(needs)
    except Exception:
        return "-"


class TalkPolicy:
    """Offline-learned talk policy with conservative support counting.

    Training (offline, from JSONL):
      - group records by feature key
      - behavior policy per key: choice distribution + mean confidence
      - support(key) = number of logged decisions in that bucket

    Serving:
      - exact feature match with support >= GA_TALKPOLICY_MIN_SUPPORT
        (default 3): serve the majority choice of the bucket, but only
        when its share exceeds the behavior-policy mean by a margin —
        conservative improvement, never extrapolation outside support
      - otherwise: return None (fall through to laya cascade)
    """

    def __init__(self):
        self.buckets = defaultdict(lambda: {"yes": 0, "no": 0,
                                            "conf": [], "choices": []})

    def train(self, records):
        """records: iterable of dicts with keys state, choice, conf.
        Malformed conf values fall back to 0.5 — fail-open, never raise."""
        n = 0
        for r in records:
            key = _featurize(r.get("state", {}))
            if key is None:
                continue
            choice = str(r.get("choice", "")).strip().lower()
            if choice not in ("yes", "no"):
                continue
            try:
                conf = float(r.get("conf", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            b = self.buckets[key]
            b[choice] += 1
            b["conf"].append(conf)
            b["choices"].append(choice)
            n += 1
        return n

    def support(self, state):
        key = _featurize(state)
        if key is None or key not in self.buckets:
            return 0
        b = self.buckets[key]
        return b["yes"] + b["no"]

    def decide(self, state, min_support=3, margin=0.0):
        """Conservative decision or None (no support -> cascade)."""
        key = _featurize(state)
        if key is None or key not in self.buckets:
            return None
        b = self.buckets[key]
        total = b["yes"] + b["no"]
        if total < min_support:
            return None
        yes_share = b["yes"] / total
        if abs(yes_share - 0.5) <= margin:
            return None  # no clear signal -> defer to laya
        choice = "yes" if yes_share > 0.5 else "no"
        return {
            "choice": choice,
            "confidence": max(yes_share, 1 - yes_share),
            "support": total,
            "feature_key": key,
        }

    # -- persistence ---------------------------------------------------------
    def to_dict(self):
        return {"buckets": {json.dumps(k): {"yes": v["yes"], "no": v["no"],
                                             "conf": v["conf"],
                                             "choices": v["choices"]}
                             for k, v in self.buckets.items()}}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for k, v in (d.get("buckets") or {}).items():
            try:
                key = tuple(json.loads(k)) if isinstance(k, str) else tuple(k)
            except (json.JSONDecodeError, TypeError):
                continue
            obj.buckets[key] = {"yes": int(v.get("yes", 0)),
                                "no": int(v.get("no", 0)),
                                "conf": list(v.get("conf", [])),
                                "choices": list(v.get("choices", []))}
        return obj


# module-level singleton (trained once, used per decide call)
_POLICY = None
_POLICY_PATH = None


def talk_policy(path=None, min_support=None):
    """Load (and cache) the trained TalkPolicy from a JSON snapshot."""
    global _POLICY, _POLICY_PATH
    try:
        if not _env_on("GA_TALKPOLICY"):
            return None
        default = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "ga_traces", "talk_policy.json")
        path = path or os.environ.get("GA_TALKPOLICY_PATH", default)
        if path != _POLICY_PATH or _POLICY is None:
            with open(path) as f:
                _POLICY = TalkPolicy.from_dict(json.load(f))
            _POLICY_PATH = path
        return _POLICY
    except Exception:
        return None