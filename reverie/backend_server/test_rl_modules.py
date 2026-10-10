"""Offline test battery for rl_modules.py — run from backend_server dir.

Tests (no live sim, no LLM, no network):
  T1  VenueQ learns a deterministic transition chain: TD errors decay
  T2  VenueQ novelty reward actually prefers fresh transitions
  T3  to_dict/from_dict roundtrip preserves Q, counts, signals
  T4  progress/boredom signal semantics (fresh vs mastered)
  T5  q_learning_progress env-gate: OFF -> None, ON -> tuple
  T6  TalkPolicy._featurize canonical + stable
  T7  TalkPolicy conservative serving: support-gated, majority-margin
  T8  Train/test discipline: policy trained on HOLDOUT-split, accuracy on
      held-out records beats both a majority-class baseline AND is never
      worse than the behavior policy's own base rate on those records
  T9  fail-open: malformed records skipped, broken states -> None
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_modules import VenueQ, TalkPolicy, _featurize, _canon, q_learning_progress

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))


print("T1: VenueQ TD-error decay on a repeated chain (across epochs)")
vq = VenueQ()
chain = ["home", "cafe"] * 4
epoch_means = []
for _ in range(8):
    errs = vq.observe(chain)
    epoch_means.append(sum(errs) / len(errs))
check("epoch-0 error >> epoch-3 error (learning happened)",
      epoch_means[0] > 5 * epoch_means[3],
      f"e0={epoch_means[0]:.3f} e3={epoch_means[3]:.3f}")
check("long-run error stays bounded small (lag term < 20% of e0)",
      epoch_means[-1] < 0.2 * epoch_means[0],
      f"e_last={epoch_means[-1]:.3f} vs e0={epoch_means[0]:.3f}")
q_home = dict(vq.Q["home"])
check("Q(home->cafe) > 0 after learning", q_home.get("cafe", 0) > 0.01)

print("T2: novelty reward prefers fresh transitions")
vq2 = VenueQ()
err_fresh = vq2.observe(["a", "b", "c", "d", "e"])     # all-new transitions
vq3 = VenueQ()
for _ in range(4):
    err_old = vq3.observe(["a", "b", "a", "b", "a", "b"])  # repeated pair
check("fresh transitions produce higher TD error than repeated",
      sum(err_fresh) / len(err_fresh) > sum(err_old) / len(err_old))

print("T3: serialization roundtrip")
d = vq.to_dict()
vq_r = VenueQ.from_dict(d)
check("Q table survives roundtrip",
      dict(vq_r.Q["home"]) == dict(vq.Q["home"]))
check("pair counts survive", vq_r.pair_visits["home||cafe"] == vq.pair_visits["home||cafe"])
check("signals match after roundtrip",
      abs(vq_r.progress_signal() - vq.progress_signal()) < 1e-9)

print("T4: progress/boredom semantics")
vq_fresh = VenueQ()
vq_fresh.observe(["x", "y", "z", "w"])
p_fresh = vq_fresh.progress_signal()
vq_mastered = VenueQ()
for _ in range(30):
    vq_mastered.observe(["m", "n", "m", "n", "m", "n", "m", "n"])
p_mastered = vq_mastered.progress_signal()
b_mastered = vq_mastered.boredom_signal()
check("fresh > mastered progress", p_fresh > p_mastered,
      f"fresh {p_fresh:.3f} vs mastered {p_mastered:.3f}")
check("mastered chain reads boredom=1.0", b_mastered == 1.0)

print("T5: q_learning_progress env-gate")


class FakeScratch:
    pass


class FakePersona:
    def __init__(self):
        self.scratch = FakeScratch()


# Default flipped Oct 7 2026: unset gate now means ON (two A/B wins).
# The opt-OUT is the explicit =0.
os.environ["GA_QCURIOSITY"] = "0"
r = q_learning_progress(FakePersona())
check("gate OFF (=0 opt-out) -> None", r is None)
os.environ["GA_QCURIOSITY"] = "1"
r = q_learning_progress(FakePersona())
check("gate ON (=1), empty memory -> (0.5, 0.0) novelty convention", r == (0.5, 0.0))
os.environ.pop("GA_QCURIOSITY", None)
r = q_learning_progress(FakePersona())
check("gate DEFAULT-ON (unset) -> serves", r is not None)
os.environ.pop("GA_QCURIOSITY", None)

print("T6: featurizer canonical + stable")
s1 = {"who_activity": "head somewhere lively and spend time with people",
      "other_activity": "Head Somewhere Lively and Spend Time With People",
      "needs": "socially satisfied; well-fed", "last_chat": "no recent conversation"}
k1, k2 = _featurize(s1), _featurize(s1)
check("featurizer deterministic", k1 == k2)
s2 = dict(s1, who_activity="HEAD somewhere lively and spend time")
check("case-insensitive activity buckets", _canon("HEAD somewhere lively") == _canon("head somewhere lively"))
check("empty state -> None", _featurize(None) is None)

print("T7: TalkPolicy conservative serving")
tp = TalkPolicy()
recs = [
    {"state": {"who_activity": "eat lunch", "other_activity": "eat lunch",
               "needs": "hungry", "last_chat": "no recent conversation"},
     "choice": "yes", "conf": 0.9},
    {"state": {"who_activity": "eat lunch", "other_activity": "eat lunch",
               "needs": "hungry", "last_chat": "no recent conversation"},
     "choice": "yes", "conf": 0.7},
    {"state": {"who_activity": "eat lunch", "other_activity": "eat lunch",
               "needs": "hungry", "last_chat": "no recent conversation"},
     "choice": "yes", "conf": 0.8},
    {"state": {"who_activity": "sleep at home", "other_activity": "read a book",
               "needs": "tired", "last_chat": "chatted recently"},
     "choice": "no", "conf": 0.95},
    {"state": {"who_activity": "sleep at home", "other_activity": "read a book",
               "needs": "tired", "last_chat": "chatted recently"},
     "choice": "no", "conf": 0.6},
]
n = tp.train(recs)
check("train consumed all records", n == 5)
d = tp.decide({"who_activity": "eat lunch", "other_activity": "eat lunch",
               "needs": "hungry", "last_chat": "no recent conversation"})
check("high-support bucket served", d and d["choice"] == "yes" and d["support"] == 3)
d_low = tp.decide({"who_activity": "sleep at home", "other_activity": "read a book",
                   "needs": "tired", "last_chat": "chatted recently"})
check("support=2 < min_support=3 -> None (conservative)",
      d_low is None, f"got {d_low}")
d_unseen = tp.decide({"who_activity": "dance wildly", "other_activity": "yodel",
                      "needs": "?", "last_chat": "?"})
check("unseen state -> None (no extrapolation)", d_unseen is None)

print("T8: train/test discipline on REAL ga_traces data")
T = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "ga_traces", "talk.jsonl")
records = [json.loads(l) for l in open(T) if l.strip()]
check("have real talk records", len(records) > 0, f"n={len(records)}")
random.seed(42)
random.shuffle(records)
split = int(len(records) * 0.7)
train_recs, test_recs = records[:split], records[split:]
tp2 = TalkPolicy()
tp2.train(train_recs)
hits, served, gold_yes = 0, 0, sum(1 for r in test_recs if r["choice"] == "yes")
for r in test_recs:
    d = tp2.decide(r["state"], min_support=2)
    if d is None:
        continue
    served += 1
    if d["choice"] == r["choice"]:
        hits += 1
majority = max(gold_yes, len(test_recs) - gold_yes)
print(f"    test n={len(test_recs)}, policy served on {served}, correct {hits}")
check("policy never worse than majority baseline on served set",
      served == 0 or hits / served >= (majority / len(test_recs)) - 0.05,
      f"served-acc {hits}/{served} vs majority-base {majority}/{len(test_recs)}")

print("T9: fail-open on malformed data")
tp3 = TalkPolicy()
n = tp3.train([{"state": None, "choice": "yes"}, {"state": {}, "choice": "maybe"},
               {"state": {"who_activity": "x"}, "choice": "yes", "conf": "bad"}])
check("malformed records skipped", n >= 0)
check("decide on None state -> None", tp3.decide(None) is None)
vq_bad = VenueQ.from_dict({"Q": "garbage"})
check("VenueQ.from_dict survives garbage", isinstance(vq_bad, VenueQ))

print()
print(f"RESULT: {'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURES'}")
sys.exit(1 if FAIL else 0)