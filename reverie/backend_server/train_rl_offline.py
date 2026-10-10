"""Build the offline artifacts for the RL project (run from backend_server).

Outputs:
  1. ga_traces/talk_policy.json  — trained TalkPolicy buckets (Tier 2)
  2. Per-persona warm-start VenueQ tables from a finished sim's real
     venue history (Tier 1) — written into that sim's scratch files so
     the next fork boots with pre-learned Q tables.

Trainer discipline: training data = ALL ga_traces talk records + the
teacher's talk harvest (jev_harvest decisions with q == 'talk'). The
referee is NEVER this data — it is the holdout split in the test suite
plus the gold talk blocks from decide_pairs.json when present.
"""
import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_modules import TalkPolicy, VenueQ
from routines import _arena_of

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
TRACES = os.path.join(ROOT, "ga_traces")
HARVEST = os.path.join(ROOT, "jev_harvest")


def load_talk_records():
    """All talk-decision records: live traces + teacher harvest talk rows."""
    recs = []
    p = os.path.join(TRACES, "talk.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                r = json.loads(line)
                # HARD-GATE records are deterministic gate fires (critical
                # need, state={who, critical_need}), not behavior-policy
                # decisions — including them would poison the offline
                # policy with a forced bucket. Skip them.
                if r.get("hard_gate"):
                    continue
                if r.get("kind") == "talk" or "choice" in r:
                    recs.append(r)
            except json.JSONDecodeError:
                pass
    for f in glob.glob(os.path.join(HARVEST, "decisions*.jsonl")):
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
                if r.get("q") == "talk":
                    # teacher harvest: {q, state, questions, answer}
                    ans = r.get("answer") or {}
                    # answer shape: {talk: {choice/probabilities...}}
                    t = ans.get("talk") or ans
                    choice = t.get("choice")
                    if choice is None and isinstance(t.get("probabilities"), dict):
                        pr = t["probabilities"]
                        choice = max(pr, key=pr.get)
                    if choice is None:
                        continue
                    recs.append({"state": r.get("state") or {},
                                 "choice": str(choice).lower(),
                                 "conf": t.get("confidence", 0.9)})
            except (json.JSONDecodeError, AttributeError):
                pass
    return recs


def main():
    print("== Tier 2: TalkPolicy training ==")
    recs = load_talk_records()
    print(f"   source records: {len(recs)} (live traces + teacher harvest)")
    tp = TalkPolicy()
    n = tp.train(recs)
    print(f"   trained on {n} usable records")
    buckets = {k: v for k, v in tp.buckets.items() if (v["yes"] + v["no"]) > 0}
    print(f"   buckets: {len(buckets)}")
    for k, v in sorted(buckets.items(), key=lambda kv: -(kv[1]["yes"] + kv[1]["no"]))[:6]:
        print(f"     {k}: yes={v['yes']} no={v['no']}")
    out = os.path.join(TRACES, "talk_policy.json")
    with open(out, "w") as f:
        json.dump(tp.to_dict(), f, indent=1)
    print(f"   wrote {out}")

    print("== Tier 1: VenueQ warm-start from finished sim ==")
    sim = sys.argv[1] if len(sys.argv) > 1 else "live3d_n60"
    sim_dir = os.path.join(ROOT, "environment/frontend_server/storage", sim)
    if not os.path.isdir(sim_dir):
        print(f"   sim {sim} not found — skipping warm-start")
        return
    # Venue sequences come from MOVEMENT files (descriptions carry
    # @-addresses), NOT memory nodes — event nodes record plain
    # sentences whose _arena_of extraction yields None (measured Oct 6:
    # 601/601 events unparseable on n60). Movement JSON: top-level
    # {persona: {<Name>: {description, ...}}, meta, artifacts}.
    import re as _re
    mv_files = sorted(glob.glob(os.path.join(sim_dir, "movement", "*.json")),
                      key=lambda p: int(os.path.basename(p)[:-5]))
    per_agent = {}
    for mf in mv_files:
        try:
            with open(mf) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        for name, pd in (d.get("persona") or {}).items():
            arena = _arena_of(str(pd.get("description", "")))
            per_agent.setdefault(name, []).append(arena)
    print(f"   movement files walked: {len(mv_files)}")
    personas_dir = os.path.join(sim_dir, "personas")
    for name, seq_raw in sorted(per_agent.items()):
        # collapse consecutive repeats: in-place actions dominate; the
        # Q-learner models VENUE TRANSITIONS, not in-place ticks
        seq = [a for i, a in enumerate(seq_raw) if a and (i == 0 or a != seq_raw[i-1])]
        if len(seq) < 5:
            print(f"   {name}: only {len(seq)} transitions — skipped")
            continue
        pd = os.path.join(personas_dir, name)
        scratch_p = os.path.join(pd, "bootstrap_memory", "scratch.json")
        if not os.path.exists(scratch_p):
            print(f"   {name}: no scratch file")
            continue
        try:
            sc = json.load(open(scratch_p))
        except json.JSONDecodeError:
            print(f"   !! corrupt scratch: {name}")
            continue
        vq = VenueQ()
        errs = vq.observe(seq)
        sc["venue_q"] = vq.to_dict()
        with open(scratch_p, "w") as f:
            json.dump(sc, f, indent=2)
        print(f"   {name}: {len(seq)} venue transitions, "
              f"{len(vq.Q)} arenas, TD mean {sum(errs)/len(errs):.3f}, "
              f"progress={vq.progress_signal():.2f}, boredom={vq.boredom_signal():.0f}")
    print("DONE")


if __name__ == "__main__":
    main()