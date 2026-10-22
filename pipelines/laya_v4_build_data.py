#!/usr/bin/env python3
"""Build the Laya v4 training/eval set from the m3.1 teacher harvest (blind labels).

Input : snapshot copies of ga_traces/m31_teacher/teacher_{poignancy,talk,react}.jsonl
        (copied into OUT_DIR first; the live files keep growing).
Output: OUT_DIR/{train,calib,test,offdiag_persona,offdiag_time}.jsonl
        one line per UNIQUE decision state: {kind, persona, t, state, questions,
        target (soft distribution over options, in option order), label (teacher
        argmax index), n_votes}

Split (persona- AND time-disjoint):
  test  = persona in TEST_PERSONAS and t >= T_CUT
  train = persona not in TEST_PERSONAS and t < T_CUT   (10% of it -> calib slice)
  off-diagonal blocks are never trained on; kept as secondary eval sets.

Soft targets: each teacher vote puts `confidence` on its answer; the rest goes to
the +-1 neighbours (score) or the other option (choice). Votes on an identical
state are averaged (the harvest re-labelled repeated states at temperature 0.4).
"""
import collections
import datetime
import json
import os
import random
import shutil
import sys

SRC = "/home/andus/Projects/generative_agents/ga_traces/m31_teacher"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/andus/Projects/laya-test/v4_data"
T_CUT = datetime.datetime(2026, 10, 21, 20, 0).timestamp()
TEST_PERSONAS = None  # chosen by search below if None
SEED = 20261022

RUBRIC_INS = ("Rate the likely poignancy of the event for this person, considering "
              "their identity and circumstances.")
TALK_INS = ("Should the first agent initiate a conversation with the other agent right "
            "now, or continue their own activity? Weigh the person's current needs as stated.")
REACT_INS = ("Two agents are heading toward the same venue in a small town. Should the "
             "first agent wait for the second agent to finish their activity, or continue "
             "with their own plan?")


def load(kind, snap=True):
    src = os.path.join(SRC, f"teacher_{kind}.jsonl")
    dst = os.path.join(OUT_DIR, f"teacher_{kind}.jsonl")
    if snap and not os.path.exists(dst):
        shutil.copy(src, dst)
    rows = []
    for line in open(dst):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def vote_dist(kind, rec, keys):
    """One teacher vote -> distribution over option indices, or None if unusable."""
    t = rec.get("teacher") or {}
    a, c = t.get("answer"), t.get("confidence")
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = 0.8
    c = min(0.99, max(0.34, c))
    k = len(keys)
    if kind == "poignancy":
        try:
            a = int(a)
        except (TypeError, ValueError):
            return None
        if not 1 <= a <= 10:
            return None
        i = a - 1
        d = [0.0] * k
        d[i] = c
        nb = [j for j in (i - 1, i + 1) if 0 <= j < k]
        for j in nb:
            d[j] += (1 - c) / len(nb)
        return d
    a = str(a).strip()
    if a in keys:
        i = keys.index(a)
    else:
        opts = rec["options"]
        hit = [j for j, kk in enumerate(keys) if a == opts[kk] or a.startswith(kk + ":")
               or opts[kk].startswith(a) or a.startswith(opts[kk][:40])]
        if not hit and kind == "react":
            hit = [keys.index("1")] if a.lower().startswith(("wait", "1")) else \
                  [keys.index("2")] if a.lower().startswith(("continue", "2")) else []
        if not hit and kind == "talk":
            hit = [keys.index(a.lower())] if a.lower() in keys else []
        if len(hit) != 1:
            return None
        i = hit[0]
    d = [(1 - c) / (k - 1)] * k
    d[i] = c
    return d


def questions_for(kind, rec):
    if kind == "poignancy":
        q = rec["questions"]["poignancy"]
        return {"poignancy": {"type": "score", "instructions": q["instructions"],
                              "criteria": list(q["criteria"])}}
    ins = TALK_INS if kind == "talk" else REACT_INS
    return {kind: {"type": "choice", "instructions": ins, "criteria": dict(rec["options"])}}


def build_items(kind, rows):
    groups = collections.OrderedDict()
    dropped = collections.Counter()
    for r in rows:
        if not r.get("teacher") or r["teacher"].get("answer") is None:
            dropped["no_teacher"] += 1
            continue
        if kind != "poignancy" and not r.get("options"):
            dropped["no_options"] += 1
            continue
        qs = questions_for(kind, r)
        crit = qs[kind if kind != "poignancy" else "poignancy"]["criteria"]
        keys = list(range(len(crit))) if kind == "poignancy" else list(crit.keys())
        d = vote_dist(kind, r, keys)
        if d is None:
            dropped["unparsed_answer"] += 1
            continue
        # sim-side state fields are truncated to 600 chars by _typed_choice; mirror it
        state = {k: (v[:600] if isinstance(v, str) else v) for k, v in r["state"].items()}
        key = json.dumps([state, qs], sort_keys=True)
        g = groups.setdefault(key, {"kind": kind, "persona": r["persona"], "t": r["t"],
                                    "state": state, "questions": qs, "votes": [],
                                    "sim_choice": r.get("sim_choice")})
        g["votes"].append(d)
        g["t"] = min(g["t"], r["t"])
    items = []
    for g in groups.values():
        k = len(g["votes"][0])
        tgt = [sum(v[i] for v in g["votes"]) / len(g["votes"]) for i in range(k)]
        g["target"] = tgt
        g["label"] = max(range(k), key=lambda i: tgt[i])
        g["n_votes"] = len(g.pop("votes"))
        items.append(g)
    return items, dropped


def choose_test_personas(items_by_kind, n=6, tries=4000):
    personas = sorted({it["persona"] for its in items_by_kind.values() for it in its})
    rng = random.Random(SEED)
    best = None
    for _ in range(tries):
        P = set(rng.sample(personas, n))
        fr = {}
        for kind, its in items_by_kind.items():
            tot = len(its)
            te = sum(1 for it in its if it["persona"] in P and it["t"] >= T_CUT)
            fr[kind] = te / tot
        # want every kind's test share in [0.10, 0.16]; minimise deviation from 0.12
        score = sum(abs(f - 0.12) + (5 if f < 0.10 else 0) for f in fr.values())
        if best is None or score < best[0]:
            best = (score, sorted(P), fr)
    return best


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    items_by_kind, report = {}, {}
    for kind in ("poignancy", "talk", "react"):
        rows = load(kind)
        items, dropped = build_items(kind, rows)
        items_by_kind[kind] = items
        report[kind] = {"raw_rows": len(rows), "unique_states": len(items), "dropped": dict(dropped)}
    if TEST_PERSONAS:
        P = set(TEST_PERSONAS)
    else:
        score, P, fr = choose_test_personas(items_by_kind)
        P = set(P)
    splits = collections.defaultdict(list)
    for kind, its in items_by_kind.items():
        for it in its:
            inP, late = it["persona"] in P, it["t"] >= T_CUT
            name = ("test" if inP and late else "train" if not inP and not late
                    else "offdiag_persona" if inP else "offdiag_time")
            splits[name].append(it)
    rng = random.Random(SEED)
    tr = splits["train"]
    rng.shuffle(tr)
    n_cal = min(400, len(tr) // 10)
    splits["calib"], splits["train"] = tr[:n_cal], tr[n_cal:]
    for name, its in splits.items():
        with open(os.path.join(OUT_DIR, f"{name}.jsonl"), "w") as f:
            for it in its:
                f.write(json.dumps(it) + "\n")
        report.setdefault("splits", {})[name] = dict(collections.Counter(it["kind"] for it in its))
    report["test_personas"] = sorted(P)
    report["t_cut"] = datetime.datetime.fromtimestamp(T_CUT).isoformat()
    json.dump(report, open(os.path.join(OUT_DIR, "build_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
