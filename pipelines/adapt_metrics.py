#!/usr/bin/env python3
"""Adaptability metrics for a finished sim.

Usage: venv/bin/python pipelines/adapt_metrics.py <sim> [<sim> ...] [--from-step N]
Reads the sim's movement files and /tmp/<sim>_backend.log (same sources as
chatfix_ab_metrics.py). Rates are per sim-day so runs of different length compare.

Metrics
  repeated_failure_rate  depletion events for a (persona, item) that already had one
                         <24 sim-h earlier / all depletion events
  goal_top1_share        share of the most common [Drives] goal string
  goal_distinct          distinct drive-injected goal strings
  chat_share             persona-minutes spent "conversing" / all persona-minutes
  chat_longest_min       longest unbroken "conversing" stretch of any persona
  job_presence           (when scratch.job exists) share of shift minutes at the venue
"""
import collections, datetime, json, os, re, sys

REPO = "/home/andus/Projects/generative_agents"
ST = f"{REPO}/environment/frontend_server/storage"
_DEPL = re.compile(r"resource_depleted_([A-Za-z ]+?)_([a-z_]+)_(\d{14})")
_DRIVE = re.compile(r"\[Drives\] (.+?) drive=(\w+) -> '(.*)'")


def steps(sim, first):
    f = sorted(int(x[:-5]) for x in os.listdir(f"{ST}/{sim}/movement") if x[:-5].isdigit())
    return [s for s in f if s >= first]


def depletion_events(log):
    """Distinct (persona, item, time) depletion events (the log repeats each key)."""
    seen = set()
    for line in open(log, errors="ignore"):
        for m in _DEPL.finditer(line):
            seen.add((m.group(1), m.group(2),
                      datetime.datetime.strptime(m.group(3), "%Y%m%d%H%M%S")))
    return sorted(seen, key=lambda e: e[2])


def repeated_failures(events, window_h=24):
    last, rep = {}, 0
    for p, item, t in events:
        prev = last.get((p, item))
        if prev is not None and (t - prev).total_seconds() < window_h * 3600:
            rep += 1
        last[(p, item)] = t
    return rep


def goals(log):
    c = collections.Counter()
    for line in open(log, errors="ignore"):
        m = _DRIVE.search(line)
        if m:
            c[m.group(3)] += 1
    return c


def chats(sim, first):
    total = chat = longest = 0
    run = collections.Counter()
    for st in steps(sim, first)[::6]:  # 1-minute resolution
        try:
            d = json.load(open(f"{ST}/{sim}/movement/{st}.json"))
        except Exception:
            continue
        for p, v in d["persona"].items():
            total += 1
            if "conversing" in v["description"].lower():
                chat += 1
                run[p] += 1
                longest = max(longest, run[p])
            else:
                run[p] = 0
    return chat / max(total, 1), longest


def economy_stats(sim):
    """Poverty-trap and community-fund numbers from the live sim state."""
    out = {}
    try:
        w = []
        for d in os.listdir(f"{ST}/{sim}/personas"):
            sc = json.load(open(f"{ST}/{sim}/personas/{d}/bootstrap_memory/scratch.json"))
            w.append(sc.get("wallet") or 0)
        out["agents_under_30"] = sum(1 for x in w if x < 30)
        out["median_wallet"] = round(sorted(w)[len(w) // 2], 1)
        e = json.load(open(f"{ST}/{sim}/resources/economy_v2.json"))
        f = e.get("fund") or {}
        out.update(fund_balance=f.get("balance"), fund_in=f.get("in"), fund_out=f.get("out"),
                   community_paid=round(sum(x["amount"] for x in e.get("pay_log", [])
                                            if x["kind"] == "community"), 2))
    except Exception:
        pass
    return out


def report(sim, first):
    log = f"/tmp/{sim}_backend.log"
    st = steps(sim, first)
    first_t = datetime.datetime.strptime(
        json.load(open(f"{ST}/{sim}/movement/{st[0]}.json"))["meta"]["curr_time"], "%B %d, %Y, %H:%M:%S")
    last_t = datetime.datetime.strptime(
        json.load(open(f"{ST}/{sim}/movement/{st[-1]}.json"))["meta"]["curr_time"], "%B %d, %Y, %H:%M:%S")
    days = max((last_t - first_t).total_seconds() / 86400, 1e-9)
    ev = [e for e in depletion_events(log) if e[2] >= first_t]
    g = goals(log)
    share, longest = chats(sim, first)
    by = collections.Counter(p for p, _, _ in ev)
    return {
        "sim": sim, "sim_days": round(days, 2),
        "depletion_events": len(ev), "depletion_per_day": round(len(ev) / days, 1),
        "repeated_failure_rate": round(repeated_failures(ev) / max(len(ev), 1), 3),
        "worst_personas": by.most_common(3),
        "goal_distinct": len(g),
        "goal_top1_share": round(g.most_common(1)[0][1] / sum(g.values()), 3) if g else None,
        "chat_share": round(share, 3), "chat_longest_min": longest,
        **economy_stats(sim),
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    first = 0
    if "--from-step" in args:
        i = args.index("--from-step")
        first = int(args[i + 1])
        del args[i:i + 2]
    for sim in args:
        print(json.dumps(report(sim, first)))
