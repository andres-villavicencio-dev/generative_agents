#!/usr/bin/env python3
"""Metrics for pipelines/chatfix_ab.py (control=chatfixctl, treatment=chatfixexp)."""
import collections, datetime, json, os, re, statistics, sys

REPO = "/home/andus/Projects/generative_agents"
ST = f"{REPO}/environment/frontend_server/storage"
FIRST = 17280
ARMS = {"control (legacy)": "chatfixctl", "treatment (fix)": "chatfixexp"}


def fresh_rows(sim, last):
    rows = []
    for st in range(FIRST, last + 1):
        try:
            d = json.load(open(f"{ST}/{sim}/movement/{st}.json"))
        except Exception:
            continue
        t = datetime.datetime.strptime(d["meta"]["curr_time"], "%B %d, %Y, %H:%M:%S")
        rows.append((st, t, {p: v["description"] for p, v in d["persona"].items()}))
    return rows


def frontier(sim):
    f = [int(x[:-5]) for x in os.listdir(f"{ST}/{sim}/movement") if x[:-5].isdigit()]
    return max(f)


def replan_stats(sim):
    pat = re.compile(r"\[ChatReplan\] (.+?) mode=(\w+) chat=(\d+)m window=(\d+)-(\d+)h "
                     r"slice_before=(\d+)m llm_total=(\d+)m slice_after=(\d+)m sched_total=(\d+)m")
    out = []
    try:
        for line in open(f"/tmp/{sim}_backend.log", errors="ignore"):
            m = pat.search(line)
            if m:
                out.append(dict(name=m.group(1), mode=m.group(2), chat=int(m.group(3)),
                                before=int(m.group(6)), llm=int(m.group(7)),
                                after=int(m.group(8)), total=int(m.group(9))))
    except FileNotFoundError:
        pass
    return out


def main():
    last = min(frontier(s) for s in ARMS.values())
    print(f"compared over fresh steps {FIRST}..{last}\n")
    res = {}
    for label, sim in ARMS.items():
        rows = fresh_rows(sim, last)
        r = {}
        rp = replan_stats(sim)
        r["chat replans logged"] = len(rp)
        if rp:
            dele = [x["before"] - x["after"] for x in rp]
            r["median minutes deleted per replan"] = statistics.median(dele)
            r["total hours deleted"] = round(sum(dele) / 60, 1)
            r["replans deleting >=60 min"] = f"{sum(1 for d in dele if d >= 60)}/{len(rp)}"
            r["median day length after splice (should be 1440)"] = statistics.median(x["total"] for x in rp)
        # behaviour on the clock
        nap = collections.defaultdict(set); lunch = {}; dinner = {}
        for st, t, ps in rows:
            for p, d in ps.items():
                a = d.split("@")[0].lower()
                if 8 <= t.hour < 12 and (a.startswith("sleeping") or "going to bed" in a):
                    nap[p].add(t.hour)
                if re.search(r"\blunch\b", a) and p not in lunch: lunch[p] = t.hour + t.minute / 60
                if re.search(r"\bdinner\b", a) and p not in dinner: dinner[p] = t.hour + t.minute / 60
        r["agents asleep/in bed between 08:00 and 12:00"] = len(nap)
        r["agents who start 'dinner' before 12:40"] = len(dinner)
        r["median hour of first 'lunch'"] = round(statistics.median(lunch.values()), 1) if lunch else None
        r["end clock"] = rows[-1][1].strftime("%b %d %H:%M") if rows else None
        # memory: chats vs planning thoughts created after the fork
        chats = plan = 0
        sched_bad = 0
        for p in os.listdir(f"{ST}/{sim}/personas"):
            b = f"{ST}/{sim}/personas/{p}/bootstrap_memory"
            n = json.load(open(f"{b}/associative_memory/nodes.json"))
            for v in n.values():
                if v["created"] > "2023-02-15 05:52:40":
                    if v["type"] == "chat": chats += 1
                    if v["type"] == "thought" and v["description"].startswith("For "): plan += 1
            try:
                sc = json.load(open(f"{b}/scratch.json"))
                tot = sum(int(d) for _, d in sc.get("f_daily_schedule", []))
                if tot != 1440: sched_bad += 1
            except Exception:
                pass
        r["chat memories (both sides)"] = chats
        r["planning thoughts written"] = plan
        r["agents whose saved day != 1440 min"] = sched_bad
        log = open(f"/tmp/{sim}_backend.log", errors="ignore").read()
        r["tracebacks"] = log.count("Traceback")
        r["LLM retries exhausted"] = log.count("All retries exhausted")
        res[label] = r
    keys = list(next(iter(res.values())))
    w = max(len(k) for k in keys)
    print(f"{'':{w}}  " + "  ".join(f"{l:>22}" for l in res))
    for k in keys:
        print(f"{k:{w}}  " + "  ".join(f"{str(res[l].get(k)):>22}" for l in res))
    json.dump(res, open(f"{REPO}/chatfix_ab_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
