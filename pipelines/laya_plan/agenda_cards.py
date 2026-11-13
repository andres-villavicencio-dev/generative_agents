#!/usr/bin/env python3
"""Synthetic agenda-bearing free-choice cards for Laya 'choose' (v11).

Each card is built with the REAL prompt builder (run_gpt_prompt_choose_activity) from a real
persona's scratch + an agenda, so Laya trains on exactly the text it sees live. Events are
generic (dinners, meetings, appointments, parties, shifts...) - nothing about any one event.
Labels come later from cloud teachers only (agenda_label.py). Output:
  ga_traces/agenda_choice/cards.jsonl  {key, persona, t, prompt, meta}
Usage: agenda_cards.py [n_cards] [seed]"""
import datetime, hashlib, json, os, random, sys, types
os.environ["GA_CHOICE_LAYA"] = "0"
REPO = "/home/andus/Projects/generative_agents"
sys.path.insert(0, f"{REPO}/reverie/backend_server")
import agenda as ag
import persona.prompt_template.run_gpt_prompt as rgp
from persona.memory_structures.scratch import Scratch

BASE = f"{REPO}/environment/frontend_server/storage/base_the_ville_n55/personas"
OUT = f"{REPO}/ga_traces/agenda_choice/cards.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 20261111)

PLACES = ["Hobbs Cafe", "The Rose and Crown Pub", "Johnson Park", "The Willows Market and Pharmacy",
          "Oak Hill College", "Willow Clinic", "Harvey Oak Supply Store", "the community hall"]
SOCIAL = ["dinner with {f}", "coffee with {f}", "{f}'s birthday gathering", "a neighborhood potluck",
          "open mic night", "a Valentine's Day party", "an art walk", "a book club meeting",
          "a game night", "a farewell party for {f}", "a picnic with {f}", "a poetry reading"]
CIVIC = ["a town hall meeting", "a community garden workshop", "a neighborhood cleanup", "a mayoral candidates' forum"]
PERSONAL = ["a dentist appointment", "a meeting with {f} about the project", "a piano lesson", "a study group",
            "a job interview", "picking up a parcel", "a fitting at the tailor"]
WORK = ["covering a shift for {f}", "the staff meeting", "a delivery handover"]
PLAN_ITEMS = ["working a shift at Hobbs Cafe", "reading at home", "having lunch", "relaxing at home",
              "studying for exams", "painting in the studio", "tidying up the house", "taking a walk in the park",
              "resting after work", "grocery shopping", "writing at the desk", "cooking dinner",
              "working at The Willows Market and Pharmacy", "practicing the piano", "browsing the bulletin board",
              "doing laundry", "attending class at Oak Hill College", "sketching in Johnson Park"]
PULLS = ["buying groceries", "working a shift at The Willows Market and Pharmacy",
         "having a drink with friends at The Rose and Crown Pub", "community cleanup at Johnson Park"]
NOTES_BY_SOURCE = {"chat": ["invited by {f}", "{f} asked you to come", "you told {f} you would be there", "{f} mentioned it"],
                   "notice": ["you read the notice", "posted on the community board"],
                   "self": ["you arranged it", "your own appointment"],
                   "plan": ["part of your plan"]}


def personas():
    out = []
    for n in sorted(os.listdir(BASE)):
        f = f"{BASE}/{n}/bootstrap_memory/scratch.json"
        if os.path.exists(f):
            out.append((n, f))
    return out


def build(name, f, others):
    s = Scratch(f)
    day = rng.choice([13, 14, 15])
    hour = rng.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    now = datetime.datetime(2023, 2, day, hour, rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50]))
    s.curr_time = s.act_start_time = now
    s.needs = {"hunger": rng.randint(35, 95), "energy": rng.randint(30, 95), "hydration": rng.randint(40, 95),
               "social": rng.randint(25, 95), "hygiene": rng.randint(40, 95), "bladder": rng.randint(30, 95)}
    s.act_address = f"the Ville:{rng.choice(PLACES)}:main room:desk"
    s.agenda, s.resource_goals = [], ([rng.choice(PULLS)] if rng.random() < 0.3 else [])
    friend = rng.choice([o for o in others if o != name])
    kind = rng.choices(["social", "civic", "personal", "work"], [0.5, 0.15, 0.25, 0.1])[0]
    title = rng.choice({"social": SOCIAL, "civic": CIVIC, "personal": PERSONAL, "work": WORK}[kind]).format(f=friend)
    place = rng.choice(PLACES)
    source = rng.choices(["chat", "notice", "self", "plan"], [0.55, 0.15, 0.2, 0.1])[0]
    note = rng.choice(NOTES_BY_SOURCE[source]).format(f=friend)
    status = rng.choices(["known", "going"], [0.6, 0.4])[0]
    timing = rng.choices(["imminent", "ongoing", "later", "tomorrow"], [0.45, 0.25, 0.2, 0.1])[0]
    now_min = now.hour * 60 + now.minute
    dur = rng.choice([60, 90, 120, 120, 180])
    if timing == "imminent":
        start = now_min + rng.randint(8, 40)
    elif timing == "ongoing":
        start = now_min - rng.randint(5, 60)
    elif timing == "later":
        start = now_min + rng.randint(90, 240)
    else:
        start = rng.randint(9 * 60, 19 * 60)
    start = max(0, min(start, 1380))
    date = (now.date() + datetime.timedelta(days=1 if timing == "tomorrow" else 0)).isoformat()
    ev = ag.add_event(s, title, place, date, start, min(start + dur, 1439), source=source, status=status,
                      note=note, now=now)
    if ev is None:
        return None
    p = types.SimpleNamespace(name=name, scratch=s, a_mem=types.SimpleNamespace(seq_event=[]))
    item = rng.choice(PLAN_ITEMS)
    nxt = rng.choice(PLAN_ITEMS) if rng.random() < 0.6 else ""
    pulls = list(s.resource_goals)
    notes = {}
    for e in ag.due(s, now, p, None):
        g = f"attending {e['title']}" + (f" at {e['place']}" if e.get("place") else "")
        pulls.append(g)
        notes[g] = ag.describe(e, now, p, None)
    s._agenda_pull_notes = notes
    cap = {}
    orig = rgp.safe_generate_response
    rgp.safe_generate_response = lambda prompt, *a, **k: cap.setdefault("p", prompt) and None
    try:
        rgp.run_gpt_prompt_choose_activity(p, item, nxt, pulls)
    finally:
        rgp.safe_generate_response = orig
    if "p" not in cap:
        return None
    key = hashlib.md5((name + cap["p"]).encode()).hexdigest()[:16]
    return {"key": key, "persona": name, "t": now.timestamp(), "prompt": cap["p"],
            "meta": {"timing": timing, "status": status, "source": source, "kind": kind,
                     "plan": item, "n_pulls": len(pulls), "due": bool(notes)}}


if __name__ == "__main__":
    ps = personas()
    names = [n for n, _ in ps]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen, rows = set(), []
    while len(rows) < N:
        n, f = rng.choice(ps)
        r = build(n, f, names)
        if r and r["key"] not in seen:
            seen.add(r["key"]); rows.append(r)
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    import collections
    print(len(rows), "cards;", "due:", sum(r["meta"]["due"] for r in rows),
          collections.Counter(r["meta"]["timing"] for r in rows))
