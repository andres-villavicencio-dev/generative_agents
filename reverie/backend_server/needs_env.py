"""Needs engine v2 (GA_NEEDS_V2=1): the ENVIRONMENT decides whether a need is relieved.

v1 (reverie.satisfy_needs_for_action) granted a lump of relief whenever an action's text contained a
keyword, so six of nine needs sat at 100. v2 splits the job:

  * INTENT  - what basic need is the agent acting on? A typed Laya decision (GA_NEEDS_LAYA=1, falls
              back to word-boundary keywords when Laya is off or unreachable).
  * OUTCOME - did the world let it happen? Decided here from what actually occurred: a food/drink item
              really consumed or bought (ledger written by reverie.consume_resources_for_action), the
              agent standing in a bathroom / on a seat / at a work object, a real conversation (tick).

Relief is per outcome with a per-need cooldown (no more one lump per 5-minute sub-task), and decay is
rebalanced so a neglected need reaches the critical threshold within hours (see docs/NEEDS_ENGINE_REPORT.md).
"""
import datetime
import json
import os
import re
import time

ENABLED = os.environ.get("GA_NEEDS_V2", "0") == "1"
USE_LAYA = os.environ.get("GA_NEEDS_LAYA", "0") == "1"
LOG_ENABLED = os.environ.get("GA_NEEDS_LOG", "1") == "1"

# Laya answer options (stable slugs); the instruction text carries the meaning.
INTENT_OPTIONS = [
    "eat: eating a meal or snack",
    "drink: drinking water, coffee, tea or another drink",
    "wash: showering, bathing, washing up or brushing teeth",
    "toilet: using the toilet",
    "rest: sitting, lounging, napping or relaxing",
    "sleep: sleeping in bed",
    "socialize: talking or spending time with someone",
    "engage: reading, studying, working on or creating something",
    "none: none of these (walking, shopping, cleaning, errands, ...)",
]
INTENT_SLUGS = [o.split(":")[0] for o in INTENT_OPTIONS]
INTENT_CRITERIA = {o.split(": ", 1)[0]: o.split(": ", 1)[1] for o in INTENT_OPTIONS}   # slug -> description (trainer/server format)

# --- outcome tables -------------------------------------------------------------------------------
FOOD_ITEMS = {"eggs", "bread", "sandwiches", "pastries"}
DRINK_ITEMS = {"milk", "coffee_beans", "beer", "wine", "juice", "water"}
# Tap water: a fixture that has a tap (home kitchen/bathroom, venue counters) ...
WATER_OBJECTS = ("sink", "water", "fountain", "tap", "cooking area", "kitchen counter", "refrigerator",
                 "dispenser", "cooler", "counter", "coffee machine", "drinking")
# ... or a public place that has drinking water (fountain in the park, water cooler in common rooms, a glass
# of tap water at any venue). Bedrooms, bathrooms (sink only) and the street do not count.
WATER_ARENAS = ("kitchen", "dining", "cafe", "pub", "park", "common room", "classroom", "cafeteria",
                "lobby", "store", "garden")
SEAT_OBJECTS = ("sofa", "couch", "armchair", "chair", "bench", "bed", "seat", "stool", "lounge")
WORK_OBJECTS = ("computer", "desk", "bookshelf", "bookcase", "easel", "piano", "microphone", "guitar",
                "library", "table", "workbench", "canvas", "typewriter", "board", "blackboard")
WASH_OBJECTS = ("shower", "bathtub", "bath", "sink")

# (relief per successful outcome, cooldown in sim-minutes)
RELIEF = {
    "hunger":      (50.0, 0),    # real meal; reverie already enforces MEAL_COOLDOWN_MIN on consumption
    "hydration":   (30.0, 20),
    "hygiene":     (70.0, 120),
    "bladder":     (80.0, 45),
    "comfort":     (25.0, 30),
    "energy":      (10.0, 30),
    "stimulation": (20.0, 60),
}
SOCIAL_GAIN_PER_TICK = float(os.environ.get("GA_NEEDS_SOCIAL_GAIN", str(1 / 12)))

# Decay per 10-sim-second tick (awake). Chosen so a neglected need reaches 20 within hours.
# points/hour -> /360: hunger 5, hydration 8, bladder 12, hygiene 2.8, energy 4, social 3,
# comfort 2, stimulation 3, status unchanged.
DECAY_V2 = {
    "hunger": 1 / 72, "hydration": 1 / 45, "bladder": 1 / 30, "hygiene": 1 / 130,
    "energy": 1 / 90, "social": 1 / 120, "comfort": 1 / 180, "stimulation": 1 / 120,
    "status": 1 / 540,
}
SLEEP_ENERGY_GAIN = 1 / 45        # per tick while asleep (~8 pts/hour)
SLEEP_DECAY_FACTOR = 0.3          # hunger / hydration / bladder while asleep

# --- keyword fallback (word-boundary, one intent) -------------------------------------------------
_KW = [
    ("toilet", r"\b(toilet|restroom|bathroom break|use the bathroom|using the bathroom)\b"),
    ("wash", r"\b(shower(ing)?|bath(e|ing)?|brush(ing)? (my |his |her |their )?teeth|wash(ing)? (up|face|hands))\b"),
    ("sleep", r"\b(sleep(ing)?|asleep|go(ing)? to bed|in bed)\b"),
    ("eat", r"\b(eat(ing)?|breakfast|lunch|dinner|meal|snack|brunch|supper)\b"),
    ("drink", r"\b(drink(ing)?|coffee|tea|water|juice|beer|wine)\b"),
    ("socialize", r"\b(chat(ting)?|talk(ing)?|conversation|converse|party|socializ\w+)\b"),
    ("rest", r"\b(rest(ing)?|nap(ping)?|relax(ing)?|lounge|lounging|sit(ting)? (down|on|in))\b"),
    ("engage", r"\b(read(ing)?|study(ing)?|work(ing)? on|research(ing)?|writ(e|ing)|paint(ing)?|compos(e|ing)|practic(e|ing))\b"),
]
_KW = [(k, re.compile(p, re.I)) for k, p in _KW]


def keyword_intent(desc):
    d = (desc or "").split(" @ ")[0]
    for slug, rx in _KW:
        if rx.search(d):
            return slug
    return "none"


_intent_cache = {}


def laya_intent(persona, desc, address):
    """Typed Laya decision on its own server (GA_NEEDS_LAYA_URL, default = the sim's Laya) so the need model
    can be A/B-tested without changing the model that decides everything else. Returns a slug, or None
    to signal fallback. Never raises."""
    try:
        import urllib.request
        from persona.prompt_template import laya_scoring as L
        key = ((desc or "")[:200], (address or "")[:120])
        if key in _intent_cache:
            return _intent_cache[key]
        url = os.environ.get("GA_NEEDS_LAYA_URL", L.LAYA_URL).rstrip("/")
        body = {"model": "laya-v3",
                "state": {"who": persona.name,
                          "situation": f"{persona.name} is doing: {desc[:300]}\nLocation: {address[:120]}"},
                "questions": {"need": {"type": "choice",
                                       "instructions": "Which basic need is this person acting on right now? "
                                                       "Pick the single best match.",
                                       "criteria": INTENT_CRITERIA}}}
        req = urllib.request.Request(f"{url}/v1/systemone", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=L.LAYA_TIMEOUT) as resp:
            ans = (json.loads(resp.read()).get("answers") or {}).get("need")
        if not isinstance(ans, dict):
            return None
        probs = {k: v for k, v in (ans.get("probabilities") or {}).items() if not str(k).startswith("__")}
        legend = ans.get("legend") or {}
        mass = {}
        for k, v in probs.items():
            lab = str(legend.get(k, k))
            for slug_, desc_ in INTENT_CRITERIA.items():
                if lab in (slug_, desc_):
                    mass[slug_] = mass.get(slug_, 0.0) + float(v)
        if not mass:
            return None
        slug = max(mass, key=mass.get)
        if slug in INTENT_SLUGS:
            if len(_intent_cache) > 20000:
                _intent_cache.clear()
            _intent_cache[key] = slug
            return slug
    except Exception:
        pass
    return None


def get_intent(persona, desc, address):
    if USE_LAYA:
        s = laya_intent(persona, desc, address)
        if s:
            return s, "laya"
    return keyword_intent(desc), "keyword"


# --- ledger: what the world actually gave the agent ------------------------------------------------
def note_consumed(persona, item):
    """Called by reverie.consume_resources_for_action on every successful consume/purchase."""
    try:
        led = getattr(persona.scratch, "_relief_ledger", None)
        if led is None:
            led = persona.scratch._relief_ledger = []
        led.append(item)
    except Exception:
        pass


def _tile(maze, persona):
    try:
        t = maze.access_tile(persona.scratch.curr_tile)
        return ((t.get("sector") or "").lower(), (t.get("arena") or "").lower(),
                (t.get("game_object") or "").lower())
    except Exception:
        return ("", "", "")


def _ready(persona, need, now, cooldown_min):
    if cooldown_min <= 0 or now is None:
        return True
    last = getattr(persona.scratch, "_relief_last", {}).get(need)
    return last is None or (now - last).total_seconds() >= cooldown_min * 60


def _stamp(persona, need, now):
    d = getattr(persona.scratch, "_relief_last", None)
    if d is None:
        d = persona.scratch._relief_last = {}
    d[need] = now


def _give(persona, need, amount, now, out):
    amt, cd = RELIEF[need]
    if not _ready(persona, need, now, cd):
        return
    persona.scratch.needs[need] = min(100, persona.scratch.needs.get(need, 0) + (amount if amount else amt))
    _stamp(persona, need, now)
    out[need] = out.get(need, 0) + (amount if amount else amt)


LOCATION_BOUND = ("wash", "toilet", "rest", "engage", "drink")


def _water_here(arena, obj):
    return any(w in obj for w in WATER_OBJECTS) or any(a in arena for a in WATER_ARENAS)


def _located(persona, intent, ledger, arena, sector, obj, now, out):
    if intent == "wash":
        if "level" in ledger or (("bath" in arena or "bath" in sector or "shower" in obj or "bath" in obj)
                                 and any(w in obj for w in WASH_OBJECTS)):
            _give(persona, "hygiene", None, now, out)
    elif intent == "toilet":
        if "bathroom" in arena or "restroom" in arena or "bathroom" in obj or "toilet" in obj:
            _give(persona, "bladder", None, now, out)
    elif intent == "drink":
        if _water_here(arena, obj):
            _give(persona, "hydration", 25.0, now, out)
    elif intent == "rest":
        if any(w in obj for w in SEAT_OBJECTS):
            _give(persona, "comfort", None, now, out)
            _give(persona, "energy", None, now, out)
    elif intent == "engage":
        if any(w in obj for w in WORK_OBJECTS):
            _give(persona, "stimulation", None, now, out)


WRAP_UP_MIN = 2   # minutes an agent lingers after a toilet/shower need is met


def _wrap_up(persona, intent, now):
    """Once a toilet/wash need is met, end that action shortly so the planner resumes the agent's schedule
    instead of leaving them at the fixture for the rest of the slot."""
    if intent not in ("toilet", "wash") or now is None:
        return
    s = persona.scratch
    try:
        if getattr(s, "chatting_with", None) or not getattr(s, "act_start_time", None):
            return
        x = s.act_start_time
        if x.second != 0:
            x = x.replace(second=0) + datetime.timedelta(minutes=1)
        elapsed = max(0, int((now - x).total_seconds() // 60))
        new = elapsed + WRAP_UP_MIN
        if new < (s.act_duration or 0):
            s.act_duration = new
    except Exception:
        pass


def _act_key(s):
    return (getattr(s, "act_description", None), str(getattr(s, "act_start_time", None)))


def recheck(persona, maze, now):
    """Called every step. A place-bound intent (toilet, shower, seat, desk) is satisfied the moment the agent's
    tile qualifies, once per action instance. Stateless w.r.t. checkpoints: derived from the current action,
    so an action already running when a sim resumes is still evaluated."""
    s = persona.scratch
    desc = getattr(s, "act_description", None)
    if not desc or not getattr(s, "needs", None):
        return {}
    key = _act_key(s)
    if getattr(s, "_relief_key", None) == key:
        return {}
    # Classify on the description alone: the address changes along the way (the destination, then the tile the
    # agent actually stands on) and a "kitchen" address once classified a bathroom visit as eating, which latched
    # the action as done and left agents on the toilet with no relief (106 accidents in n70).
    intent, src = get_intent(persona, desc, "")
    if intent not in LOCATION_BOUND:
        s._relief_key = key
        return {}
    sector, arena, obj = _tile(maze, persona)
    out = {}
    _located(persona, intent, [], arena, sector, obj, now, out)
    if out:
        s._relief_key = key
        _wrap_up(persona, intent, now)
        if LOG_ENABLED:
            _trace({"kind": "relief", "who": persona.name, "t": str(now), "intent": intent, "src": "recheck",
                    "desc": desc[:140], "obj": obj, "arena": arena, "ledger": [], "relief": out})
    return out


def apply_relief(persona, maze, desc, now):
    """Decide, from what actually happened, which needs this action relieved. Returns a dict for tracing.
    Always clears the consumption ledger."""
    s = persona.scratch
    ledger = list(getattr(s, "_relief_ledger", None) or [])
    s._relief_ledger = []
    if not getattr(s, "needs", None) or not desc:
        return {}
    address = getattr(s, "act_address", "") or ""
    intent, src = get_intent(persona, desc, address)
    sector, arena, obj = _tile(maze, persona)
    out = {}
    if intent == "eat":
        if any(i in FOOD_ITEMS for i in ledger):
            _give(persona, "hunger", None, now, out)
            s.needs["hydration"] = min(100, s.needs.get("hydration", 0) + 10)
    elif intent == "drink":
        if any(i in DRINK_ITEMS for i in ledger):
            _give(persona, "hydration", None, now, out)
            if "coffee_beans" in ledger:
                s.needs["stimulation"] = min(100, s.needs.get("stimulation", 0) + 10)
        elif _water_here(arena, obj):
            _give(persona, "hydration", 25.0, now, out)
        if out:
            s._relief_key = _act_key(s)
    elif intent in LOCATION_BOUND:
        _located(persona, intent, ledger, arena, sector, obj, now, out)
        if out:
            s._relief_key = _act_key(s)
            _wrap_up(persona, intent, now)   # else recheck() keeps trying each step while the action lasts
    # socialize: nothing here - social is gained per tick in a real conversation (tick_gain)
    # sleep: restored per tick while asleep (tick_gain); nothing here
    if LOG_ENABLED:
        _trace({"kind": "relief", "who": persona.name, "t": str(now), "intent": intent, "src": src,
                "desc": desc[:140], "obj": obj, "arena": arena, "ledger": ledger, "relief": out})
    return out


# --- tick -------------------------------------------------------------------------------------------
def tick_value(need, cur, is_sleeping, is_chatting, multiplier):
    """One 10-second tick under v2. Returns the new (unclamped) value."""
    rate = DECAY_V2.get(need, 0.0) * multiplier
    if need == "energy":
        return cur + SLEEP_ENERGY_GAIN if is_sleeping else cur - rate
    if need == "social":
        if is_chatting:
            return cur + SOCIAL_GAIN_PER_TICK
        return cur if is_sleeping else cur - rate
    if need in ("hunger", "hydration", "bladder"):
        return cur - rate * (SLEEP_DECAY_FACTOR if is_sleeping else 1.0)
    return cur if is_sleeping else cur - rate


# --- tracing ----------------------------------------------------------------------------------------
_trace_fh = None


def _trace_path(name):
    base = os.environ.get("GA_TRACE_DIR") or os.path.join(os.path.dirname(__file__), "../../ga_traces")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


def _trace(rec, name="needs_relief.jsonl"):
    try:
        with open(_trace_path(name), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def log_snapshot(personas, curr_time):
    """One line per agent: all nine meters. Called once per sim hour by the step loop."""
    if not LOG_ENABLED:
        return
    try:
        with open(_trace_path("needs_timeseries.jsonl"), "a") as f:
            for name, p in personas.items():
                n = getattr(p.scratch, "needs", None)
                if n:
                    f.write(json.dumps({"t": str(curr_time), "who": name,
                                        **{k: round(v, 1) for k, v in n.items()}}) + "\n")
    except Exception:
        pass


# ======================================================================================================
# Need-zero crises (GA_NEEDS_CRISIS, default on under v2): a need that reaches 0 is an unpleasant,
# highly poignant event with a consequence, and the need is partly restored so it cannot repeat at once.
# Medical collapses (energy / hunger / hydration) carry the agent to the Willow Clinic for a stay and
# bill a small fee into the community fund. The others are personal: an accident, being shunned, an
# emotional breakdown, aches, numbness. status is excluded by design (it lingers, never an emergency).
# ======================================================================================================
CRISIS_ENABLED = os.environ.get("GA_NEEDS_CRISIS", "1") == "1"
CLINIC_BED = "the Ville:Willow Clinic:exam room 1:bed"
CRISIS = {
    "energy":      dict(kind="hospital", stay=150, fee=15.0, restore=70, poign=9, also={"hunger": 40, "hydration": 50},
                        what="collapsed from exhaustion", memo="{n} had gone far too long without rest, blacked out and woke up at the "
                        "Willow Clinic hours later, frightened and humiliated, with a bill to pay."),
    "hunger":      dict(kind="hospital", stay=90, fee=10.0, restore=60, poign=9, also={"hydration": 50},
                        what="fainted from hunger", memo="{n} went so long without food that they fainted, were carried to the Willow "
                        "Clinic and put on a drip. It was terrifying, and they were billed for it."),
    "hydration":   dict(kind="hospital", stay=90, fee=8.0, restore=70, poign=9, also={},
                        what="collapsed from dehydration", memo="{n} became dangerously dehydrated, collapsed and woke up in the Willow "
                        "Clinic with a fluid drip. Never again, {n} thought, and paid the fee."),
    "bladder":     dict(kind="accident", restore=65, poign=8, hit={"hygiene": 40, "social": 15},
                        what="had an embarrassing accident", memo="{n} could not find a bathroom in time and had an accident in public. "
                        "The shame was overwhelming."),
    "hygiene":     dict(kind="shunned", restore=50, poign=7, hit={"social": 25},
                        what="was shunned for poor hygiene", memo="{n} had not washed in so long that people kept their distance and "
                        "someone said so out loud. It stung."),
    "social":      dict(kind="breakdown", restore=35, poign=8, stress=0.25, hit={},
                        what="had an emotional breakdown from loneliness", memo="{n} had been alone for so long that they broke down in "
                        "tears. Nobody was there to notice."),
    "comfort":     dict(kind="ache", restore=50, poign=7, hit={"energy": 10},
                        what="was laid low by aches and exhaustion", memo="{n}'s body gave out after too long without any real comfort: "
                        "aching and miserable, they could barely move."),
    "stimulation": dict(kind="numb", restore=45, poign=7, stress=0.1, hit={},
                        what="sank into a numb, empty mood", memo="{n} had gone so long without anything to engage them that a grey, "
                        "empty numbness settled in."),
}
CRISIS_STATS = {}


def _clinic_tile(maze):
    try:
        tiles = maze.address_tiles.get(CLINIC_BED)
        if not tiles:
            for a, t in maze.address_tiles.items():
                if a.startswith("the Ville:Willow Clinic") and a.endswith(":bed") and t:
                    return a, sorted(t)[0]
            return None, None
        return CLINIC_BED, sorted(tiles)[0]
    except Exception:
        return None, None


def _remember(persona, now, text, poignancy, kws):
    try:
        from persona.prompt_template.gpt_structure import get_embedding
        try:
            emb = get_embedding(text)
        except Exception:
            emb = [0.0] * 768
        persona.a_mem.add_event(now, None, persona.name, "suffered", "a crisis", text,
                                set(kws) | {persona.name}, poignancy, (text, emb), [])
    except Exception as e:
        print(f"[NeedsCrisis] memory failed for {persona.name}: {e}")


def _charge(persona, economy, fee, now):
    paid = min(float(fee), max(0.0, float(getattr(persona.scratch, "wallet", 0) or 0)))
    if paid <= 0:
        return 0.0
    persona.scratch.wallet = round(persona.scratch.wallet - paid, 2)
    try:
        if economy is not None:
            f = economy._fund()
            f["balance"] = round(f["balance"] + paid, 2)
            f["in"] = round(f["in"] + paid, 2)
            economy._log_pay("fee", persona.name, paid, None, note="clinic fee")
    except Exception as e:
        print(f"[NeedsCrisis] fee credit failed: {e}")
    return paid


def check_crises(persona, maze, economy, now):
    """Call once per step per persona (after the needs tick). Returns a list of crisis dicts that fired."""
    if not (ENABLED and CRISIS_ENABLED):
        return []
    s = persona.scratch
    needs = getattr(s, "needs", None)
    if not needs or s.chatting_with:
        return []
    until = getattr(s, "_crisis_until", None)
    if until is not None and now is not None and now < until:
        return []                                   # already being treated
    fired = []
    for need, spec in CRISIS.items():
        if needs.get(need, 100) > 0.05:
            continue
        if need == "energy" and "sleep" in (s.act_description or "").lower():
            continue
        first = persona.name.split()[0]
        memo = spec["memo"].format(n=first)
        paid = 0.0
        place = None
        if spec["kind"] == "hospital":
            addr, tile = _clinic_tile(maze)
            if addr:
                place = addr
                paid = _charge(persona, economy, spec["fee"], now)
                s.add_new_action(addr, spec["stay"], f"{spec['what']} and being treated at the Willow Clinic",
                                 "🏥", (persona.name, "is", "being treated at the Willow Clinic"),
                                 None, None, None, None,
                                 "hospital bed", "🛏️", ("hospital bed", "is", "occupied"))
                s.planned_path = [tuple(tile)]       # execute() takes this as the next tile: a one-step transport
                s.act_path_set = True
                s._crisis_until = now + datetime.timedelta(minutes=spec["stay"]) if now else None
                memo += (f" The clinic charged ${paid:.0f}." if paid else " The clinic waived the fee.")
        needs[need] = float(spec["restore"])
        for k, v in (spec.get("also") or {}).items():
            needs[k] = max(needs.get(k, 0), float(v))
        for k, v in (spec.get("hit") or {}).items():
            needs[k] = max(0.0, needs.get(k, 0) - float(v))
        if spec.get("stress"):
            s.financial_stress = min(1.0, float(getattr(s, "financial_stress", 0) or 0) + spec["stress"])
        if spec["kind"] != "hospital":
            s.act_pronunciatio = {"accident": "😳", "shunned": "😔", "breakdown": "😭", "ache": "😣", "numb": "😶"}.get(spec["kind"], "😟")
        _remember(persona, now, memo, spec["poign"], {"crisis", need, spec["kind"]})
        CRISIS_STATS[need] = CRISIS_STATS.get(need, 0) + 1
        rec = {"kind": "crisis", "who": persona.name, "t": str(now), "need": need, "type": spec["kind"],
               "fee": paid, "place": place, "restored_to": spec["restore"]}
        _trace(rec, "needs_crisis.jsonl")
        print(f"[NeedsCrisis] {persona.name}: {need}=0 -> {spec['what']}" + (f" (clinic, ${paid:.0f})" if place else ""))
        fired.append({**rec, "what": spec["what"], "memo": memo, "poignancy": spec["poign"]})
        break                                        # one crisis per step per agent
    return fired
