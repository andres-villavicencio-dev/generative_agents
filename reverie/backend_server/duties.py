"""duties.py - venue work that has to be DONE, verified by the environment, and paid per completion.

Why: jobs used to be "stand near the venue for N minutes". Nothing in the world was ever left undone, so
owners had no reason to hire and pay had no link to work. Here a venue accumulates *work* (customers dirty
tables, sales queue orders, counters run low, litter piles up), a duty is generated from that work, a worker
claims it, and the environment completes it only if the worker actually stood at the right object doing the
right thing for long enough. Completing it drains the work, applies a real effect (stock moves from the back
room to the counter, a table is clean, an order is served) and yields the pay. No presence pay.

Pure logic: no maze, LLM or clock inside, so it is unit-testable. The caller (economy / reverie) supplies
`now`, the agent's tile (arena, object) and a stock adapter. State is a plain JSON-able dict (`state`),
persisted by the caller.

Flag: GA_DUTIES (default on; GA_DUTIES=0 turns it off) (everything else is a no-op when off).
"""
import os
import datetime

ENABLED = os.environ.get("GA_DUTIES", "1") == "1"

CLAIM_TTL_MIN = 90          # an unfinished claim is released after this long (the worker wandered off)
ON_SITE_RESET_MIN = 6       # leaving the object for longer than this restarts the duty's work timer
MAX_OPEN_PER_TYPE = 4       # cap on simultaneously open duties of one type at one venue

# Work gauges accumulate in the world; a duty type fires when its gauge crosses `threshold`.
# phrase -> the action text the planner writes for the agent (also what a reader sees in the movement file).
# objs  -> substrings that must appear in the agent's tile object OR arena for the work to count.
# stock -> if set, completing moves stock {back room -> counter} up to the counter's max and pays per unit.
DUTY_TYPES = {
    # ---- Hobbs Cafe ----
    "cafe.refill_counter": dict(venue="cafe", phrase="refilling the counter at Hobbs Cafe", objs=("counter",),
                                minutes=4, base=0.40, per_unit=0.04, stock=True),
    "cafe.clean_table": dict(venue="cafe", phrase="clearing and wiping tables at Hobbs Cafe",
                             objs=("table", "customer seating", "seating"), minutes=3, base=0.80,
                             gauge="dirt", threshold=1.0, drain=1.0),
    "cafe.serve_order": dict(venue="cafe", phrase="serving a customer at the Hobbs Cafe counter", objs=("counter",),
                             minutes=2, base=0.60, gauge="orders", threshold=1.0, drain=1.0),
    # ---- The Rose and Crown Pub ----
    "pub.serve_order": dict(venue="pub", phrase="pouring and serving drinks at the pub bar", objs=("bar", "counter", "shelf"),
                            minutes=2, base=0.80, gauge="orders", threshold=1.0, drain=1.0),
    "pub.stock_shelf": dict(venue="pub", phrase="restocking the bar shelf at The Rose and Crown Pub", objs=("shelf", "bar"),
                            minutes=5, base=0.50, per_unit=0.04, stock=True),
    "pub.clean_tables": dict(venue="pub", phrase="wiping pub tables and washing glasses", objs=("table", "sink", "seating"),
                             minutes=4, base=1.00, gauge="dirt", threshold=1.0, drain=1.0),
    # ---- Harvey Oak Supply Store ----
    "grocery.stock_shelf": dict(venue="grocery", phrase="stocking the shelves at Harvey Oak Supply Store", objs=("shelf", "shelves"),
                                minutes=6, base=0.60, per_unit=0.03, stock=True),
    "grocery.ring_up": dict(venue="grocery", phrase="ringing up a customer at Harvey Oak Supply Store", objs=("counter", "register", "shelf"),
                            minutes=2, base=0.60, gauge="orders", threshold=1.0, drain=1.0),
    "grocery.tidy_aisle": dict(venue="grocery", phrase="tidying the aisles at Harvey Oak Supply Store", objs=("shelf", "shelves", "aisle"),
                               minutes=4, base=0.80, gauge="dirt", threshold=1.5, drain=1.0),
    # ---- The Willows Market and Pharmacy ----
    "market.stock_shelf": dict(venue="market", phrase="stocking the shelves at The Willows Market", objs=("shelf", "shelves", "display"),
                               minutes=6, base=0.60, per_unit=0.03, stock=True),
    "market.ring_up": dict(venue="market", phrase="ringing up a customer at The Willows Market", objs=("counter", "register", "shelf"),
                           minutes=2, base=0.60, gauge="orders", threshold=1.0, drain=1.0),
    "market.tidy_aisle": dict(venue="market", phrase="tidying the aisles at The Willows Market", objs=("shelf", "shelves", "display"),
                              minutes=4, base=0.80, gauge="dirt", threshold=1.5, drain=1.0),
}

# Public (fund-paid) duties; gauges accumulate on their own (litter, patients, backlog) and are drained by work.
CIVIC_DUTY_TYPES = {
    "park.pick_litter": dict(venue="park", phrase="picking up litter and tending the beds at Johnson Park",
                             objs=("park", "bench", "tree", "garden"), minutes=5, base=1.20,
                             gauge="dirt", threshold=1.0, drain=1.0),
    "clinic.treat_patient": dict(venue="clinic", phrase="treating a patient at the Willow Clinic",
                                 objs=("clinic", "bed", "desk", "counter"), minutes=6, base=3.00,
                                 gauge="orders", threshold=1.0, drain=1.0),
    "records.file_record": dict(venue="records", phrase="filing records at the Oak Hill College library desk",
                                objs=("library", "desk", "bookshelf", "computer"), minutes=5, base=1.50,
                                gauge="dirt", threshold=1.0, drain=1.0),
    "gazette.write_article": dict(venue="gazette", phrase="writing an article at the Smallville Gazette desk",
                                  objs=("gazette", "desk", "computer"), minutes=6, base=2.00,
                                  gauge="dirt", threshold=1.0, drain=1.0),
    "townhall.paperwork": dict(venue="townhall", phrase="processing paperwork at Town Hall",
                               objs=("town hall", "desk", "counter"), minutes=5, base=1.80,
                               gauge="dirt", threshold=1.0, drain=1.0),
    "lecture.prepare_lesson": dict(venue="lecture", phrase="preparing a lesson in the Oak Hill College classroom",
                                   objs=("classroom", "desk", "board"), minutes=6, base=2.00,
                                   gauge="dirt", threshold=1.0, drain=1.0),
    "tutor.tutoring": dict(venue="tutor", phrase="tutoring a student in the Oak Hill College classroom",
                           objs=("classroom", "desk", "table"), minutes=6, base=1.80,
                           gauge="dirt", threshold=1.0, drain=1.0),
    "library.shelve_books": dict(venue="library", phrase="shelving books in the Oak Hill College library",
                                 objs=("library", "bookshelf", "bookcase", "shelf"), minutes=5, base=1.20,
                                 gauge="dirt", threshold=1.0, drain=1.0),
    "police.patrol": dict(venue="police", phrase="making a patrol round from the Police Station",
                          objs=("police", "desk"), minutes=6, base=1.50, gauge="dirt", threshold=1.0, drain=1.0),
}

ALL_TYPES = {**DUTY_TYPES, **CIVIC_DUTY_TYPES}
CIVIC_DUTY_TYPES_VENUES = sorted({v['venue'] for v in CIVIC_DUTY_TYPES.values()})

# How fast the world generates work (per customer visit / per hour of open time).
VISIT_DIRT = 0.35           # a customer visit adds this much dirt (3 visits dirty a table)
SALE_ORDER = 1.0            # a sale queues one order to be served
HOURLY_DIRT = {"park": 0.5, "records": 0.25, "police": 0.25, "gazette": 0.25, "townhall": 0.25,
               "lecture": 0.25, "tutor": 0.25, "library": 0.25}   # background accumulation per sim hour
COUNTER_LOW_FRACTION = 0.5  # counter stock below this share of its max asks for a refill


def new_state():
    return {"venues": {}, "open": {}, "seq": 0, "done": 0, "paid": 0.0, "last_tick": None}


def _venue(state, vid):
    return state["venues"].setdefault(vid, {"dirt": 0.0, "orders": 0.0, "done": 0, "unserved": 0})


# ---- world events that create work -------------------------------------------------------------------
def note_visit(state, vid):
    """A customer used the venue: tables get dirty."""
    if vid:
        _venue(state, vid)["dirt"] += VISIT_DIRT


def note_sale(state, vid, units=1):
    """A sale was made: someone has to serve it."""
    if vid:
        v = _venue(state, vid)
        v["orders"] += SALE_ORDER


def accumulate(state, now):
    """Hourly background work (litter, records backlog)."""
    last = state.get("last_tick")
    state["last_tick"] = now.isoformat()
    if not last:
        return
    try:
        dt_h = (now - datetime.datetime.fromisoformat(last)).total_seconds() / 3600.0
    except Exception:
        return
    if dt_h <= 0 or dt_h > 2:
        return
    for vid, rate in HOURLY_DIRT.items():
        _venue(state, vid)["dirt"] += rate * dt_h


# ---- generation -----------------------------------------------------------------------------------
def generate(state, stock=None):
    """Create open duties from the current world state. `stock` (optional adapter) provides
    counter_low(vid) -> bool and backroom_has(vid) -> bool for the restock types."""
    created = []
    for tid, spec in ALL_TYPES.items():
        vid = spec["venue"]
        open_n = sum(1 for d in state["open"].values() if d["type"] == tid)
        if open_n >= MAX_OPEN_PER_TYPE:
            continue
        if spec.get("stock"):
            if stock is None or not stock.counter_low(vid) or not stock.backroom_has(vid):
                continue
            if open_n:
                continue
            want = 1
        else:
            g = _venue(state, vid)[spec["gauge"]]
            pending = sum(1 for d in state["open"].values() if d["type"] == tid)
            want = int((g - pending * spec["drain"]) // spec["threshold"]) if spec["threshold"] > 0 else 0
            want = max(0, min(want, MAX_OPEN_PER_TYPE - open_n))
        for _ in range(want):
            state["seq"] += 1
            did = f"d{state['seq']}"
            state["open"][did] = {"id": did, "type": tid, "venue": vid, "claimed_by": None,
                                  "claimed_at": None, "on_site_min": 0.0, "last_seen": None}
            created.append(did)
    return created


def open_for(state, vid=None, unclaimed=True):
    out = [d for d in state["open"].values()
           if (vid is None or d["venue"] == vid) and (not unclaimed or not d["claimed_by"])]
    return sorted(out, key=lambda d: d["id"])


def claimed_by(state, name):
    for d in state["open"].values():
        if d["claimed_by"] == name:
            return d
    return None


def claim(state, name, duty_id, now):
    d = state["open"].get(duty_id)
    if not d or d["claimed_by"] not in (None, name) or claimed_by(state, name) not in (None, d):
        return None
    d["claimed_by"], d["claimed_at"], d["on_site_min"], d["last_seen"] = name, now.isoformat(), 0.0, None
    return d


def release_stale(state, now):
    n = 0
    for d in state["open"].values():
        if d["claimed_by"] and d["claimed_at"]:
            try:
                age = (now - datetime.datetime.fromisoformat(d["claimed_at"])).total_seconds() / 60.0
            except Exception:
                continue
            if age > CLAIM_TTL_MIN:
                d["claimed_by"], d["claimed_at"], d["on_site_min"], d["last_seen"] = None, None, 0.0, None
                n += 1
    return n


def action_text(duty):
    return ALL_TYPES[duty["type"]]["phrase"]


# ---- verification + completion ----------------------------------------------------------------------
def at_object(spec, arena, obj):
    arena, obj = (arena or "").lower(), (obj or "").lower()
    return any(k in obj or k in arena for k in spec["objs"])


def tick_worker(state, name, arena, obj, now, stock=None):
    """Called every step for an agent holding a duty. Counts on-site time at the right object; when enough
    has accumulated the duty completes. Returns a completion dict or None."""
    d = claimed_by(state, name)
    if d is None:
        return None
    spec = ALL_TYPES[d["type"]]
    here = at_object(spec, arena, obj)
    last = datetime.datetime.fromisoformat(d["last_seen"]) if d.get("last_seen") else None
    d["last_seen"] = now.isoformat()
    if last is None:
        return None
    dt = (now - last).total_seconds() / 60.0
    if dt <= 0:
        return None
    if not here:
        d["away_min"] = d.get("away_min", 0.0) + dt
        if d["away_min"] > ON_SITE_RESET_MIN:
            d["on_site_min"] = 0.0
        return None
    d["away_min"] = 0.0
    if dt > ON_SITE_RESET_MIN:
        d["on_site_min"] = 0.0
        return None
    d["on_site_min"] += dt
    if d["on_site_min"] < spec["minutes"]:
        return None
    return _complete(state, d, spec, stock, now)


def _complete(state, d, spec, stock, now):
    vid = spec["venue"]
    units = 0.0
    if spec.get("stock"):
        units = stock.refill(vid) if stock is not None else 0.0
        if units <= 0:
            # nothing left to move: the duty evaporates unpaid (the work was already done / nothing to do)
            state["open"].pop(d["id"], None)
            return {"duty": d["id"], "type": d["type"], "venue": vid, "worker": d["claimed_by"],
                    "pay": 0.0, "units": 0.0, "ok": False, "reason": "nothing to restock"}
    else:
        v = _venue(state, vid)
        g = spec["gauge"]
        if v[g] < spec["drain"] * 0.999:
            state["open"].pop(d["id"], None)
            return {"duty": d["id"], "type": d["type"], "venue": vid, "worker": d["claimed_by"],
                    "pay": 0.0, "units": 0.0, "ok": False, "reason": "already done"}
        v[g] = max(0.0, v[g] - spec["drain"])
    pay = round(spec["base"] + spec.get("per_unit", 0.0) * units, 2)
    state["open"].pop(d["id"], None)
    _venue(state, vid)["done"] += 1
    state["done"] += 1
    state["paid"] = round(state["paid"] + pay, 2)
    return {"duty": d["id"], "type": d["type"], "venue": vid, "worker": d["claimed_by"],
            "pay": pay, "units": units, "ok": True, "reason": "", "t": now.isoformat()}


def demand_value(state, vid, window_value=None):
    """Outstanding paid work at a venue (for hiring): value of open unclaimed duties + queued gauges."""
    total = 0.0
    for d in state["open"].values():
        if d["venue"] == vid:
            spec = ALL_TYPES[d["type"]]
            total += spec["base"]
    v = state["venues"].get(vid, {})
    total += v.get("unserved", 0) * 0.5
    return round(total, 2)
