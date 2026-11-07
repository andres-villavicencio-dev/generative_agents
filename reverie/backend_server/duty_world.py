"""duty_world.py - the impure glue between duties.py (pure work engine) and the sim (maze, personas, economy).

Per step (GA_DUTIES=1):
  accumulate background work -> generate duties from world state -> release stale claims
  -> hand an open duty to each eligible idle worker as a REAL timed action at the right object
  -> verify (duties.tick_worker) that workers actually stand at that object -> pay per completion.
"""
import datetime
import duties as D

SHIFT_OPEN, SHIFT_CLOSE = 8, 20            # sim hours an employee takes duties
OWNER_DAILY_CAP = 6                         # duties an owner does per day when the venue has no employee
CIVIC_TASK_DUTY_VENUE = {"park": "park", "clinic": "clinic", "records": "records", "police": "police",
                         "gazette": "gazette", "townhall": "townhall", "lecture": "lecture",
                         "tutor": "tutor", "library": "library"}
_emoji = {"restock": "📦", "clean": "🧹", "serve": "🛎️"}


def _addr_keys(venue_id):
    import economy as E
    if venue_id in E.VENUES:
        return (E.VENUES[venue_id]["address_key"],)
    if venue_id in E.CIVIC_TASKS:
        return tuple(E.CIVIC_TASKS[venue_id]["keys"])
    return ()


def duty_address(maze, venue_id, spec):
    keys = _addr_keys(venue_id)
    if not keys:
        return None
    cands = []
    for addr, tiles in getattr(maze, "address_tiles", {}).items():
        a = addr.lower()
        if tiles and all(k in a for k in keys):
            cands.append(addr)
    if not cands:
        return None
    for addr in sorted(cands):
        last = addr.lower().split(":")[-1]
        if any(k in last for k in spec["objs"]):
            return addr
    return sorted(cands)[0]


def _tile_info(maze, persona):
    try:
        t = maze.access_tile(persona.scratch.curr_tile)
        return ((t.get("arena") or "") + " " + (t.get("sector") or "")).strip(), (t.get("game_object") or "")
    except Exception:
        return "", ""


def _busy(s):
    return bool(getattr(s, "chatting_with", None)) or getattr(s, "_crisis_until", None) is not None and \
        getattr(s, "curr_time", None) is not None and s.curr_time < s._crisis_until


def _asleep(s):
    d = (s.act_description or "").lower()
    return "sleep" in d or "asleep" in d


def workers(economy, now):
    """[(name, venue_id, role)] who may take a duty right now."""
    import economy as E
    out = []
    st = economy.state
    h = now.hour
    if SHIFT_OPEN <= h < SHIFT_CLOSE:
        for vid, names in st["employees"].items():
            for n in names:
                out.append((n, vid, "employee"))
        for vid, cfg in E.VENUES.items():
            if not st["employees"].get(vid):
                out.append((cfg["owner"], vid, "owner"))
    if E.CIVIC_START <= h < E.CIVIC_END:
        for n, tid in st["civic"].items():
            v = CIVIC_TASK_DUTY_VENUE.get(tid)
            if v:
                out.append((n, v, "civic"))
    return out


STATUS_CIVIC = 4.0    # status points per verified civic duty
STATUS_SHOP = 1.0     # per verified shop duty


def _pay(economy, name, vid, role, completion, now):
    import economy as E
    pay = completion["pay"]
    if pay <= 0:
        return
    s = economy._personas[name].scratch
    if role == "civic":
        f = economy._fund()
        pay = round(min(pay, max(0.0, f["balance"])), 2)
        f["balance"] = round(f["balance"] - pay, 2)
        f["out"] = round(f["out"] + pay, 2)
        f["today_out_acc"] = round(f.get("today_out_acc", 0.0) + pay, 2)
    else:
        till = economy.state["tills"].get(vid, 0.0)
        from_till = min(pay, max(0.0, till))
        economy.state["tills"][vid] = round(till - from_till, 2)
        rest = round(pay - from_till, 2)
        if rest > 0 and name != E.VENUES[vid]["owner"]:
            o = economy._personas[E.VENUES[vid]["owner"]].scratch
            o.wallet = round(getattr(o, "wallet", 0.0) - rest, 2)
    s.wallet = round(getattr(s, "wallet", 0.0) + pay, 2)
    economy._relax_stress(s, 0.05)
    # Doing useful work earns standing: civic service most (the town sees it), shop work a little.
    try:
        gain = STATUS_CIVIC if role == "civic" else STATUS_SHOP
        s.needs["status"] = min(100, s.needs.get("status", 50) + gain)
    except Exception:
        pass
    economy._log_pay("duty", name, pay, vid if vid in E.VENUES else None,
                     note=f"{completion['type']} x{completion.get('units', 0) or 1}")


def step(economy, personas, maze, now, resource_manager=None, trace=None):
    """One sim step of duty management. Returns the list of completions."""
    if not D.ENABLED or economy is None:
        return []
    st = economy.state.get("duties")
    if not st:
        st = economy.state["duties"] = D.new_state()
    D.accumulate(st, now)
    D.generate(st, resource_manager)
    D.release_stale(st, now)
    for vid in {d["venue"] for d in st["open"].values()}:       # the day's peak unclaimed paid work, for hiring
        v = st["venues"].setdefault(vid, {})
        v["backlog_peak"] = max(v.get("backlog_peak", 0.0), D.demand_value(st, vid))
    done = []
    elig = {}
    for name, vid, role in workers(economy, now):
        if name in personas:
            elig.setdefault(name, (vid, role))
    # 1) verify work in progress
    for name, p in personas.items():
        if D.claimed_by(st, name) is None:
            continue
        arena, obj = _tile_info(maze, p)
        c = D.tick_worker(st, name, arena, obj, now, resource_manager)
        if c:
            vid, role = elig.get(name, (c["venue"], "employee"))
            if c["ok"]:
                _pay(economy, name, c["venue"], role, c, now)
                _wrap(p, now)
            if trace:
                trace({**c, "t": str(now), "role": role})
            done.append(c)
    # 2) hand out new duties to idle eligible workers
    today = now.toordinal()
    for name, (vid, role) in elig.items():
        p = personas[name]
        s = p.scratch
        if D.claimed_by(st, name) is not None or _busy(s) or _asleep(s):
            continue
        # A hired worker (employee / civic post holder) on shift drops routine activity for the job;
        # owners only take duty work in a natural gap in their day.
        if role == "owner" and not s.act_check_finished():
            continue
        if role == "owner":
            cnt = st["venues"].setdefault(vid, {}).setdefault("owner_day", {})
            if cnt.get("d") != today:
                cnt["d"], cnt["n"] = today, 0
            if cnt["n"] >= OWNER_DAILY_CAP:
                continue
        opts = D.open_for(st, vid)
        if not opts:
            continue
        duty = max(opts, key=lambda d: D.ALL_TYPES[d["type"]]["base"])     # Laya `duty` decision plugs in here
        spec = D.ALL_TYPES[duty["type"]]
        addr = duty_address(maze, vid, spec)
        if not addr or not D.claim(st, name, duty["id"], now):
            continue
        if role == "owner":
            st["venues"][vid]["owner_day"]["n"] += 1
        obj = addr.split(":")[-1]
        txt = D.action_text(duty)
        try:
            s.add_new_action(addr, spec["minutes"] + 6, txt, "🛠️", (name, "is", txt),
                             None, None, None, None, obj, "🛠️", (obj, "is", "being used"))
        except Exception as e:
            print(f"[Duties] could not start duty for {name}: {e}")
            D.release_stale(st, now + datetime.timedelta(minutes=D.CLAIM_TTL_MIN + 1))
    return done


def _wrap(persona, now):
    """Work done: end the action so the planner resumes."""
    s = persona.scratch
    try:
        x = s.act_start_time
        if x is not None:
            if x.second != 0:
                x = x.replace(second=0) + datetime.timedelta(minutes=1)
            elapsed = max(0, int((now - x).total_seconds() // 60))
            if elapsed + 1 < (s.act_duration or 0):
                s.act_duration = elapsed + 1
    except Exception:
        pass
