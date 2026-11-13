"""Per-persona agenda: dated events an agent knows about, plans to attend,
declined or missed (an invitation heard in a chat, a notice read on the
board, an appointment it made itself).

Why this exists (Valentine's party analysis, Nov 2026): the schedule was
executed mechanically and nothing stored a *future* event with a time and a
place, so agents heard about the party but never went. The agenda is the
structured memory behind the "schedule as a tool" the agent checks and edits:

    check   -> render()           what is coming up, minutes until, travel time
    add     -> add_event()        learn of / accept an event (any future date)
    modify  -> modify_event()
    cancel  -> cancel_event()     decline, with a reason
    follow  -> overlay()          'going' events are pinned into f_daily_schedule

Events are plain dicts so they save into scratch.json unchanged. Nothing here
knows about any particular event (emergence, not hard-coding).
"""
import datetime
import os
import re

STATUSES = ("known", "going", "declined", "done", "missed")
LIVE = ("known", "going")


def enabled():
  """GA_AGENDA, default ON (project rule: feature flags are on unless set off)."""
  return os.environ.get("GA_AGENDA", "1") != "0"


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
def _store(scratch):
  a = getattr(scratch, "agenda", None)
  if not isinstance(a, list):
    a = []
    scratch.agenda = a
  return a


def _words(text):
  text = re.sub(r"[\u2019']s\b", "", str(text).lower())      # possessives
  return {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2}


def _similar(a, b):
  wa, wb = _words(a), _words(b)
  if not wa or not wb:
    return False
  return len(wa & wb) / len(wa | wb) >= 0.34


def _norm_date(d, now):
  """ISO string for a date given as ISO, 'today', 'tomorrow', 'Feb 14',
  'February 14, 2023', a weekday name, or a date object; None if unparseable."""
  if isinstance(d, datetime.datetime):
    return d.date().isoformat()
  if isinstance(d, datetime.date):
    return d.isoformat()
  t = str(d or "").strip().lower()
  if not t or t == "today":
    return now.date().isoformat()
  if t == "tomorrow":
    return (now.date() + datetime.timedelta(days=1)).isoformat()
  m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", t)
  if m:
    try:
      return datetime.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    except ValueError:
      return None
  months = {n: i for i, n in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
  m = re.match(r"^([a-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?$", t)
  if m and m[1] in months:
    try:
      return datetime.date(int(m[3] or now.year), months[m[1]], int(m[2])).isoformat()
    except ValueError:
      return None
  days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
  if t[:3] in days:
    ahead = (days.index(t[:3]) - now.weekday()) % 7
    return (now.date() + datetime.timedelta(days=ahead)).isoformat()
  return None


def parse_clock(v):
  """Minutes after midnight for 17, '17:00', '5pm', '5:30 PM', 1020 (>=24 means
  already minutes), or None."""
  if isinstance(v, bool):
    return None
  if isinstance(v, (int, float)):
    v = int(v)
    return v * 60 if 0 <= v < 24 else (v if 0 <= v < 1440 else None)
  m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*([ap])?\.?m?\.?\s*$", str(v or ""), re.I)
  if not m:
    return None
  h, mi, ap = int(m[1]), int(m[2] or 0), (m[3] or "").lower()
  if ap == "p" and h < 12:
    h += 12
  if ap == "a" and h == 12:
    h = 0
  if not (0 <= h <= 23 and 0 <= mi <= 59):
    return None
  return h * 60 + mi


def add_event(scratch, title, place="", date=None, start=None, end=None,
              source="chat", status="known", note="", now=None):
  """Add an event, or update the matching one (same date, close start, similar
  title). Returns the event dict, or None if the date/time cannot be read."""
  now = now or scratch.curr_time
  d = _norm_date(date, now)
  s = parse_clock(start)
  if d is None or s is None or not str(title).strip():
    return None
  e = parse_clock(end)
  if e is None or e <= s:
    e = min(s + 60, 1439)
  if status not in STATUSES:
    status = "known"
  store = _store(scratch)
  for ev in store:
    same_slot = bool(place and ev.get("place") and ev["start"] == s
                     and _similar(ev["place"], place))
    if ev["date"] == d and abs(ev["start"] - s) <= 30 and (_similar(ev["title"], title) or same_slot):
      if status == "going" or ev["status"] not in LIVE:
        ev["status"] = status
      for k, v in (("place", place), ("note", note)):
        if v and not ev.get(k):
          ev[k] = str(v)[:200]
      ev["end"] = max(ev["end"], e)
      ev.pop("pinned", None)
      return ev
  nid = 1 + max([x.get("id", 0) for x in store] or [0])
  ev = {"id": nid, "title": str(title).strip()[:120], "place": str(place or "")[:120],
        "date": d, "start": s, "end": e, "source": source, "status": status,
        "note": str(note or "")[:200], "added": now.strftime("%Y-%m-%d %H:%M")}
  store.append(ev)
  return ev


def start_in_past(date, start, now):
  """Error text if the given date+start is already behind <now>, else None.
  An unreadable date/time returns None (add_event reports that itself)."""
  d, c = _norm_date(date, now), parse_clock(start)
  if d is None or c is None:
    return None
  when = datetime.datetime.combine(datetime.date.fromisoformat(d), datetime.time(c // 60, c % 60))
  if when < now.replace(second=0, microsecond=0):
    return (f"Error: {d} {c // 60:02d}:{c % 60:02d} is already in the past (it is "
            f"{now.strftime('%Y-%m-%d %H:%M')}). Only record an event whose date and start time "
            "were actually stated and are still ahead; if it was not stated, reply done.")
  return None


def get_event(scratch, ev_id):
  for ev in _store(scratch):
    if str(ev.get("id")) == str(ev_id).lstrip("#"):
      return ev
  return None


def modify_event(scratch, ev_id, **fields):
  ev = get_event(scratch, ev_id)
  if ev is None:
    return None
  now = scratch.curr_time
  for k, v in fields.items():
    if v in (None, ""):
      continue
    if k == "date":
      nd = _norm_date(v, now)
      if nd:
        ev["date"] = nd
    elif k in ("start", "end"):
      c = parse_clock(v)
      if c is not None:
        ev[k] = c
    elif k in ("title", "place", "note"):
      ev[k] = str(v)[:200]
    elif k == "status" and v in STATUSES:
      ev["status"] = v
  if ev["end"] <= ev["start"]:
    ev["end"] = min(ev["start"] + 60, 1439)
  ev.pop("pinned", None)
  return ev


def cancel_event(scratch, ev_id, reason=""):
  ev = get_event(scratch, ev_id)
  if ev is None:
    return None
  ev["status"] = "declined"
  if reason:
    ev["note"] = str(reason)[:200]
  ev.pop("pinned", None)
  return ev


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------
def _abs_start(ev):
  d = datetime.date.fromisoformat(ev["date"])
  return datetime.datetime.combine(d, datetime.time()) + datetime.timedelta(minutes=ev["start"])


def _abs_end(ev):
  d = datetime.date.fromisoformat(ev["date"])
  return datetime.datetime.combine(d, datetime.time()) + datetime.timedelta(minutes=ev["end"])


def upcoming(scratch, now=None, horizon_min=36 * 60, statuses=LIVE):
  """Live events not yet over that start within horizon_min (or are ongoing),
  soonest first, each paired with minutes until start (negative = ongoing)."""
  now = now or scratch.curr_time
  out = []
  for ev in _store(scratch):
    if ev["status"] not in statuses:
      continue
    try:
      st, en = _abs_start(ev), _abs_end(ev)
    except Exception:
      continue
    if en <= now:
      continue
    until = int((st - now).total_seconds() // 60)
    if until <= horizon_min:
      out.append((until, ev))
  out.sort(key=lambda x: x[0])
  return out


def travel_minutes(persona, place, maze=None, default=8, slack=3):
  """Walking minutes to <place> (a loose text hint). One tile per 10 s step.
  Falls back to <default> without a maze or a matching address."""
  try:
    if maze is None or not place:
      return default
    tile = persona.scratch.curr_tile
    want = _words(place)
    best = None
    for addr, tiles in maze.address_tiles.items():
      if want and want <= _words(addr.replace(":", " ")) and tiles:
        d = min(abs(t[0] - tile[0]) + abs(t[1] - tile[1]) for t in tiles)
        best = d if best is None else min(best, d)
    if best is None:
      return default
    return int(-(-best * 10 // 60)) + slack
  except Exception:
    return default


def _when(ev, now):
  st = _abs_start(ev)
  hhmm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
  if st.date() == now.date():
    return f"today {hhmm(ev['start'])}-{hhmm(ev['end'])}"
  if st.date() == now.date() + datetime.timedelta(days=1):
    return f"tomorrow {hhmm(ev['start'])}-{hhmm(ev['end'])}"
  return f"{st.strftime('%a %b %d')} {hhmm(ev['start'])}-{hhmm(ev['end'])}"


def describe(ev, now, persona=None, maze=None):
  """One line: '#2 Valentine's party at Hobbs Cafe — today 17:00-19:00, starts in
  35 min (~6 min away) [going]'."""
  until = int((_abs_start(ev) - now).total_seconds() // 60)
  if until > 0:
    tail = f"starts in {until} min"
    if persona is not None and ev.get("place") and until <= 6 * 60:
      tail += f" (~{travel_minutes(persona, ev['place'], maze)} min away)"
  else:
    tail = f"under way, {int((_abs_end(ev) - now).total_seconds() // 60)} min left"
  where = f" at {ev['place']}" if ev.get("place") else ""
  note = f" ({ev['note']})" if ev.get("note") else ""
  return f"#{ev['id']} {ev['title']}{where} — {_when(ev, now)}, {tail} [{ev['status']}]{note}"


def render(scratch, now=None, persona=None, maze=None, horizon_min=36 * 60, limit=5):
  """Text view for prompts and the check tool. '' when nothing is coming up."""
  now = now or scratch.curr_time
  rows = upcoming(scratch, now, horizon_min)[:limit]
  return "\n".join(describe(ev, now, persona, maze) for _, ev in rows)


def time_context(scratch, now=None, persona=None, maze=None):
  """'It is Tuesday 16:25. Coming up: ...' for choice/chat/replan prompts."""
  now = now or scratch.curr_time
  head = f"It is {now.strftime('%A %H:%M')}."
  body = render(scratch, now, persona, maze, horizon_min=24 * 60, limit=3)
  return head + (" Coming up on your agenda:\n" + body if body else "")


def due(scratch, now=None, persona=None, maze=None, lead_slack=10, statuses=LIVE):
  """Events the agent should be deciding about right now: starting within
  (travel + lead_slack) minutes, or already under way."""
  now = now or scratch.curr_time
  out = []
  for until, ev in upcoming(scratch, now, 12 * 60, statuses):
    lead = (travel_minutes(persona, ev.get("place"), maze) if persona is not None else 8) + lead_slack
    if until <= lead:
      out.append(ev)
  return out


def mark_past(scratch, now=None):
  """Events whose time is over: going -> done, known -> missed."""
  now = now or scratch.curr_time
  for ev in _store(scratch):
    try:
      over = _abs_end(ev) <= now
    except Exception:
      continue
    if over and ev["status"] == "going":
      ev["status"] = "done"
    elif over and ev["status"] == "known":
      ev["status"] = "missed"


# --------------------------------------------------------------------------
# follow: pin 'going' events into today's schedule
# --------------------------------------------------------------------------
def overlay(scratch, now=None):
  """Pin today's 'going' events into f_daily_schedule (once per edit; the
  'pinned' flag is cleared by add/modify/cancel and by reset_overlay after the
  daily schedule is regenerated). Returns the number of events pinned."""
  from persona.prompt_template.run_gpt_prompt import pin_commitment
  now = now or scratch.curr_time
  today = now.date().isoformat()
  n = 0
  for ev in _store(scratch):
    if ev["status"] != "going" or ev["date"] != today or ev.get("pinned") == today:
      continue
    now_min = now.hour * 60 + now.minute
    if ev["end"] <= now_min:
      continue
    start = max(ev["start"], now_min)
    sched = getattr(scratch, "f_daily_schedule", None) or []
    if not sched:
      continue
    label = ev["title"] + (f" at {ev['place']}" if ev.get("place") else "")
    scratch.f_daily_schedule = pin_commitment(sched, start, max(5, ev["end"] - start), label)
    ev["pinned"] = today
    n += 1
  return n


def reset_overlay(scratch):
  """Call after the daily schedule is rebuilt: events must be pinned again."""
  for ev in _store(scratch):
    ev.pop("pinned", None)


def trace(rec):
  """Append one decision/tool record to <GA_TRACE_DIR>/agenda.jsonl (fail-open)."""
  import json
  try:
    base = os.environ.get("GA_TRACE_DIR") or os.path.join(os.path.dirname(__file__), "../../ga_traces")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "agenda.jsonl"), "a") as f:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  except Exception:
    pass
