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


_WEEKDAY_RE = re.compile(
  r"^(mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|"
  r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b\.?")


def _norm_date(d, now):
  """ISO string for a date given as ISO, 'today', 'tomorrow', 'Feb 14',
  'February 14, 2023', a weekday name (alone or in front of a date:
  'Saturday, February 15'), or a date object; None if unparseable."""
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
  wd = _WEEKDAY_RE.match(t)
  if wd:
    # "Saturday, February 15": the stated date wins over the weekday name
    # (before, the weekday alone was read and the date dropped -> Feb 18)
    rest = t[wd.end():].lstrip(" ,").strip()
    if rest:
      stated = _norm_date(rest, now)
      if stated is not None:
        return stated
    ahead = (days.index(wd[1][:3]) - now.weekday()) % 7
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
  title; its status changes only if <status> is an explicit decision, not
  "known"). Returns the event dict, or None if the date/time cannot be read."""
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
      # Hearing about an event again ("known") changes nothing: it must not
      # bring back one the agent declined, or downgrade one it is going to.
      # Only an explicit decision (going, declined...) changes the status.
      if status != "known":
        ev["status"] = status
      for k, v in (("place", place), ("note", note)):
        if v and not ev.get(k):
          ev[k] = str(v)[:200]
      ev["end"] = max(ev["end"], e)
      unpin(scratch, ev)
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
  unpin(scratch, ev)       # the caller pins it again (overlay) if still going
  return ev


def cancel_event(scratch, ev_id, reason=""):
  ev = get_event(scratch, ev_id)
  if ev is None:
    return None
  ev["status"] = "declined"
  if reason:
    ev["note"] = str(reason)[:200]
  unpin(scratch, ev)
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


ASK_WHEN_UNSURE = ("If someone mentions an event or a plan and you do not know exactly which day, "
                   "what time or where, ask them (for example \"What day and what time?\") instead "
                   "of guessing.")


# --------------------------------------------------------------------------
# asking: get the missing day/time from the person who knows it
# --------------------------------------------------------------------------
_NUMW = r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_SOFT_TIME = re.compile(r"^(?:tonight|this (?:evening|afternoon))$", re.I)
_RANGE_TIME = re.compile(
  r"\bfrom\s+" + _NUMW + r"\s*(?:to|until|till|-|–)\s*" + _NUMW + r"\b|"
  r"\b(?:around|until|till)\s+" + _NUMW + r"\b(?!\s*(?:people|of|or|more|minutes?|hours?|days?|weeks?))", re.I)
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
# a day still ahead: "tomorrow", "on Friday", "February 14", "this weekend"
_DAY_CUE = re.compile(
  r"(?<!last )(?<!a )\b(?:tonight|tomorrow(?: (?:morning|afternoon|evening|night))?|"
  r"this (?:evening|weekend)|next week(?:end)?|"
  r"(?:this |next |on )?(?:mon|tues|wednes|thurs|fri|satur|sun)day(?: (?:morning|afternoon|evening|night))?|"
  + _MONTH + r"\.? \d{1,2}(?:st|nd|rd|th)?|(?:the )?\d{1,2}(?:st|nd|rd|th) of " + _MONTH + r")\b", re.I)
# something the two of them would do together, or an invitation
_JOINT = re.compile(
  r"\b(?:let(?:'|’)s|shall we|why don(?:'|’)t (?:we|you)|we (?:can|could|should)|"
  r"join (?:me|us)|come (?:to|by|over|along|with)|meet (?:you|me|up)|"
  r"you(?:'|’)?(?:re| are) (?:invited|welcome to)|invit(?:e|ing) you|"
  r"(?:would|do) you (?:like|want) to (?:come|join|meet))\b", re.I)
_ASKED_WHEN = re.compile(r"\b(?:what time|what day|when)\b[^.!?\n]*\?", re.I)
_TITLE_STOP = {"the", "and", "with", "for", "from", "day", "event"}


def states_clock_time(text):
  """True if the text states an hour: "at 5", "5:30 pm", "noon", "from 5 to 7",
  "around six". A part of the day ("tonight", "this afternoon") or a day name
  is not a clock time: an event cannot be recorded from it."""
  from persona.prompt_template.run_gpt_prompt import _TIME_CUE
  t = str(text or "")
  return (any(not _SOFT_TIME.match(m.group(0)) for m in _TIME_CUE.finditer(t))
          or bool(_RANGE_TIME.search(t)))


def _names_event(ev, line):
  """True if <line> names the event: at least two words of its title (place
  words and filler left out), and two thirds of them."""
  want = _words(ev.get("title", "")) - _words(ev.get("place", "")) - _TITLE_STOP
  if len(want) < 2:
    return False
  hit = len(want & _words(line))
  return hit >= 2 and hit * 3 >= len(want) * 2


def ask_hint(curr_chat, me, my_scratch=None, other_scratch=None, now=None):
  """One line for <me>'s chat prompt, telling it to ask the other speaker for
  the day and time they left out; '' when there is nothing to ask.

  It fires when nobody in the conversation has stated a clock time and the
  other speaker's last line either
    (a) names an event that is on THEIR agenda and not on mine (they know when
        it is, I do not), or
    (b) proposes something to do together on a coming day ("let's have lunch
        tomorrow", "come by on Friday").
  At most one question per conversation: after <me> asked when/what time, the
  hint stays off. It never contains the answer (the other agent has to say it).

  Why a situational line: the general sentence (ASK_WHEN_UNSURE) sits ~2,000
  tokens above the task. In n76 it raised when-questions overall, but only 7 of
  259 untimed day mentions were followed by one. A line placed right after the
  conversation, naming what was left out, got a question in 22 of 30 replays
  (4 of 30 without). A bare day word ("tonight", "this afternoon") is not a
  trigger: in those replays it produced questions like "What time is tonight?"."""
  try:
    turns = [(str(t[0]), str(t[1])) for t in (curr_chat or []) if len(t) >= 2]
  except Exception:
    return ""
  if not turns or turns[-1][0] == me:
    return ""
  other, last = turns[-1]
  if states_clock_time("\n".join(u for _, u in turns)):
    return ""
  if any(a == me and _ASKED_WHEN.search(u) for a, u in turns):
    return ""
  if other_scratch is not None:
    now = now or getattr(other_scratch, "curr_time", None)
    mine = _store(my_scratch) if my_scratch is not None else []
    try:
      theirs = [ev for _, ev in upcoming(other_scratch, now, horizon_min=14 * 24 * 60)]
    except Exception:
      theirs = []
    for ev in theirs:
      if _names_event(ev, last) and not any(_similar(m.get("title", ""), ev["title"]) for m in mine):
        return (f"{other} brought up \"{ev['title']}\" but has not said when it is. If you might "
                f"go, ask {other} which day and what time it is (and where, if you do not know).")
  cue = _DAY_CUE.search(last)
  if cue is not None and _JOINT.search(last):
    return (f"{other} suggested something for {cue.group(0)} but nobody has said at what time. "
            f"If you agree to it, ask {other} what time exactly (and where, if you do not know).")
  return ""


def answer_hint(curr_chat, me, my_scratch, now=None):
  """One line for <me>'s chat prompt when the other speaker just asked when
  something is and it is an event on MY agenda that the conversation named:
  the day, time and place as I recorded them, so I answer with them instead of
  making a time up. '' otherwise."""
  try:
    turns = [(str(t[0]), str(t[1])) for t in (curr_chat or []) if len(t) >= 2]
    if not turns or turns[-1][0] == me or my_scratch is None:
      return ""
    other, last = turns[-1]
    if not _ASKED_WHEN.search(last):
      return ""
    now = now or my_scratch.curr_time
    recent = "\n".join(u for _, u in turns[-4:])
    for _, ev in upcoming(my_scratch, now, horizon_min=14 * 24 * 60):
      if _names_event(ev, recent):
        where = f" at {ev['place']}" if ev.get("place") else ""
        day = _abs_start(ev).strftime("%A %B %d").replace(" 0", " ")
        return (f"{other} is asking when. On your agenda: \"{ev['title']}\"{where}, {day}, "
                f"{_when(ev, now)}. Tell {other} that day and that time.")
  except Exception:
    pass
  return ""


def chat_hints(curr_chat, me, my_scratch=None, other_scratch=None, now=None):
  """The lines placed right after the conversation in <me>'s chat prompt
  (GA_AGENDA_ASK, default ON): give the day and time when asked, ask for them
  when the other speaker left them out. '' when neither applies."""
  if os.environ.get("GA_AGENDA_ASK", "1") == "0":
    return ""
  return (answer_hint(curr_chat, me, my_scratch, now)
          or ask_hint(curr_chat, me, my_scratch, other_scratch, now))


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
def _slice(pieces, a, b):
  """The part of <pieces> ([[task, minutes], ...], starting at minute 0)
  between minutes a and b."""
  out, t = [], 0
  for task, d in pieces:
    lo, hi = max(t, a), min(t + d, b)
    t += d
    if hi > lo:
      out.append([task, hi - lo])
  return out


def _swap(pieces, base, label, src, src_base):
  """In <pieces> (starting at absolute minute <base>) replace every stretch
  named <label> by what <src> (starting at <src_base>) holds for the same
  minutes. A stretch <src> does not cover is kept. Neighbours that end up
  with the same task are joined, so a block that was split around a pinned
  event becomes one block again."""
  src_end = src_base + sum(d for _, d in src)
  out, t = [], base                      # out: [task, minutes, touched]
  for task, d in pieces:
    lo, hi = max(t, src_base), min(t + d, src_end)
    if task != label or hi <= lo:
      out.append([task, d, False])
    else:
      if lo > t:
        out.append([task, lo - t, True])
      out += [[x, m, True] for x, m in _slice(src, lo - src_base, hi - src_base)]
      if t + d > hi:
        out.append([task, t + d - hi, True])
    t += d
  joined = []
  for task, d, touched in out:
    if joined and joined[-1][0] == task and (touched or joined[-1][2]):
      joined[-1][1] += d
      joined[-1][2] = True
    else:
      joined.append([task, d, touched])
  return [[task, d] for task, d, _ in joined]


def unpin(scratch, ev):
  """Take the event's block out of today's schedule and put back what it had
  replaced. Called when the event is cancelled, changed, or no longer 'going'.
  Before, only the 'pinned' flag was cleared: a cancelled event stayed in
  f_daily_schedule (the agent still went), and a moved one was there twice.
  Returns True if the schedule changed."""
  pin = ev.pop("pin", None)
  ev.pop("pinned", None)
  try:
    sched = getattr(scratch, "f_daily_schedule", None)
    if not pin or not sched or pin["date"] != scratch.curr_time.date().isoformat():
      return False
    label, old, a = pin["label"], pin.get("replaced") or [], pin["start"]
    new = _swap(sched, 0, label, old, a)
    # another event pinned on top of this one remembers this label as what
    # it replaced: it must give back the original too, not this block
    for other in _store(scratch):
      op = other.get("pin")
      if other is not ev and op and op.get("date") == pin["date"]:
        op["replaced"] = _swap(op.get("replaced") or [], op["start"], label, old, a)
    if new == sched:
      return False
    scratch.f_daily_schedule = new
    return True
  except Exception:
    return False


def overlay(scratch, now=None):
  """Pin today's 'going' events into f_daily_schedule (once per edit; the
  'pinned' flag is cleared by add/modify/cancel and by reset_overlay after the
  schedule is rebuilt). ev['pin'] records the minutes taken and what was there
  before, so unpin() can give it back. Returns the number of events pinned."""
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
    minutes = max(5, ev["end"] - start)
    total = sum(d for _, d in sched)
    a = max(0, min(start, total - 5))            # the same clamps as pin_commitment
    b = min(a + minutes, total)
    under = _slice(sched, a, b)
    old = ev.get("pin")
    if old and old.get("date") == today:
      # pinned again after a replan that kept the block: what lies under it
      # is what the first pin replaced, not the block itself
      under = _swap(under, a, old["label"], old.get("replaced") or [], old["start"])
    org = getattr(scratch, "f_daily_schedule_hourly_org", None)
    if org and any(task == label for task, _ in under):
      under = _swap(under, a, label, _slice(org, a, b), a)     # last resort: the day's own plan
    scratch.f_daily_schedule = pin_commitment(sched, start, minutes, label)
    ev["pin"] = {"date": today, "start": a, "end": b, "label": label, "replaced": under}
    ev["pinned"] = today
    n += 1
  return n


def reset_overlay(scratch):
  """Call after the schedule is rebuilt: events must be pinned again. The
  record of what each pin replaced is kept; overlay() reuses it when the
  rebuilt schedule still holds the block."""
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
