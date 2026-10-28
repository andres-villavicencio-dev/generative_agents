"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: plan.py
Description: This defines the "Plan" module for generative agents. 
"""
import datetime
import math
import os
import random 
import re

# set by ReverieServer so planners can sign agents up for civic work
_ECON_REF = {}
import sys
import time
import threading

# PERF (improvement 2 — social attention budget): n30 telemetry showed 1,964
# decide_to_talk LLM calls firing in simultaneous waves (one per agent per
# perceived persona-event), all queued on CONVO_LOCK — the midday chat-cascade
# wall. Two mitigations:
#   1. A semaphore caps concurrent decide-to-talk/decide-to-react LLM calls, so
#      a busy cafe doesn't stampede the lane daemon (waves resolve 2-at-a-time
#      instead of 25-at-a-time).
#   2. lets_talk() short-circuits on low-poignancy events (poig < 3) before any
#      LLM call — banal stimuli never reach the model.
_SOCIAL_THINK_SEM = threading.Semaphore(int(os.environ.get("GA_SOCIAL_THINK_PAR", "2")))

# DURABLE CONVERSATION JOURNAL (Oct 21 2026): scratch chat state only hits
# disk at checkpoint saves (every 360 steps); a crash mid-conversation
# destroyed the only copy — the n68 first-real-conversation (Ayesha Khan /
# Latoya Williams, 10 exchanges) died with the process at 14:05 and was
# unrecoverable. generate_convo is atomic (one call returns the full turn
# list), so journaling the completed conversation at _chat_react time makes
# every chat crash-proof and gives the information-transfer objective a
# first-class dataset: who told what to whom, when. Fail-open: journaling
# must never cost the sim a conversation.
import json as _json
_CONVO_JOURNAL_PATH = os.environ.get(
    "GA_CONVO_JOURNAL",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..", "ga_traces", "conversations.jsonl")))
_CONVO_JOURNAL_LOCK = threading.Lock()

def _journal_conversation(init_persona, target_persona, convo, summary,
                          duration_min):
  """Append one completed conversation to ga_traces/conversations.jsonl."""
  try:
    rec = {
        "ts": time.time(),
        "sim": os.path.basename(init_persona.scratch.sim_folder or ""),
        "sim_time": str(init_persona.scratch.curr_time),
        "init": init_persona.name,
        "target": target_persona.name,
        "summary": summary,
        "duration_min": duration_min,
        "turns": [[row[0], row[1]] for row in (convo or [])],
    }
    line = _json.dumps(rec, ensure_ascii=False)
    with _CONVO_JOURNAL_LOCK:
      with open(_CONVO_JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
  except Exception as _e:
    print(f"[ConvoJournal] write failed (fail-open): {_e}")

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.run_gpt_prompt import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.converse import *
from persona.cognitive_modules.event_triple_parser import parse_event_triple
from collections import OrderedDict

##############################################################################
# CHAPTER 2: Generate
##############################################################################

def generate_wake_up_hour(persona):
  """
  Generates the time when the persona wakes up. This becomes an integral part
  of our process for generating the persona's daily plan.
  
  Persona state: identity stable set, lifestyle, first_name

  INPUT: 
    persona: The Persona class instance 
  OUTPUT: 
    an integer signifying the persona's wake up hour
  EXAMPLE OUTPUT: 
    8
  """
  if debug: print ("GNS FUNCTION: <generate_wake_up_hour>")
  return int(run_gpt_prompt_wake_up_hour(persona)[0])


_PLAN_T = r'(\d{1,2})(?::(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?'
_WORKLIKE_RE = re.compile(
  r'\b(work|shift|store|shop|counter|caf[eé]|class|lecture|office|study|'
  r'research|practic|rehears|paint|writ|compos|open|serv|manag|teach|'
  r'pharmac|clinic|restock|stock|sell|sales|customer)\w*', re.I)
_PLAN_FILLERS = ["relaxing and unwinding at home",
                 "tidying up their home",
                 "taking a short stroll nearby",
                 "browsing on their phone"]
# Quick chores never stretch across open time (Klaus showered 3h in test).
_SHORT_ACT_RE = re.compile(
  r'\b(shower|bath|brush|wash|drink|breakfast|lunch|dinner|meal|snack|eat|'
  r'coffee|tea|restroom|bathroom|toilet|dress|get dressed)\w*', re.I)


def _plan_ampm(s):
  if not s:
    return None
  return 'am' if s.strip().lower().startswith('a') else 'pm'


def _plan_hour(h, m, ampm):
  h = int(h); m = int(m) if m else 0
  if ampm == 'pm' and h != 12:
    h += 12
  elif ampm == 'am' and h == 12:
    h = 0
  return h, m


def parse_plan_item(req):
  """One daily_req line -> (start_hour|None, end_hour|None, activity).

  FIX (Sep 2026, filler epidemic): the old parser knew only 'at X am/pm'
  and 'from X am to Y pm' (both sides with am/pm) and DISCARDED end times,
  so '1:00 pm ... close store at 8:00 pm' became 2h work + 5h phone/stroll
  fillers, and 'until noon' / 'around 5 pm' / 'between 1 and 4 pm' /
  '7:00 to 8:00 pm' were misplaced or lost. end_hour is exclusive and
  rounded UP when minutes are present (until 4:30 pm -> 17)."""
  text = re.sub(r'\b(?:noon|midday)\b', '12:00 pm', req or '', flags=re.I)
  text = re.sub(r'\bmidnight\b', '12:00 am', text, flags=re.I)
  T = _PLAN_T
  start = end = None
  spans = []

  m = re.search(r'\b(?:from|between)\s+' + T + r'\s*(?:to|until|till|and|-|–)\s*' + T,
                text, re.I)
  if not m:
    m = re.search(r'(?<![\d:])' + T + r'\s*(?:-|–|to)\s*' + T, text, re.I)
    if m and not (m.group(3) or m.group(6)):
      m = None
  if m:
    a1, a2 = _plan_ampm(m.group(3)), _plan_ampm(m.group(6))
    if a2 is None:
      a2 = a1
    if a1 is None:
      a1 = a2
      if (_plan_hour(m.group(1), m.group(2), a1)[0] >
          _plan_hour(m.group(4), m.group(5), a2)[0]) and a2 == 'pm':
        a1 = 'am'   # "from 11 to 1 pm" = 11am-1pm
    sh, _ = _plan_hour(m.group(1), m.group(2), a1)
    eh, em = _plan_hour(m.group(4), m.group(5), a2)
    if eh == 0:
      eh = 24
    if em:
      eh += 1
    start, end = sh, eh
    spans.append(m.span())
  else:
    me = re.search(r'\b(?:until|till|by|before)\s+' + T, text, re.I)
    if me and me.group(3):
      eh, em = _plan_hour(me.group(1), me.group(2), _plan_ampm(me.group(3)))
      if eh == 0:
        eh = 24
      end = eh + (1 if em else 0)
      spans.append(me.span())
    ms = re.search(r'\b(?:at|around|about|after|approximately|approx\.?|~)\s+' + T,
                   text, re.I)
    if ms and ms.group(3):
      start = _plan_hour(ms.group(1), ms.group(2), _plan_ampm(ms.group(3)))[0]
      spans.append(ms.span())
    elif start is None:
      mb = re.search(r'(?<![\d:])(\d{1,2})(?::(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)(?![a-z])',
                     text, re.I)
      if mb and not any(a <= mb.start() < b for a, b in spans):
        start = _plan_hour(mb.group(1), mb.group(2), _plan_ampm(mb.group(3)))[0]
        spans.append(mb.span())
  if start is not None and end is not None and end <= start:
    end = None

  act = text
  for a, b in sorted(spans, reverse=True):
    act = act[:a] + ' ' + act[b:]
  act = re.sub(r'\s+', ' ', act).strip()
  act = re.sub(r'^(?:at|from|until|to|between|around|by|and)\b\s*', '', act, flags=re.I)
  act = re.sub(r'\s*\b(?:at|from|until|to|between|around|by|and|in the)\s*$', '', act, flags=re.I)
  act = act.strip(' .,;:-')
  return start, end, act


def build_schedule_from_req(daily_req, wake_up_hour, name=''):
  """daily_req -> 24 hourly activity strings (slot h = hour h).

  Priority: explicit ranges fill their whole span; point-anchored items
  own their start hour and run until the next item (<=4h, or <=8h for
  work-like blocks - a store shift, classes); anything left over after
  the plan is exhausted gets rotating fillers (consecutive fillers differ
  so hour-compression can't merge them into one giant block)."""
  items = []
  last_start = last_end = None
  for idx, req in enumerate(daily_req or []):
    s, e, act = parse_plan_item(req)
    if not act:
      continue
    if s is None:
      if last_end is not None:
        s = last_end
      elif last_start is not None:
        s = last_start + 1
    if s is None or s > 23:
      continue
    if e is not None and e <= s:
      e = None
    items.append((s, e, idx, act))
    last_start, last_end = s, e

  # sleep hour: an explicit evening bed/sleep item wins; else stock rule
  sleep_hour = None
  for s, e, idx, act in items:
    if s >= 20 and re.search(r'\b(sleep|bed|bedtime)\b', act, re.I):
      sleep_hour = s
  if sleep_hour is None:
    covered = [(e if e is not None else s + 1) for s, e, _, _ in items]
    sleep_hour = min(max(max(covered) if covered else 21, 21), 23)
  sleep_hour = max(sleep_hour, wake_up_hour + 1)

  slots = [None] * 24
  # 1) explicit ranges, plan order (later lines override earlier ones)
  for s, e, idx, act in sorted(items, key=lambda x: x[2]):
    if e is not None:
      for h in range(s, min(e, 24)):
        slots[h] = act
  # 2) point anchors own their start hour (lunch at 12 inside a work range)
  point_starts = sorted({s for s, e, _, _ in items})
  for s, e, idx, act in sorted(items, key=lambda x: x[2]):
    if e is None:
      slots[s] = act
  # 3) open-ended items run until the next item starts
  for s, e, idx, act in sorted(items, key=lambda x: (x[0], x[2])):
    if e is not None:
      continue
    nxt = [p for p in point_starts if p > s]
    stop = min(nxt[0] if nxt else sleep_hour, sleep_hour)
    limit = (8 if _WORKLIKE_RE.search(act)
             else 1 if _SHORT_ACT_RE.search(act) else 3)
    for h in range(s + 1, min(stop, s + limit, 24)):
      if slots[h] is not None:
        break
      slots[h] = act

  out = []
  first_item_h = min((s for s, _, _, _ in items), default=None)
  wake_fills = 0
  for h in range(24):
    if h < wake_up_hour:
      out.append("sleeping")
    elif h >= sleep_hour:
      out.append("going to bed and sleeping")
    elif slots[h]:
      out.append(slots[h])
    elif (first_item_h is None or h < first_item_h or h <= wake_up_hour + 1) and wake_fills < 2:
      out.append("waking up and completing morning routine")
      wake_fills += 1
    else:
      out.append(_PLAN_FILLERS[h % len(_PLAN_FILLERS)])
  if name:
    print(f"[schedule-deterministic v2] {name}: wake={wake_up_hour} "
          f"sleep={sleep_hour} items={len(items)}")
  return out


def generate_first_daily_plan(persona, wake_up_hour, resource_manager=None):
  """
  Generates the daily plan for the persona.
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form:
  'wake up and complete the morning routine at 6:00 am',
  'eat breakfast at 7:00 am',..
  Note that the actions come without a period.

  Persona state: identity stable set, lifestyle, cur_data_str, first_name

  INPUT:
    persona: The Persona class instance
    wake_up_hour: an integer that indicates when the hour the persona wakes up
                  (e.g., 8)
    resource_manager: Optional WorldResourceManager for resource context (Phase 3.2)
  OUTPUT:
    a list of daily actions in broad strokes.
  EXAMPLE OUTPUT:
    ['wake up and complete the morning routine at 6:00 am',
     'have breakfast and brush teeth at 6:30 am',
     'work on painting project from 8:00 am to 12:00 pm',
     'have lunch at 12:00 pm',
     'take a break and watch TV from 2:00 pm to 4:00 pm',
     'work on painting project from 4:00 pm to 6:00 pm',
     'have dinner at 6:00 pm', 'watch TV from 7:00 pm to 8:00 pm']
  """
  if debug: print ("GNS FUNCTION: <generate_first_daily_plan>")
  return run_gpt_prompt_daily_plan(persona, wake_up_hour, resource_manager=resource_manager)[0]


def generate_hourly_schedule(persona, wake_up_hour): 
  """
  Based on the daily req, creates an hourly schedule -- one hour at a time. 
  The form of the action for each of the hour is something like below: 
  "sleeping in her bed"
  
  The output is basically meant to finish the phrase, "x is..."

  Persona state: identity stable set, daily_plan

  INPUT: 
    persona: The Persona class instance 
    persona: Integer form of the wake up hour for the persona.  
  OUTPUT: 
    a list of activities and their duration in minutes: 
  EXAMPLE OUTPUT: 
    [['sleeping', 360], ['waking up and starting her morning routine', 60], 
     ['eating breakfast', 60],..
  """
  if debug: print ("GNS FUNCTION: <generate_hourly_schedule>")

  hour_str = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM", 
              "05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM", 
              "10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", 
              "03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
              "08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
  # FIX: Build schedule ENTIRELY from daily_req — no LLM calls needed.
  # The original code called LLM 18 times per agent per day, but small models (gemma3,
  # qwen3) pattern-match the leading "sleeping" entries and output "sleeping" for every
  # waking hour too. The daily_req list has explicit times ("at 8:00 am") that are
  # authoritative. We use those directly and skip 54+ redundant LLM calls on day 1.
  def _build_req_time_map(daily_req):
    """Parse daily_req list → sorted [(start_hour, activity_text)] pairs.

    Handles multiple time patterns:
      - "at 8:00 am" → start at that hour
      - "from 8:00 am to 8:00 pm" → start at first hour
      - "until 8:00 pm" → this is an END time, not a start; skip it
        (the activity before it should forward-fill to cover this range)

    Uses position in daily_req as tiebreaker so later items override
    earlier ones at the same hour (preserving the plan's narrative order).
    """
    import re

    def _parse_hour(hour_str, ampm):
      h = int(hour_str); ampm = ampm.lower()
      if ampm == 'pm' and h != 12: h += 12
      elif ampm == 'am' and h == 12: h = 0
      return h

    pairs = []
    last_h = None
    for idx, req in enumerate(daily_req):
      h = None
      # Pattern 1: "from X am to Y pm" — use start time
      m = re.search(r'\bfrom\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+to\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', req, re.IGNORECASE)
      if m:
        h = _parse_hour(m.group(1), m.group(3))
      # Pattern 2: "at X:XX am/pm"
      if h is None:
        m = re.search(r'\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', req, re.IGNORECASE)
        if m:
          h = _parse_hour(m.group(1), m.group(3))
      # Pattern 3: "until X pm" — activity runs UNTIL this time, infer start
      # from the previous item's hour + 1
      if h is None:
        m = re.search(r'\buntil\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b', req, re.IGNORECASE)
        if m:
          # Use next hour after last known anchor as start
          h = (last_h + 1) if last_h is not None else None

      # Items with no time at all: assign next hour after previous
      if h is None and last_h is not None:
        h = last_h + 1

      if h is not None:
        # Strip all time expressions from the activity text
        act = re.sub(r'\b(?:at|from|until|to)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b', '', req, flags=re.IGNORECASE).strip()
        act = re.sub(r'\s+', ' ', act).strip().rstrip('.,').strip()
        if act:
          pairs.append((h, idx, act))
          last_h = h
    # Sort by hour, then by position in daily plan (later items override earlier at same hour)
    pairs.sort(key=lambda x: (x[0], x[1]))
    # Deduplicate: at each hour, keep only the last item (by plan order)
    seen = {}
    for h, idx, act in pairs:
      seen[h] = act  # last one wins
    return sorted(seen.items())

  def _build_schedule_from_req(daily_req, wake_up_hour):
    """Build 24-slot hourly activity list directly from daily_req + wake_up_hour.
    
    Each slot corresponds to the hour at that index (slot 0 = 00:00, slot 8 = 08:00).
    Slots before wake_up_hour = 'sleeping'.
    Remaining slots filled from req_time_map using forward-fill.
    Final hours default to 'going to bed and sleeping'.
    """
    req_map = _build_req_time_map(daily_req)
    schedule = []
    
    # Determine sleep hour: last req hour + 2 hours buffer, capped at 23
    if req_map:
      last_req_hour = max(h for h, _ in req_map)
      # FIX: minimum sleep hour = 21 (9pm) so agents with sparse req maps (e.g. only
      # "study at 8am") don't get a sleep_hour of 10, which makes them "go to bed" mid-morning.
      sleep_hour = max(last_req_hour + 2, 21)
      sleep_hour = min(sleep_hour, 23)
    else:
      sleep_hour = 22

    def _req_for_hour(h):
      """Forward-fill: return last req whose start hour <= h."""
      act = None
      for rh, ra in req_map:
        if rh <= h:
          act = ra
        else:
          break
      return act

    # FIX A (duration sanity): the original forward-fill spread one anchored
    # activity across EVERY un-anchored hour until sleep_hour — Eddy's
    # "Practice piano for an hour" became a 420-min block, Isabella's nap
    # 360 min, freezing agents for half a sim-day on one action. Now any
    # forward-filled stretch is capped at 2 hours; the gap after it gets
    # "relaxing at home" instead of an endless repeat of the same activity.
    CAP_FILL_HOURS = 2
    _fill_anchor = None   # (req_hour, act) the stretch started from
    _fill_len = 0
    # Rotating fillers: consecutive fill hours must differ, else the
    # hour-compression step merges them back into ONE giant block (the
    # 8h "relaxing" merge the first offline CALL test caught).
    _FILLERS = ["relaxing and unwinding at home",
                "tidying up their home",
                "taking a short stroll nearby",
                "browsing on their phone"]
    _wake_fills = 0
    for h in range(24):
      if h < wake_up_hour:
        schedule.append("sleeping")
      elif h >= sleep_hour:
        schedule.append("going to bed and sleeping")
      else:
        act = _req_for_hour(h)
        if act and (_fill_anchor is None or _fill_anchor[1] != act):
          _fill_anchor = (h, act)
          _fill_len = 1
          schedule.append(act)
          _wake_fills = 0
        elif act and _fill_len < CAP_FILL_HOURS:
          _fill_len += 1
          schedule.append(act)
        elif act:
          _fill_len += 1
          # Past the cap: rotate fillers so no two consecutive hours match.
          schedule.append(_FILLERS[h % len(_FILLERS)])
        else:
          # Before first req starts but after wake_up — wake-up routine,
          # itself capped at 2 hours, then rotating leisure fillers.
          if _wake_fills < CAP_FILL_HOURS:
            schedule.append("waking up and completing morning routine")
          else:
            schedule.append(_FILLERS[h % len(_FILLERS)])
          _wake_fills += 1

    print(f"[schedule-deterministic] {persona.scratch.name}: wake={wake_up_hour} sleep={sleep_hour} req_anchors={len(req_map)}")
    for h, act in enumerate(schedule):
      if h >= wake_up_hour:
        print(f"  {h:02d}:00 → {act}")
    return schedule

  # v2 builder (module-level, parses ranges/end times) — see parse_plan_item.
  n_m1_activity = build_schedule_from_req(persona.scratch.daily_req,
                                          wake_up_hour, persona.scratch.name)
  
  # Step 1. Compressing the hourly schedule to the following format: 
  # The integer indicates the number of hours. They should add up to 24. 
  # [['sleeping', 6], ['waking up and starting her morning routine', 1], 
  # ['eating breakfast', 1], ['getting ready for the day', 1], 
  # ['working on her painting', 2], ['taking a break', 1], 
  # ['having lunch', 1], ['working on her painting', 3], 
  # ['taking a break', 2], ['working on her painting', 2], 
  # ['relaxing and watching TV', 1], ['going to bed', 1], ['sleeping', 2]]
  _n_m1_hourly_compressed = []
  prev = None 
  prev_count = 0
  for i in n_m1_activity: 
    if i != prev:
      prev_count = 1 
      _n_m1_hourly_compressed += [[i, prev_count]]
      prev = i
    else: 
      if _n_m1_hourly_compressed: 
        _n_m1_hourly_compressed[-1][1] += 1

  # Step 2. Expand to min scale (from hour scale)
  # [['sleeping', 360], ['waking up and starting her morning routine', 60], 
  # ['eating breakfast', 60],..
  n_m1_hourly_compressed = []
  for task, duration in _n_m1_hourly_compressed: 
    n_m1_hourly_compressed += [[task, duration*60]]

  return n_m1_hourly_compressed


def generate_task_decomp(persona, task, duration): 
  """
  A few shot decomposition of a task given the task description 

  Persona state: identity stable set, curr_date_str, first_name

  INPUT: 
    persona: The Persona class instance 
    task: the description of the task at hand in str form
          (e.g., "waking up and starting her morning routine")
    duration: an integer that indicates the number of minutes this task is 
              meant to last (e.g., 60)
  OUTPUT: 
    a list of list where the inner list contains the decomposed task 
    description and the number of minutes the task is supposed to last. 
  EXAMPLE OUTPUT: 
    [['going to the bathroom', 5], ['getting dressed', 5], 
     ['eating breakfast', 15], ['checking her email', 5], 
     ['getting her supplies ready for the day', 15], 
     ['starting to work on her painting', 15]] 

  FIX: cap duration at 120 minutes before sending to LLM. Large durations
  (e.g. 720 min "working at the cafe") cause qwen3 to generate 144 subtasks
  and time out. We clamp to 2 hours max — the caller will expand the last
  entry to fill remaining time anyway (see run_gpt_prompt_task_decomp).

  """
  if debug: print ("GNS FUNCTION: <generate_task_decomp>")
  # FIX: cap duration passed to LLM at 120 min to avoid timeouts on huge blocks
  capped_duration = min(duration, 120)
  if capped_duration < duration:
    print(f"[task_decomp] capping duration {duration}min → {capped_duration}min for: {task[:60]}")
  result = run_gpt_prompt_task_decomp(persona, task, capped_duration)[0]
  # Scale the decomposed subtask durations back up proportionally so total = original duration
  if result and capped_duration < duration:
    scale = duration / capped_duration
    result = [[t, max(5, round(d * scale / 5) * 5)] for t, d in result]
    # Adjust last item to make total match exactly
    total = sum(d for _, d in result)
    if total != duration and result:
      result[-1][1] += duration - total
  return result


def generate_action_sector(act_desp, persona, maze): 
  """TODO 
  Given the persona and the task description, choose the action_sector. 

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT: 
    act_desp: description of the new action (e.g., "sleeping")
    persona: The Persona class instance 
  OUTPUT: 
    action_arena (e.g., "bedroom 2")
  EXAMPLE OUTPUT: 
    "bedroom 2"
  """
  if debug: print ("GNS FUNCTION: <generate_action_sector>")
  return run_gpt_prompt_action_sector(act_desp, persona, maze)[0]


def generate_action_arena(act_desp, persona, maze, act_world, act_sector): 
  """TODO 
  Given the persona and the task description, choose the action_arena. 

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT: 
    act_desp: description of the new action (e.g., "sleeping")
    persona: The Persona class instance 
  OUTPUT: 
    action_arena (e.g., "bedroom 2")
  EXAMPLE OUTPUT: 
    "bedroom 2"
  """
  if debug: print ("GNS FUNCTION: <generate_action_arena>")
  return run_gpt_prompt_action_arena(act_desp, persona, maze, act_world, act_sector)[0]


def generate_action_game_object(act_desp, act_address, persona, maze):
  """TODO
  Given the action description and the act address (the address where
  we expect the action to task place), choose one of the game objects. 

  Persona state: identity stable set, n-1 day schedule, daily plan

  INPUT: 
    act_desp: the description of the action (e.g., "sleeping")
    act_address: the arena where the action will take place: 
               (e.g., "dolores double studio:double studio:bedroom 2")
    persona: The Persona class instance 
  OUTPUT: 
    act_game_object: 
  EXAMPLE OUTPUT: 
    "bed"
  """
  if debug: print ("GNS FUNCTION: <generate_action_game_object>")
  if not persona.s_mem.get_str_accessible_arena_game_objects(act_address): 
    return "<random>"
  return run_gpt_prompt_action_game_object(act_desp, persona, maze, act_address)[0]


def _zlib_pref(name, n):
  """Deterministic per-persona bucket in [0, n)."""
  import zlib
  return zlib.crc32(name.encode()) % n


def _laya_shadow_location(act_desp, persona, maze, sector, arena):
  """Keep laya's spatial training pairs flowing under the unified call:
  ask laya (off-thread, shadow) the same three questions; its picks land in
  ga_traces/spatial.jsonl next to the teacher's unified answer."""
  try:
    import persona.prompt_template.jev_scoring as _J
    if not (_J.USE_JEV_SCORING and _J.JEV_BACKEND == "laya"):
      return
    world = maze.access_tile(persona.scratch.curr_tile)["world"]
    tree = persona.s_mem.tree.get(world, {})
    act = act_desp.split("(")[0].strip()
    name = persona.scratch.name
    secs = [x for x in tree if "'s house" not in x or persona.scratch.last_name in x]
    jobs = [(f"{name} is in {maze.access_tile(persona.scratch.curr_tile)['sector']}.\n"
             f"Activity: {act}\nWhich area should they go to?\n__JEV_OPTIONS__", secs)]
    if sector in tree:
      jobs.append((f"{name} is going to {sector}.\nActivity: {act}\n"
                   f"Which room should they use?\n__JEV_OPTIONS__", list(tree[sector].keys())))
      if arena in tree[sector]:
        objs = [o for o in (tree[sector][arena] or []) if o]
        if objs:
          jobs.append((f"{name} is in {arena}.\nActivity: {act}\n"
                       f"Which object should they use?\n__JEV_OPTIONS__", objs))
    for prompt, opts in jobs:
      if len(opts) > 1:
        _J._shadow(_J._laya_score_choice, prompt, opts, 0.0, False)
  except Exception:
    pass


_VENUE_TOKENS = ("harvey oak supply store", "supply store", "hobbs cafe", "hobbs café",
                 "willows market", "rose and crown")


def _venue_in_activity(act_desp, persona, act_world):
  """The persona-known sector an activity explicitly names ("... at/to <venue>"),
  or None."""
  import re as _re
  low = (act_desp or "").lower()
  tree = persona.s_mem.tree.get(act_world, {})
  for tok in _VENUE_TOKENS:
    if _re.search(r"(\b(at|to|in)\s+|\()(the )?" + _re.escape(tok), low):
      key = tok.replace("é", "e")
      for sec in tree:
        if key.split()[0] in sec.lower().replace("é", "e") and (
            key.split()[-1] in sec.lower().replace("é", "e")):
          return sec
  return None


def resolve_action_location(act_desp, persona, maze):
  """(world, sector, arena, game_object) for an action. PERF (Oct 21 2026):
  one unified cloud call (run_gpt_prompt_action_location) instead of the
  sector -> arena -> object chain; any level it can't validate falls back to
  the legacy per-level call. GA_UNIFIED_LOCATION=0 restores the chain."""
  import re as _re
  act_world = maze.access_tile(persona.scratch.curr_tile)["world"]
  sector = arena = obj = None
  if os.environ.get("GA_UNIFIED_LOCATION", "1") == "1":
    try:
      sector, arena, obj = run_gpt_prompt_action_location(act_desp, persona, maze)
    except Exception as e:
      print(f"[Location] unified call failed ({type(e).__name__}: {e}); legacy chain")
      sector = arena = obj = None
    if sector:
      _laya_shadow_location(act_desp, persona, maze, sector, arena)
  # An activity that names its venue ("buying groceries at the Harvey Oak Supply
  # Store") must happen there: a small model picked the home fridge instead,
  # and the "stay in the current area" rule then pinned the agent at it for
  # hours. Override the area; the room/object are re-resolved for it.
  # Civic tasks name a sector + room ("helping out at the Oak Hill College
  # library"); the model resolved it to "Dorm for Oak Hill College" (the
  # fuzzy "oak hill college" match), so the worker sat in a dorm room and the
  # civic presence check never matched (Klaus, communityexp5: wage unpaid).
  try:
    import economy as _eco
    _low = (act_desp or "").lower()
    _tree = persona.s_mem.tree.get(act_world, {})
    for _t in _eco.CIVIC_TASKS.values():
      if _low.lstrip().startswith(_t["name"].lower()):   # the civic goal text itself, not any mention of the place
        _sec = next((x for x in _tree if x.lower() == _t["keys"][0]), None)
        if _sec:
          _ar = next((a for a in _tree[_sec] if a.lower() == _t["keys"][-1]), None)
          if _ar or len(_t["keys"]) == 1 or _t["keys"][-1] == _t["keys"][0]:
            if str(sector).lower() != _sec.lower() or (_ar and str(arena).lower() != _ar.lower()):
              print(f"[Location] {persona.name}: civic '{act_desp[:50]}' resolved to "
                    f"'{sector}' -> forcing '{_sec}:{_ar}'")
            sector, arena, obj = _sec, _ar, None
        break
  except Exception:
    pass
  # Job board (GA_JOBBOARD): "check the community bulletin board" means standing at the
  # board in Johnson Park; the model otherwise resolved it to a home object (a sleeper
  # "read" the board from bed).
  try:
    import economy as _eco2, re as _re2
    _low2 = (act_desp or "").lower()
    if (_eco2.jobboard_enabled() and "bulletin board" in _low2
        and _re2.search(r"\b(check|read|reading|visit|look|stop by|go to|head to|browse|review|consult)", _low2)):
      _tree2 = persona.s_mem.tree.get(act_world, {})
      _sec2 = next((x for x in _tree2 if x.lower() == "johnson park"), None)
      if _sec2:
        _ar2 = next((a for a in _tree2[_sec2] if a.lower() == "park"), None)
        if _ar2:
          _ob2 = next((o for o in _tree2[_sec2][_ar2] if "bulletin" in o.lower()), None)
          sector, arena, obj = _sec2, _ar2, _ob2
  except Exception:
    pass
  _forced = None if (sector and arena) else _venue_in_activity(act_desp, persona, act_world)
  if _forced and sector and _forced.lower() != str(sector).lower():
    print(f"[Location] {persona.name}: '{act_desp[:50]}' resolved to "
          f"'{sector}' -> forcing '{_forced}'")
    sector, arena, obj = _forced, None, None
  if not sector:
    sector = generate_action_sector(act_desp, persona, maze)
  if not arena:
    arena = generate_action_arena(act_desp, persona, maze, act_world, sector)
    arena = _re.sub(r'^Answer:\s*', '', arena, flags=_re.IGNORECASE).strip()
    arena = arena.lstrip("{[(\"'").rstrip("}])\"'").strip()
  if not obj:
    obj = generate_action_game_object(act_desp, f"{act_world}:{sector}:{arena}",
                                      persona, maze)
  return act_world, sector, arena, obj


EMOJI_LOOKUP = {
  "sleep": "😴", "sleeping": "😴", "nap": "😴", "napping": "😴",
  "eat": "🍽️", "eating": "🍽️", "breakfast": "🍳", "lunch": "🥪", "dinner": "🍽️",
  "cook": "🧑‍🍳", "cooking": "🧑‍🍳", "preparing meal": "🧑‍🍳",
  "work": "💼", "working": "💼", "study": "📚", "studying": "📚",
  "read": "📖", "reading": "📖", "write": "✍️", "writing": "✍️",
  "walk": "🚶", "walking": "🚶",
  "talk": "💬", "talking": "💬", "chat": "💬", "chatting": "💬", "conversation": "💬",
  "shower": "🚿", "showering": "🚿", "bath": "🛁",
  "brush": "🪥", "brushing teeth": "🪥",
  "dress": "👔", "getting dressed": "👔", "changing clothes": "👔",
  "coffee": "☕", "drinking coffee": "☕", "tea": "🍵",
  "exercise": "🏃", "exercising": "🏃", "workout": "💪",
  "paint": "🎨", "painting": "🎨", "draw": "✏️", "drawing": "✏️",
  "music": "🎵", "listen": "🎧", "listening": "🎧", "play piano": "🎹",
  "garden": "🌱", "gardening": "🌱", "water plant": "🌱",
  "clean": "🧹", "cleaning": "🧹",
  "shop": "🛒", "shopping": "🛒", "buy": "🛒",
  "rest": "😌", "resting": "😌", "relax": "😌", "relaxing": "😌",
  "sit": "🪑", "sitting": "🪑",
  "think": "🤔", "thinking": "🤔", "contemplate": "🤔",
  "phone": "📱", "call": "📞",
  "tidying": "🧹", "stroll": "🚶",
  "computer": "💻", "laptop": "💻", "browse": "💻",
  "watch": "📺", "watching": "📺", "tv": "📺",
  "wait": "⌛", "waiting": "⌛", "idle": "⌛",
  "wake up": "⏰", "waking up": "⏰",
  "get ready": "🧑", "morning routine": "🧑",
  # object states (act_obj_desp) — the bulk of the old LLM fallbacks
  "slept": "😴", "occupied": "🚫", "in use": "🛠️", "being used": "🛠️",
  "explored": "🔍", "explore": "🔍", "empty": "▫️", "idle object": "▫️",
  "drink": "🥤", "water": "💧", "lively": "🎉", "spend time with": "🎉",
  "holding": "✋", "notebook": "📓", "book": "📕", "grocer": "🛒",
}


def _lookup_pronunciatio(description):
  """Try to match action description to emoji via lookup table.
  Returns emoji string or None if no match found.

  FIX (perf 6): the old paren-stripping took the INSIDE of the first
  parenthetical — "browsing on their phone (1)" stripped to "1", a
  guaranteed lookup miss that pushed ~40% of pronunciatio calls to the
  LLM (measured on n30: 59% hit rate instead of 75%). Now: strip the
  leading main-clause BEFORE the first paren, plus trailing
  annotation parens like "(1)"/"(2. ...)".
  """
  desc = description.lower().strip()
  if "@" in desc:  # drop trailing location: "cooking @ kitchen"
    desc = desc.split("@")[0].strip()
  if "(" in desc:
    # main clause before the first paren is the real action
    head = desc.split("(")[0].strip()
    if not head:  # paren-led description — fall back to old behavior
      head = desc.split("(")[-1].split(")")[0].strip()
    # kill residual trailing annotation parens on the head, e.g. "watch tv (1) (2)"
    import re as _re
    head = _re.sub(r"\s*\([^)]*\)\s*", " ", head).strip()
    desc = head if head else desc
  if desc in EMOJI_LOOKUP:
    return EMOJI_LOOKUP[desc]
  for keyword, emoji in EMOJI_LOOKUP.items():
    if keyword in desc:
      return emoji
  return None


def generate_action_pronunciatio(act_desp, persona):
  """Given an action description, creates an emoji string description.
  Uses a lookup table for common actions, falling back to LLM for novel ones.

  INPUT:
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT:
    a string of emoji that translates action description.
  EXAMPLE OUTPUT:
    "🧈🍞"
  """
  if debug: print ("GNS FUNCTION: <generate_action_pronunciatio>")

  lookup = _lookup_pronunciatio(act_desp)
  if lookup:
    return lookup

  # The emoji is purely visual (sprite bubble) and never feeds back into
  # cognition; the LLM fallback cost ~10s of cloud time per miss on the
  # step's critical path (Oct 21 2026). GA_PRONUNCIATIO_LLM=1 re-enables it.
  if os.environ.get("GA_PRONUNCIATIO_LLM", "0") != "1":
    return "🙂"

  try:
    x = run_gpt_prompt_pronunciatio(act_desp, persona)[0]
  except:
    x = "🙂"

  if not x:
    return "🙂"
  return x


def _parse_event_triple(action_description, persona_name):
  """Parse (subject, predicate, object) from action description via regex.
  Returns None only when there is no " is " in the text. The parser lives in
  event_triple_parser (import-light, unit-testable)."""
  return parse_event_triple(action_description, persona_name)


# The triple is only a memory-index key, so identical descriptions from the
# same persona never need a second LLM call (bounded; Oct 21 2026).
_EVENT_TRIPLE_CACHE = OrderedDict()
_EVENT_TRIPLE_CACHE_MAX = 2048
_EVENT_TRIPLE_LOCK = threading.Lock()


def generate_action_event_triple(act_desp, persona):
  if debug: print ("GNS FUNCTION: <generate_action_event_triple>")
  parsed = _parse_event_triple(act_desp, persona.name)
  if not parsed and " is " not in f" {act_desp} ":
    # Callers pass the bare activity ("buying groceries at ..."); the LLM
    # prompt itself builds "<name> is <activity>" — parse that same
    # sentence (without it ~98% of inputs still fell through to the LLM).
    parsed = _parse_event_triple(f"{persona.name} is {act_desp}", persona.name)
  if parsed:
    return parsed
  key = (persona.name, act_desp)
  with _EVENT_TRIPLE_LOCK:
    if key in _EVENT_TRIPLE_CACHE:
      _EVENT_TRIPLE_CACHE.move_to_end(key)
      return _EVENT_TRIPLE_CACHE[key]
  result = run_gpt_prompt_event_triple(act_desp, persona)[0]
  with _EVENT_TRIPLE_LOCK:
    _EVENT_TRIPLE_CACHE[key] = result
    while len(_EVENT_TRIPLE_CACHE) > _EVENT_TRIPLE_CACHE_MAX:
      _EVENT_TRIPLE_CACHE.popitem(last=False)
  return result


# PERF (Oct 21 2026): the object-state pair (act_obj_desc ->
# act_obj_event_triple) was 2 serial cloud calls per activity change whose
# answers were almost always "being used" / "in use" and
# "(<object>, is, <state>)". Templated by default; GA_OBJ_DESC_LLM=1
# restores the LLM path.
def _obj_desc_llm():
  return os.environ.get("GA_OBJ_DESC_LLM", "0") == "1"


def _template_obj_desc(act_desp):
  a = (act_desp or "").lower()
  if re.search(r"\b(sleep|sleeping|asleep|nap|napping)\b", a):
    return "being slept in"
  if re.search(r"\b(clean|cleaning|tidy|tidying|wash|washing)\b", a):
    return "being cleaned"
  if re.search(r"\b(fix|fixing|repair|repairing)\b", a):
    return "being fixed"
  return "being used"


def generate_act_obj_desc(act_game_object, act_desp, persona): 
  if debug: print ("GNS FUNCTION: <generate_act_obj_desc>")
  if not _obj_desc_llm():
    return _template_obj_desc(act_desp)
  result = run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona)
  if result is None:
    return f"{act_game_object} is idle"
  return result[0]


def generate_act_obj_event_triple(act_game_object, act_obj_desc, persona): 
  if debug: print ("GNS FUNCTION: <generate_act_obj_event_triple>")
  if not _obj_desc_llm():
    return (act_game_object, "is", act_obj_desc)
  return run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona)[0]


def generate_convo(maze, init_persona, target_persona): 
  curr_loc = maze.access_tile(init_persona.scratch.curr_tile)

  # convo = run_gpt_prompt_create_conversation(init_persona, target_persona, curr_loc)[0]
  # convo = agent_chat_v1(maze, init_persona, target_persona)
  convo = agent_chat_v2(maze, init_persona, target_persona)
  all_utt = ""

  for row in convo: 
    speaker = row[0]
    utt = row[1]
    all_utt += f"{speaker}: {utt}\n"

  convo_length = math.ceil(int(len(all_utt)/8) / 30)
  # A conversation takes its participants' time: at least one sim-minute
  # per reply (GA_CHAT_SECONDS_PER_REPLY). This sets chatting_end_time —
  # the pair stays in "chat with X" for that long; the world clock is
  # untouched (Oct 22 2026; replies used to push the GLOBAL clock).
  import chat_time as _chat_time
  convo_length = max(convo_length, _chat_time.conversation_minutes(len(convo)))

  if debug: print ("GNS FUNCTION: <generate_convo>")
  return convo, convo_length


def generate_convo_summary(persona, convo): 
  convo_summary = run_gpt_prompt_summarize_conversation(persona, convo)[0]
  return convo_summary


def generate_decide_to_talk(init_persona, target_persona, retrieved): 
  # RL T2 (Oct 2026): offline-learned TalkPolicy from ga_traces — serves
  # ONLY on high-support state patterns (conservative offline RL); thin
  # support falls through to laya. Zero-cost dict lookup, fail-open.
  try:
    import rl_modules as _rm
    _tp = _rm.talk_policy()
    if _tp is not None:
      _lc0 = init_persona.a_mem.get_last_chat(target_persona.name)
      _d = _tp.decide({
          "who_activity": init_persona.scratch.act_description,
          "other_activity": target_persona.scratch.act_description,
          "needs": getattr(init_persona.scratch, 'needs', None),
          "last_chat": _lc0.description if _lc0 else "no recent conversation",
      })
      if _d is not None:
        if debug:
          print(f"[RLTalk] decide_to_talk -> {_d['choice']} "
                f"(conf {_d['confidence']:.2f}, support {_d['support']})")
        return _d["choice"] == "yes"
  except Exception as _rl_err:
    print(f"[RLTalk] policy error ({_rl_err.__class__.__name__}) — cascading")
  # Cloud Jev teacher path (Sep 2026 distillation harvest): typed choice from
  # the cloud System One model; harvested answers feed the v3 training set.
  if os.environ.get("GA_CLOUDJEV", "0") == "1":
    try:
      from persona.prompt_template.cloudjev_scoring import cloudjev_decide_talk
      _lc = init_persona.a_mem.get_last_chat(target_persona.name)
      _last_about = _lc.description if _lc else None
      ans, conf = cloudjev_decide_talk(
          init_persona.scratch.name, init_persona.scratch.act_description,
          target_persona.name, target_persona.scratch.act_description,
          last_chat_about=_last_about,
          needs=getattr(init_persona.scratch, 'needs', None))
      if ans is not None:
        if debug: print(f"[CLOUDJEV] decide_to_talk -> {ans} (conf {conf:.2f})")
        return ans == "yes"
    except Exception as _cloud_err:
      print(f"[CLOUDJEV] talk fast path error ({_cloud_err.__class__.__name__}: "
            f"{_cloud_err}) — cascading")
  # Laya v3 reflex fast path (rizzo handover, Sep 30 2026): typed yes/no
  # choice from the cloud-distilled engine on :8018. Falls open to rizzo
  # (if ever re-enabled) then the legacy LLM path.
  try:
    from persona.prompt_template.laya_scoring import laya_decide_talk
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_about = last_chat.description if last_chat else None
    ans, conf = laya_decide_talk(
        init_persona.scratch.name, init_persona.scratch.act_description,
        target_persona.name, target_persona.scratch.act_description,
        last_chat_about=last_about,
        needs=getattr(init_persona.scratch, 'needs', None))
    if ans is not None:
      if debug: print (f"[LAYA] decide_to_talk -> {ans} (conf {conf:.2f})")
      return ans == "yes"
  except Exception as _laya_err:
    print(f"[LAYA] talk fast path error ({_laya_err.__class__.__name__}: "
          f"{_laya_err}) — cascading")

  x =run_gpt_prompt_decide_to_talk(init_persona, target_persona, retrieved)[0]
  if debug: print ("GNS FUNCTION: <generate_decide_to_talk>")

  if x == "yes": 
    return True
  else: 
    return False


def generate_decide_to_react(init_persona, target_persona, retrieved): 
  if debug: print ("GNS FUNCTION: <generate_decide_to_react>")

  # Cloud Jev teacher path (Sep 2026 distillation harvest).
  if os.environ.get("GA_CLOUDJEV", "0") == "1":
    try:
      from persona.prompt_template.cloudjev_scoring import cloudjev_decide_react
      _act = ((init_persona.scratch.act_address or "") + " "
              + (init_persona.scratch.act_description or "")).lower()
      _excl = any(w in _act for w in ("bathroom", "shower", "toilet"))
      ans, conf = cloudjev_decide_react(
          init_persona.scratch.name, init_persona.scratch.act_description,
          target_persona.name, target_persona.scratch.act_description,
          venue=init_persona.scratch.act_address,
          exclusive=_excl)
      if ans is not None:
        if debug: print(f"[CLOUDJEV] decide_to_react -> {ans} (conf {conf:.2f})")
        return ans
    except Exception as _cloud_err:
      print(f"[CLOUDJEV] react fast path error ({_cloud_err.__class__.__name__}: "
            f"{_cloud_err}) — cascading")

  # Rizzo reflex fast path (Sep 2026 redesign): typed wait/continue choice.
  # NOTE: this replaces an unvalidated legacy path — the live decide_to_react
  # prompts offered only 2 options but asked for "three options", so the LLM
  # answered the nonexistent Option 3 on 81% of calls (runtime collapsed it
  # to False). Rizzo always answers a valid option or punts (None, None).
  try:
    from persona.prompt_template.laya_scoring import laya_decide_react
    # Layer 0: capacity-1 facilities (bathroom/shower) are exclusive — say so
    # in the prompt so the typed decision has the facts it needs. Check both
    # the address and the activity text (sim addresses can mismatch).
    _act = ((init_persona.scratch.act_address or "") + " "
            + (init_persona.scratch.act_description or "")).lower()
    _excl = any(w in _act for w in ("bathroom", "shower", "toilet"))
    ans, conf = laya_decide_react(
        init_persona.scratch.name, init_persona.scratch.act_description,
        target_persona.name, target_persona.scratch.act_description,
        venue=init_persona.scratch.act_address,
        exclusive=_excl)
    if ans is not None:
      if debug: print (f"[LAYA] decide_to_react -> {ans} (conf {conf:.2f})")
      return ans
  except Exception as _laya_err:
    print(f"[LAYA] react fast path error ({_laya_err.__class__.__name__}: "
          f"{_laya_err}) — falling back to legacy generation")

  return run_gpt_prompt_decide_to_react(init_persona, target_persona, retrieved)[0]


def generate_new_decomp_schedule(persona, inserted_act, inserted_act_dur,  start_hour, end_hour, convo_note=None): 
  # Step 1: Setting up the core variables for the function. 
  # <p> is the persona whose schedule we are editing right now. 
  p = persona
  # <today_min_pass> indicates the number of minutes that have passed today. 
  today_min_pass = (int(p.scratch.curr_time.hour) * 60 
                    + int(p.scratch.curr_time.minute) + 1)
  
  # Step 2: We need to create <main_act_dur> and <truncated_act_dur>. 
  # These are basically a sub-component of <f_daily_schedule> of the persona,
  # but focusing on the current decomposition. 
  # Here is an example for <main_act_dur>: 
  # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
  # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
  # ['wakes up and completes her morning routine (uses the restroom)', 5]
  # ['wakes up and completes her morning routine (washes her ...)', 10]
  # ['wakes up and completes her morning routine (makes her bed)', 5]
  # ['wakes up and completes her morning routine (eats breakfast)', 15]
  # ['wakes up and completes her morning routine (gets dressed)', 10]
  # ['wakes up and completes her morning routine (leaves her ...)', 5]
  # ['wakes up and completes her morning routine (starts her ...)', 5]
  # ['preparing for her day (waking up at 6am)', 5]
  # ['preparing for her day (making her bed)', 5]
  # ['preparing for her day (taking a shower)', 15]
  # ['preparing for her day (getting dressed)', 5]
  # ['preparing for her day (eating breakfast)', 10]
  # ['preparing for her day (brushing her teeth)', 5]
  # ['preparing for her day (making coffee)', 5]
  # ['preparing for her day (checking her email)', 5]
  # ['preparing for her day (starting to work on her painting)', 5]
  # 
  # And <truncated_act_dur> concerns only until where an event happens. 
  # ['wakes up and completes her morning routine (wakes up at 6am)', 5]
  # ['wakes up and completes her morning routine (wakes up at 6am)', 2]
  main_act_dur = []
  truncated_act_dur = []
  dur_sum = 0 # duration sum
  count = 0 # enumerate count
  truncated_fin = False 

  print ("DEBUG::: ", persona.scratch.name)
  for act, dur in p.scratch.f_daily_schedule: 
    if (dur_sum >= start_hour * 60) and (dur_sum < end_hour * 60): 
      main_act_dur += [[act, dur]]
      if dur_sum <= today_min_pass:
        truncated_act_dur += [[act, dur]]
      elif dur_sum > today_min_pass and not truncated_fin: 
        # We need to insert that last act, duration list like this one: 
        # e.g., ['wakes up and completes her morning routine (wakes up...)', 2]
        truncated_act_dur += [[p.scratch.f_daily_schedule[count][0], 
                               dur_sum - today_min_pass]] 
        truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
        # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass + 1) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
        print ("DEBUG::: ", truncated_act_dur)

        # truncated_act_dur[-1][-1] -= (dur_sum - today_min_pass) ######## DEC 7 DEBUG;.. is the +1 the right thing to do??? 
        truncated_fin = True
    dur_sum += dur
    count += 1

  persona_name = persona.name 
  main_act_dur = main_act_dur

  # FIX (poison-wrap bug): "on the way to" rewrite used to slice
  # multi-paren text badly (split("(")[-1] grabs the LAST group, and
  # [:-1] leaves a dangling paren) — and the insert-wrap below
  # accumulated wrapped text across replans, doubling schedule entries
  # exponentially (seen live: 864M-char descriptions, 885MB movement
  # files). Scrub everything to the plain task head before any wrapping.
  def _scrub(t):
    return t.split("(")[0].strip()

  # FIX (Oct 6 2026, live3d_n59 wedge): a degenerate schedule (e.g. a
  # 370-min 'sleeping (on the way)' block from a malformed chat replan)
  # can place start_hour/end_hour BEYOND the f_daily_schedule span, leaving
  # truncated_act_dur empty — then [-1] raised IndexError and parked the
  # backend at the console, freezing the whole town. Guard: fall back to
  # the last schedule entry, or a synthetic sleep block if the day's
  # schedule is entirely empty; never crash.
  if len(truncated_act_dur) == 0:
    if len(p.scratch.f_daily_schedule) > 0:
      truncated_act_dur += [[p.scratch.f_daily_schedule[-1][0],
                             max(1, (end_hour * 60 - today_min_pass))]]
    else:
      truncated_act_dur += [["sleep", max(1, end_hour * 60 - today_min_pass)]]
    print("[Plan] truncated_act_dur was EMPTY — degenerate schedule "
          f"(start_hour={start_hour}, end_hour={end_hour}, "
          f"today_min_pass={today_min_pass}); inserted fallback block")

  prev_head = _scrub(truncated_act_dur[-1][0])
  prev_sub = (truncated_act_dur[-1][0].split("(")[-1].rsplit(")", 1)[0].strip()
              if "(" in truncated_act_dur[-1][0] else "")
  if prev_sub:
    x = f"{prev_head} (on the way to {prev_sub})"
  else:
    x = f"{prev_head} (on the way)"
  truncated_act_dur[-1][0] = x 

  if "(" in truncated_act_dur[-1][0]: 
    # FIX: wrap a CLEANED insert — never re-wrap wrapped text. Cap length
    # so a chatty LLM can't smuggle a novel in here either.
    if chat_replan_fix_enabled():
      # Keep what the chat was ABOUT (times, places, promises). The old
      # scrub cut at the first "(" and capped at 80 chars, so
      # "...Valentine's Day party at Hobbs Cafe from 5:00 to 7:00..." reached
      # the replan as "...discussing Isabella's Valenti". Nested parens are
      # flattened (one paren group max downstream), 110 chars keeps the
      # wrapped entry under _clean_action_text's 200-char cap.
      cleaned_insert = " ".join(
          str(inserted_act).replace("(", ",").replace(")", "").split())[:110]
    else:
      cleaned_insert = _scrub(inserted_act)[:80]
    inserted_act = prev_head[:80] + f" ({cleaned_insert})"

  # To do inserted_act_dur+1 below is an important decision but I'm not sure
  # if I understand the full extent of its implications. Might want to 
  # revisit. 
  truncated_act_dur += [[inserted_act, inserted_act_dur]]
  start_time_hour = (datetime.datetime(2022, 10, 31, 0, 0) 
                   + datetime.timedelta(hours=start_hour))
  end_time_hour = (datetime.datetime(2022, 10, 31, 0, 0) 
                   + datetime.timedelta(hours=end_hour))

  if debug: print ("GNS FUNCTION: <generate_new_decomp_schedule>")
  return run_gpt_prompt_new_decomp_schedule(persona, 
                                            main_act_dur, 
                                            truncated_act_dur, 
                                            start_time_hour,
                                            end_time_hour,
                                            inserted_act,
                                            inserted_act_dur,
                                            convo_note=convo_note)[0]


##############################################################################
# CHAPTER 3: Plan
##############################################################################

def revise_identity(persona): 
  p_name = persona.scratch.name

  focal_points = [f"{p_name}'s plan for {persona.scratch.get_str_curr_date_str()}.",
                  f"Important recent events for {p_name}'s life."]
  retrieved = new_retrieve(persona, focal_points)

  statements = "[Statements]\n"
  for key, val in retrieved.items():
    for i in val: 
      statements += f"{i.created.strftime('%A %B %d -- %H:%M %p')}: {i.embedding_key}\n"

  # print (";adjhfno;asdjao;idfjo;af", p_name)
  plan_prompt = statements + "\n"
  plan_prompt += f"Given the statements above, is there anything that {p_name} should remember as they plan for"
  plan_prompt += f" *{persona.scratch.curr_time.strftime('%A %B %d')}*? "
  plan_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)\n\n"
  plan_prompt += f"Write the response from {p_name}'s perspective."
  plan_note = ChatGPT_single_request(plan_prompt)
  # print (plan_note)

  thought_prompt = statements + "\n"
  thought_prompt += f"Given the statements above, how might we summarize {p_name}'s feelings about their days up to now?\n\n"
  thought_prompt += f"Write the response from {p_name}'s perspective."
  thought_note = ChatGPT_single_request(thought_prompt)
  # print (thought_note)

  currently_prompt = f"{p_name}'s status from {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n"
  currently_prompt += f"{persona.scratch.currently}\n\n"
  currently_prompt += f"{p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}:\n" 
  currently_prompt += (plan_note + thought_note).replace('\n', '') + "\n\n"
  currently_prompt += f"It is now {persona.scratch.curr_time.strftime('%A %B %d')}. Given the above, write {p_name}'s status for {persona.scratch.curr_time.strftime('%A %B %d')} that reflects {p_name}'s thoughts at the end of {(persona.scratch.curr_time - datetime.timedelta(days=1)).strftime('%A %B %d')}. Write this in third-person talking about {p_name}."
  currently_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n"
  currently_prompt += "Follow this format below:\nStatus: <new status>"
  # print ("DEBUG ;adjhfno;asdjao;asdfsidfjo;af", p_name)
  # print (currently_prompt)
  new_currently = ChatGPT_single_request(currently_prompt)
  # print (new_currently)
  # print (new_currently[10:])

  persona.scratch.currently = new_currently

  daily_req_prompt = persona.scratch.get_str_iss() + "\n"
  daily_req_prompt += f"Today is {persona.scratch.curr_time.strftime('%A %B %d')}. Here is {persona.scratch.name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
  daily_req_prompt += f"Follow this format (the list should have 4~6 items but no more):\n"
  daily_req_prompt += f"1. wake up and complete the morning routine at <time>, 2. ..."

  new_daily_req = ChatGPT_single_request(daily_req_prompt)
  new_daily_req = new_daily_req.replace('\n', ' ')
  print ("WE ARE HERE!!!", new_daily_req)
  persona.scratch.daily_plan_req = new_daily_req


def _long_term_planning(persona, new_day, maze=None):
  """
  Formulates the persona's daily long-term plan if it is the start of a new
  day. This basically has two components: first, we create the wake-up hour,
  and second, we create the hourly schedule based on it.
  INPUT
    new_day: Indicates whether the current time signals a "First day",
             "New day", or False (for neither). This is important because we
             create the personas' long term planning on the new day.
    maze: Optional Maze instance to access resource_manager (Phase 3.2)
  """
  # We start by creating the wake up hour for the persona.
  wake_up_hour = generate_wake_up_hour(persona)

  # Get resource_manager from maze if available (Phase 3.2)
  resource_manager = None
  if maze is not None and hasattr(maze, 'resource_manager'):
    resource_manager = maze.resource_manager

  # When it is a new day, we start by creating the daily_req of the persona.
  # Note that the daily_req is a list of strings that describe the persona's
  # day in broad strokes.
  if new_day == "First day":
    # Bootstrapping the daily plan for the start of then generation:
    # if this is the start of generation (so there is no previous day's
    # daily requirement, or if we are on a new day, we want to create a new
    # set of daily requirements.
    persona.scratch.daily_req = generate_first_daily_plan(persona,
                                                          wake_up_hour,
                                                          resource_manager)
  elif new_day == "New day":
    revise_identity(persona)

    # FIX (Sep 2026): stock GA left this as a TODO — `daily_req = daily_req`
    # — so agents replayed DAY ONE's plan forever, and anyone whose first
    # plan was the canned fail_safe lived the canned day every day (20/26
    # agents: 'read a book 8-12, nap 1-4' -> filler-hours -> phones).
    # revise_identity just refreshed `currently` + today's plan requirement,
    # so a fresh plan carries the agent's evolving storylines. If today's
    # generation still collapses to the fail_safe, keep yesterday's REAL plan.
    _FS_SIG = "read a book from 8:00 am to 12:00 pm"
    _old_req = list(persona.scratch.daily_req or [])
    try:
      _new_req = generate_first_daily_plan(persona, wake_up_hour,
                                           resource_manager)
    except Exception as e:
      print(f"[plan] {persona.scratch.name}: new-day plan failed ({e!r}); keeping previous")
      _new_req = None
    _new_is_fs = bool(_new_req) and any(_FS_SIG in x for x in _new_req)
    _old_is_fs = any(_FS_SIG in x for x in _old_req)
    if _new_req and (not _new_is_fs or _old_is_fs or not _old_req):
      persona.scratch.daily_req = _new_req
    print(f"[plan] {persona.scratch.name}: new-day daily_req "
          f"{'REGENERATED' if persona.scratch.daily_req is _new_req else 'KEPT'}"
          f" ({len(persona.scratch.daily_req)} items, fail_safe={_new_is_fs})")

    # AGENCY L2 (Voyager routines + self-curriculum, env-gated): seed one
    # routine-refresh goal and/or one never-visited arena into daily_req.
    try:
      import routines as _rt
      # roster for curriculum person-arena filtering: read the LIVE sim's
      # meta (curr_sim_code.json → meta.json persona_names), cached on persona
      try:
        import json as _json
        _backend = os.path.dirname(os.path.abspath(__file__))
        _code = _json.load(open(os.path.join(
            _backend, '../../environment/frontend_server/temp_storage',
            'curr_sim_code.json')))['sim_code']
        _p = (os.path.join(_backend, '../../environment/frontend_server/storage/',
                           _code, 'reverie', 'meta.json'))
        persona._roster_names = list(getattr(persona, '_roster_names', [])
                                     or _json.load(open(_p))['persona_names'])
      except Exception:
        persona._roster_names = []
      _rg = _rt.routine_goal(persona, maze)
      if _rg and _rg not in persona.scratch.daily_req:
        persona.scratch.daily_req.append(_rg)
        print(f"[Routines] {persona.name} daily_req += '{_rg}'")
      _cg = _rt.curriculum_goal(persona, maze)
      if _cg:
        _cgoal = f"explore the {_cg['arena_name']} in {_cg.get('sector', 'the ville')} for the first time"
        if _cgoal not in persona.scratch.daily_req:
          persona.scratch.daily_req.append(_cgoal)
          print(f"[Curriculum] {persona.name} daily_req += '{_cgoal}'")
    except Exception as _rt_err:
      print(f"[Routines] error ({type(_rt_err).__name__}: {_rt_err}) — day proceeds without")

  # Based on the daily_req, we create an hourly schedule for the persona, 
  # which is a list of todo items with a time duration (in minutes) that 
  # add up to 24 hours.
  persona.scratch.f_daily_schedule = generate_hourly_schedule(persona, 
                                                              wake_up_hour)
  persona.scratch.f_daily_schedule_hourly_org = (persona.scratch
                                                   .f_daily_schedule[:])
  _commit_post_block(persona)


  # Added March 4 -- adding plan to the memory.
  thought = f"This is {persona.scratch.name}'s plan for {persona.scratch.curr_time.strftime('%A %B %d')}:"
  for i in persona.scratch.daily_req: 
    thought += f" {i},"
  thought = thought[:-1] + "."
  created = persona.scratch.curr_time
  expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
  s, p, o = (persona.scratch.name, "plan", persona.scratch.curr_time.strftime('%A %B %d'))
  keywords = set(["plan"])
  thought_poignancy = 5
  thought_embedding_pair = (thought, get_embedding(thought))
  persona.a_mem.add_thought(created, expiration, s, p, o, 
                            thought, keywords, thought_poignancy, 
                            thought_embedding_pair, None)

  # print("Sleeping for 20 seconds...")
  # time.sleep(10)
  # print("Done sleeping!")




POST_BLOCK_START_MIN = 10 * 60      # a town-service post is worked 10:00-12:00
POST_BLOCK_MIN = 120


def _insert_block(sched, start, dur, text):
  """Overwrite minutes [start, start+dur) of a [[task, minutes], ...] day schedule with
  `text`, keeping the surrounding items (trimmed) and the 1440-minute total."""
  end = start + dur
  out, t = [], 0
  for task, mins in sched:
    a, b = t, t + mins
    t = b
    if b <= start or a >= end:
      out.append([task, mins])
      continue
    if a < start:
      out.append([task, start - a])
    if b > end:
      out.append(["\0", 0])            # marker: tail kept after the block
      out.append([task, b - end])
  res, placed = [], False
  for task, mins in out:
    if task == "\0":
      continue
    res.append([task, mins])
  # place the block at the right offset
  acc, final, done = 0, [], False
  for task, mins in res:
    if not done and acc >= start:
      final.append([text, dur]); done = True
    final.append([task, mins]); acc += mins
  if not done:
    final.append([text, dur])
  return final


def _commit_post_block(persona):
  """GA_JOBBOARD: a town-service post is a commitment: the work block goes INTO the
  schedule (an agent that merely 'could' go never did), and the plan notes it."""
  try:
    import economy as _eco
    e = _ECON_REF.get("e")
    if not (_eco.jobboard_enabled() and e is not None):
      return
    tid = e.public_post(persona.name)
    if not tid:
      return
    text = _eco.CIVIC_TASKS[tid]["name"]
    sched = persona.scratch.f_daily_schedule
    if sum(m for _, m in sched) != 1440 or any(t.startswith(text) for t, _ in sched):
      return
    new = _insert_block(sched, POST_BLOCK_START_MIN, POST_BLOCK_MIN, text)
    if sum(m for _, m in new) == 1440:
      persona.scratch.f_daily_schedule = new
      persona.scratch.f_daily_schedule_hourly_org = new[:]
      print(f"[JobBoard] {persona.name}: scheduled '{text}' 10:00-12:00")
  except Exception as err:
    print(f"[JobBoard] post block failed for {persona.name}: {err!r}")

FREE_CHOICE_SAFETY = 10     # a need below this is handled by rule, not choice


def free_choice_enabled():
  return os.environ.get("GA_FREE_CHOICE", "0") == "1"


def _free_choice(persona):
  """One decision for the next block. Returns {"choice", "activity",
  "minutes", "why", "pulls"} or None (legacy behaviour: pulls first, then the
  schedule). Records compliance on scratch for the agent card / metrics."""
  s = persona.scratch
  now = s.curr_time
  sched = s.f_daily_schedule
  if not sched or now is None:
    return None
  idx = s.get_f_daily_schedule_index()
  if idx >= len(sched):
    return None
  item = sched[idx][0]
  nxt = sched[idx + 1][0] if idx + 1 < len(sched) else ""
  needs = getattr(s, "needs", None) or {}
  if any(isinstance(v, (int, float)) and v < FREE_CHOICE_SAFETY
         for k, v in needs.items() if k != "status"):
    return None
  asleep = re.search(r"\b(sleep|sleeping|asleep|bed)\b", item.lower()) is not None
  pulls = list(s.resource_goals)
  if asleep or (getattr(s, "chatting_with", None)):
    return None
  from persona.prompt_template.run_gpt_prompt import run_gpt_prompt_choose_activity
  out = run_gpt_prompt_choose_activity(persona, item, nxt, pulls)
  if out is None:
    return None
  out["pulls"] = pulls
  stats = getattr(s, "choice_stats", None)
  if not isinstance(stats, dict):
    stats = {"decisions": 0, "complied": 0, "pull": 0, "own": 0, "last": ""}
  stats["decisions"] += 1
  if out["choice"] == 0:
    stats["complied"] += 1
    shown = "follows the plan"
  elif out["choice"] == "own":
    stats["own"] += 1
    shown = f"own: {out['activity']}"
  else:
    stats["pull"] += 1
    shown = f"takes '{pulls[out['choice'] - 1]}'"
  stats["last"] = f"{shown} — {out['why']}"[:160]
  s.choice_stats = stats
  print(f"[Choice] {persona.name} {shown} (plan: {item[:50]}) why: {out['why'][:80]}")
  return out


def _determine_action(persona, maze):
  """
  Creates the next action sequence for the persona.
  The main goal of this function is to run "add_new_action" on the persona's
  scratch space, which sets up all the action related variables for the next
  action.
  As a part of this, the persona may need to decompose its hourly schedule as
  needed.
  INPUT
    persona: Current <Persona> instance whose action we are determining.
    maze: Current <Maze> instance.
  """
  # AGENCY L1 (PIANO-style drives, env-gated): when GA_DRIVES=1, continuous
  # drive salience arbitrates FIRST and injects its winning goal; the legacy
  # needs-urgency thresholds below are subsumed (they stay for fallback).
  _drives_active = False
  try:
    from agentic_drives import arbitrate as _drv_arbitrate, DRIVE_GOALS as _DRIVE_GOALS
    # AGENCY L3 (Schmidhuber curiosity, env-gated): learning progress and
    # boredom translate into a curiosity salience BOOST fed into arbitration
    # — curiosity can then WIN when life has become predictable.
    _cboost = 0.0
    try:
      import curiosity as _cu
      # RL T1 (Oct 2026): the VenueQ layer provides a TD-error-grounded
      # learning signal (default ON since Oct 7 2026; GA_QCURIOSITY=0 opts
      # out); the incumbent frequency-
      # heuristic remains the fallback (gate OFF or Q-data empty).
      _qp = _rb = None
      try:
        import rl_modules as _rm
        _qsig = _rm.q_learning_progress(persona)
        if _qsig is not None:
          _qp, _rb = _qsig
      except Exception:
        pass
      if _qp is not None:
        _prog, _bored = _qp, _rb
      else:
        _prog, _bored = _cu.learning_progress(persona)
      if _bored >= 1.0:
        _cboost = 0.5
      elif _prog >= 0.30:
        _cboost = 0.3
      if _cboost > 0:
        print(f"[Curiosity] {persona.name} lp={_prog:.2f} boredom={_bored:.0f} "
              f"boost=+{_cboost}" + (" [Q]" if _qp is not None else ""))
    except Exception:
      pass
    _winner = _drv_arbitrate(persona, maze, curiosity_boost=_cboost)
    if _winner is not None:
      _drives_active = True
      _goal = _DRIVE_GOALS[_winner]
      # Evening social life (Oct 22 2026): the pub made 3 sales in 3.5 days.
      # After 6 PM the social drive heads there — "having a drink" is the
      # resource mapping that turns the visit into a real, paid round.
      _hr = getattr(persona.scratch.curr_time, "hour", 12)
      if _winner == "social" and 18 <= _hr < 23:
        _goal = "having a drink with friends at The Rose and Crown Pub"
      _now = persona.scratch.curr_time
      _served = getattr(persona.scratch, "_drive_served", {})
      _last = _served.get(_winner)
      _cooldown_h = 3.0
      _cd_ok = (_last is None or _now is None or
                (_now - _last).total_seconds() / 3600.0 >= _cooldown_h)
      # Sleep gate (Oct 21 2026, fresh25_1 cold start): arbitration had no
      # sense of time — with an empty event history at 00:00 curiosity won
      # for nearly everyone and agents went "exploring" at midnight, which
      # is where n68's broken sleep (11am naps, dinner before noon) came
      # from. While the scheduled activity is sleep, only somatic drives
      # (hunger/energy) may inject a goal.
      _asleep = False
      try:
        _sched = persona.scratch.f_daily_schedule
        _cur = _sched[persona.scratch.get_f_daily_schedule_index()][0] if _sched else ""
        _asleep = re.search(r"\b(sleep|sleeping|asleep|bed)\b", _cur.lower()) is not None
      except Exception:
        pass
      if _asleep and _winner not in ("hunger", "energy"):
        _cd_ok = False
      if (_cd_ok and _goal not in persona.scratch.resource_goals
          and hasattr(persona.scratch, "resource_goals")):
        persona.scratch.resource_goals.insert(0, _goal)
        _served[_winner] = _now
        persona.scratch._drive_served = _served
        print(f"[Drives] {persona.name} drive={_winner} -> '{_goal}'")
  except Exception as _drv_err:
    print(f"[Drives] arbitration error ({type(_drv_err).__name__}: {_drv_err}) — falling back to legacy")

  # Check resource goals first — these take priority over normal replanning.
  # FIX (needs-driven urgency, Oct 21 2026): this block was gated behind
  # `if not _drives_active:` — dead in production, because critical bladder
  # FOLDS INTO hunger-drive salience (agentic_drives), so a drive won nearly
  # every arbitration and relief never ran: agents stayed critical-bladder,
  # the talk hard-gate (critical non-social need -> always no) muted every
  # encounter -> 0 conversations for entire sim-days. Now UNCONDITIONAL and
  # placed after the drives insert, so its insert(0) wins the resource_goals
  # pop. Goals are progressive-tense (goal strings render as "X is <goal>")
  # and home-anchored. CORRECTION (Oct 21, same day): the original diagnosis
  # claimed the maze has NO bathroom venues — wrong file (maze_meta_info.json
  # carries no arena names). special_blocks/arena_blocks.csv shows 18
  # bathroom arenas, every one fully tiled (15-22 tiles), all rendered to the
  # location picker via the s_mem tree — so 'using the bathroom at home'
  # resolves to the persona's REAL home bathroom arena, and
  # satisfy_needs_for_action's keyword relief ('bathroom' -> +70) fires on
  # the act DESCRIPTION. The venue layer was never the problem; the
  # dead-code gate was the whole defect.
  if hasattr(persona.scratch, "needs") and persona.scratch.needs:
    critical = getattr(persona.scratch, "needs_critical", 20)
    urgent_map = {
      "bladder": "using the bathroom at home",
      "hygiene": "taking a shower at home",
      "hunger": "eating a meal at the Hobbs Cafe",
      "energy": "resting at home",
      "hydration": "getting a drink of water",
    }
    for need, goal in urgent_map.items():
      val = persona.scratch.needs.get(need, 100)
      if val < critical:
        # Don't stack duplicates of the same urgent goal
        if goal not in persona.scratch.resource_goals:
          persona.scratch.resource_goals.insert(0, goal)
          print(f"[UrgentNeed] {persona.name} critical {need} ({val:.0f}) -> '{goal}'")
        break  # one urgent need at a time — highest urgency order above

  # JOBS (GA_JOBS): an employee works their shift. Hiring used to change
  # nothing about behaviour — `scratch.job` was never read by any planner.
  try:
    import economy as _eco
    _job = getattr(persona.scratch, "job", None)
    _jn = persona.scratch.curr_time
    if (_eco.jobs_enabled() and isinstance(_job, dict) and _jn is not None
        and _eco.SHIFT_START <= _jn.hour < _eco.SHIFT_END
        and getattr(persona.scratch, "_work_goal_day", None) != _jn.date()
        and hasattr(persona.scratch, "resource_goals")):
      _vn = _eco.VENUES[_job["venue"]]["name"]
      persona.scratch._work_goal_day = _jn.date()
      # two 30-min goals: the wage needs 60 on-site minutes
      persona.scratch.resource_goals[:0] = [f"working a shift at {_vn}"] * 2
      print(f"[Jobs] {persona.name} heading to work at {_vn}")
  except Exception:
    pass

  # COMMUNITY WORK (GA_COMMUNITY): agents without a job and short of money
  # take a paid civic task at a public place, once a day.
  try:
    import economy as _eco
    _cn = persona.scratch.curr_time
    if (_eco.community_enabled() and _cn is not None
        and not isinstance(getattr(persona.scratch, "job", None), dict)
        and ((getattr(persona.scratch, "wallet", 0) or 0) < _eco.CIVIC_ELIGIBLE_WALLET
             or (_ECON_REF.get("e") is not None and _ECON_REF["e"].public_post(persona.name)))
        and _eco.CIVIC_START <= _cn.hour < _eco.CIVIC_END - 1
        and getattr(persona.scratch, "_civic_goal_day", None) != _cn.date()
        and hasattr(persona.scratch, "resource_goals")
        and _ECON_REF.get("e") is not None):
      _task = _ECON_REF["e"].pick_civic_task(persona.name, _cn)
      persona.scratch._civic_goal_day = _cn.date()
      if _task:
        persona.scratch.resource_goals[:0] = [_task] * 2
        print(f"[Community] {persona.name} signed up for {_task}")
  except Exception:
    pass

  # PROACTIVE GROCERY SHOPPING (Sep 2026 economy revival): inject a
  # shopping goal before hunger goes critical, with a cooldown, so money
  # actually circulates (22/25 agents had never spent a dollar).
  try:
    _needs = getattr(persona.scratch, "needs", None)
    _hunger = _needs.get("hunger", 100) if isinstance(_needs, dict) else 100
    _wallet = getattr(persona.scratch, "wallet", 0)
    # FIX (grocery loop, Oct 21 2026 fresh25_1): the cooldown keyed only on
    # last_grocery_shop, which a FAILED purchase (store sold out) never set —
    # and under economy v2 not even a successful one did. With hunger < 45
    # the same 30-min shopping action was re-issued forever (Abigail, Wolfgang
    # 2000+ steps at the store, never eating). Cool down on the ATTEMPT too,
    # shop only while the store is open, and jitter each persona's threshold
    # so the whole town doesn't go shopping in the same step.
    _now = persona.scratch.curr_time
    _hours = None
    for _stamp in (getattr(persona.scratch, "last_grocery_attempt", None),
                   getattr(persona.scratch, "last_grocery_shop", None)):
      if _stamp is not None:
        try:
          _h = (_now - _stamp).total_seconds() / 3600.0
          _hours = _h if _hours is None else min(_hours, _h)
        except Exception:
          pass
    _cooldown_ok = (_hours is None) or (_hours >= 4)
    _open = _now is not None and 7 <= _now.hour < 20
    import zlib as _zlib
    _threshold = 40 + _zlib.crc32(persona.name.encode()) % 11   # 40..50
    _shop_goal = "buying groceries at the Harvey Oak Supply Store"
    # Second grocery (Oct 22 2026): The Willows Market sells the same
    # staples but made $0 in fresh25_1 — nothing ever sent anyone there.
    # Use it whenever Harvey Oak is short (it sold out on day 1), and as the
    # regular store for a crc32 bucket of residents (3 of the 25: Hailey
    # Johnson, Tom Moreno, Sam Moore).
    try:
      _ws = maze.resource_manager.world_state
      _harvey = next((v for k, v in _ws.items()
                      if "harvey oak supply store" in k.lower()
                      and "shelves" in k.lower()), {})
      _short = (_harvey.get("eggs", 0) < 5 or _harvey.get("bread", 0) < 3
                or _harvey.get("milk", 0) < 2)
    except Exception:
      _short = False
    if _short or _zlib_pref(persona.name, 3) == 0:
      _shop_goal = ("buying groceries at the market "
                    "(The Willows Market and Pharmacy)")
    _already = any("grocer" in g.lower()
                   for g in getattr(persona.scratch, "resource_goals", []))
    # Outcome memory (GA_LESSONS): an empty home pantry is a reason to shop even
    # while hunger is still high — otherwise the agent keeps trying to cook.
    _stock_out = False
    try:
      from persona.cognitive_modules import lessons as _lessons
      _stock_out = _lessons.enabled() and bool(
        _lessons.live(persona.scratch, _now, kind="home_stock"))
    except Exception:
      pass
    if ((_hunger < _threshold or _stock_out) and _cooldown_ok and _open
        and _wallet >= 30
        and not _already and hasattr(persona.scratch, "resource_goals")):
      persona.scratch.resource_goals.insert(0, _shop_goal)
      persona.scratch.last_grocery_attempt = _now
      print(f"[Economy] {persona.name} heading to the store "
            f"(hunger {_hunger:.0f} < {_threshold}, wallet ${_wallet:.0f})")
  except Exception:
    pass  # economy must never crash the plan loop

  # LUXURY PURCHASE (Oct 22 2026): the "status" need never forces a replan;
  # it only becomes a plan when the agent has actually SAVED enough — the
  # most expensive luxury good it can buy while keeping a reserve, at most
  # once a day, during shop hours. Until then it lingers (needs summary in
  # the daily-plan prompt + recurring thoughts, reverie._status_thoughts).
  try:
    from economy import LUXURY_GOODS, LUXURY_RESERVE
    _needs = getattr(persona.scratch, "needs", None) or {}
    _status = _needs.get("status", 100)
    _now = persona.scratch.curr_time
    _wallet = getattr(persona.scratch, "wallet", 0) or 0
    _last = getattr(persona.scratch, "last_luxury_attempt", None)
    _cool = _last is None or (_now - _last).total_seconds() >= 24 * 3600
    _goals = getattr(persona.scratch, "resource_goals", None)
    if (_status < 35 and _now is not None and 10 <= _now.hour < 19 and _cool
        and _goals is not None
        and not any("luxury" in g.lower() for g in _goals)):
      _affordable = [(p, i) for i, p in LUXURY_GOODS.items()
                     if _wallet - p >= LUXURY_RESERVE]
      if _affordable:
        _price, _item = max(_affordable)
        _goals.insert(0, f"buying a {_item} at The Willows Market and "
                         f"Pharmacy (a luxury purchase)")
        persona.scratch.last_luxury_attempt = _now
        print(f"[Status] {persona.name} saved enough (${_wallet:.0f}) — "
              f"going to buy a {_item} (${_price:.0f}), status {_status:.0f}")
  except Exception:
    pass  # status must never crash the plan loop

  # FREE CHOICE (GA_FREE_CHOICE): the schedule is a strong suggestion, not an
  # order. The rule-injected goals above become options the agent weighs
  # against its plan; it may also do something of its own. Critical needs
  # (< FREE_CHOICE_SAFETY) bypass the choice so nobody starves by choice.
  _own_act = _own_min = None
  if free_choice_enabled() and hasattr(persona.scratch, "resource_goals"):
    try:
      _decision = _free_choice(persona)
    except Exception as _fc_err:
      print(f"[Choice] error ({type(_fc_err).__name__}: {_fc_err}) — legacy")
      _decision = None
    if _decision is not None:
      _pulls = _decision["pulls"]
      _c = _decision["choice"]
      persona.scratch.resource_goals = []      # offered once; unchosen ones lapse
      if _c == "own":
        _own_act, _own_min = _decision["activity"], _decision["minutes"]
        persona.scratch.resource_goals = [_own_act]
      elif _c != 0:
        persona.scratch.resource_goals = [_pulls[_c - 1]]

  if hasattr(persona.scratch, "resource_goals") and persona.scratch.resource_goals:
    resource_goal = persona.scratch.resource_goals.pop(0)
    print(f"[ResourceGoal] {persona.name} pursuing: {resource_goal}")

    # Use the resource goal as the action description
    act_desp = resource_goal
    act_dura = _own_min or 30  # 30 minutes for resource-driven actions

    # Determine the target location based on the goal
    act_world, act_sector, act_arena, act_game_object = \
        resolve_action_location(act_desp, persona, maze)

    if act_game_object and act_game_object != "<random>":
      new_address = f"{act_world}:{act_sector}:{act_arena}:{act_game_object}"
    else:
      new_address = f"{act_world}:{act_sector}:{act_arena}"

    act_pron = generate_action_pronunciatio(act_desp, persona)
    act_event = generate_action_event_triple(act_desp, persona)
    act_obj_desp = generate_act_obj_desc(act_game_object, act_desp, persona)
    act_obj_pron = generate_action_pronunciatio(act_obj_desp, persona)
    act_obj_event = generate_act_obj_event_triple(act_game_object, act_obj_desp, persona)

    # Add the resource-goal action to persona's queue
    persona.scratch.add_new_action(new_address,
                                   int(act_dura),
                                   act_desp,
                                   act_pron,
                                   act_event,
                                   None,
                                   None,
                                   None,
                                   None,
                                   act_obj_desp,
                                   act_obj_pron,
                                   act_obj_event)
    return  # Exit early - resource goal handled

  def determine_decomp(act_desp, act_dura):
    """
    Given an action description and its duration, we determine whether we need
    to decompose it. If the action is about the agent sleeping, we generally
    do not want to decompose it, so that's what we catch here. 

    INPUT: 
      act_desp: the description of the action (e.g., "sleeping")
      act_dura: the duration of the action in minutes. 
    OUTPUT: 
      a boolean. True if we need to decompose, False otherwise. 
    """
    if "sleep" not in act_desp and "bed" not in act_desp: 
      return True
    elif "sleeping" in act_desp or "asleep" in act_desp or "in bed" in act_desp:
      return False
    elif "sleep" in act_desp or "bed" in act_desp: 
      if act_dura > 60: 
        return False
    return True

  # The goal of this function is to get us the action associated with 
  # <curr_index>. As a part of this, we may need to decompose some large 
  # chunk actions. 
  # Importantly, we try to decompose at least two hours worth of schedule at
  # any given point. 
  curr_index = persona.scratch.get_f_daily_schedule_index()
  curr_index_60 = persona.scratch.get_f_daily_schedule_index(advance=60)

  # * Decompose * 
  # During the first hour of the day, we need to decompose two hours 
  # sequence. We do that here. 
  if curr_index == 0:
    # This portion is invoked if it is the first hour of the day. 
    act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index]
    if act_dura >= 60: 
      # We decompose if the next action is longer than an hour, and fits the
      # criteria described in determine_decomp.
      if determine_decomp(act_desp, act_dura): 
        persona.scratch.f_daily_schedule[curr_index:curr_index+1] = (
                            generate_task_decomp(persona, act_desp, act_dura))
    if curr_index_60 + 1 < len(persona.scratch.f_daily_schedule):
      act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60+1]
      if act_dura >= 60: 
        if determine_decomp(act_desp, act_dura): 
          persona.scratch.f_daily_schedule[curr_index_60+1:curr_index_60+2] = (
                            generate_task_decomp(persona, act_desp, act_dura))

  if curr_index_60 < len(persona.scratch.f_daily_schedule):
    # If it is not the first hour of the day, this is always invoked (it is
    # also invoked during the first hour of the day -- to double up so we can
    # decompose two hours in one go). Of course, we need to have something to
    # decompose as well, so we check for that too. 
    if persona.scratch.curr_time.hour < 23:
      # And we don't want to decompose after 11 pm. 
      act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index_60]
      if act_dura >= 60: 
        if determine_decomp(act_desp, act_dura): 
          persona.scratch.f_daily_schedule[curr_index_60:curr_index_60+1] = (
                              generate_task_decomp(persona, act_desp, act_dura))
  # * End of Decompose * 

  # Generate an <Action> instance from the action description and duration. By
  # this point, we assume that all the relevant actions are decomposed and 
  # ready in f_daily_schedule. 
  print ("DEBUG LJSDLFSKJF")
  for i in persona.scratch.f_daily_schedule: print (i)
  print (curr_index)
  print (len(persona.scratch.f_daily_schedule))
  print (persona.scratch.name)
  print ("------")

  # 1440
  x_emergency = 0
  for i in persona.scratch.f_daily_schedule: 
    x_emergency += i[1]
  # print ("x_emergency", x_emergency)

  if 1440 - x_emergency > 0: 
    print ("x_emergency__AAA", x_emergency)
  persona.scratch.f_daily_schedule += [["sleeping", 1440 - x_emergency]]
  



  act_desp, act_dura = persona.scratch.f_daily_schedule[curr_index] 



  # Finding the target location of the action and creating action-related
  # variables.
  act_world, act_sector, act_arena, act_game_object = \
      resolve_action_location(act_desp, persona, maze)
  # Don't include <random> placeholder in the address — use arena address only
  if act_game_object and act_game_object != "<random>":
    new_address = f"{act_world}:{act_sector}:{act_arena}:{act_game_object}"
  else:
    new_address = f"{act_world}:{act_sector}:{act_arena}"
  act_pron = generate_action_pronunciatio(act_desp, persona)
  act_event = generate_action_event_triple(act_desp, persona)
  # Persona's actions also influence the object states. We set those up here. 
  act_obj_desp = generate_act_obj_desc(act_game_object, act_desp, persona)
  act_obj_pron = generate_action_pronunciatio(act_obj_desp, persona)
  act_obj_event = generate_act_obj_event_triple(act_game_object, 
                                                act_obj_desp, persona)

  # Adding the action to persona's queue. 
  persona.scratch.add_new_action(new_address, 
                                 int(act_dura), 
                                 act_desp, 
                                 act_pron, 
                                 act_event,
                                 None,
                                 None,
                                 None,
                                 None,
                                 act_obj_desp, 
                                 act_obj_pron, 
                                 act_obj_event)


def _choose_retrieved(persona, retrieved): 
  """
  Retrieved elements have multiple core "curr_events". We need to choose one
  event to which we are going to react to. We pick that event here. 
  INPUT
    persona: Current <Persona> instance whose action we are determining. 
    retrieved: A dictionary of <ConceptNode> that were retrieved from the 
               the persona's associative memory. This dictionary takes the
               following form: 
               dictionary[event.description] = 
                 {["curr_event"] = <ConceptNode>, 
                  ["events"] = [<ConceptNode>, ...], 
                  ["thoughts"] = [<ConceptNode>, ...] }
  """
  # Once we are done with the reflection, we might want to build a more  
  # complex structure here.
  
  # We do not want to take self events... for now 
  copy_retrieved = retrieved.copy()
  for event_desc, rel_ctx in copy_retrieved.items(): 
    curr_event = rel_ctx["curr_event"]
    if curr_event.subject == persona.name: 
      del retrieved[event_desc]

  # Always choose persona first.
  priority = []
  for event_desc, rel_ctx in retrieved.items(): 
    curr_event = rel_ctx["curr_event"]
    if (":" not in curr_event.subject 
        and curr_event.subject != persona.name): 
      priority += [rel_ctx]
  if priority: 
    return random.choice(priority)

  # Skip idle. 
  for event_desc, rel_ctx in retrieved.items(): 
    curr_event = rel_ctx["curr_event"]
    if "is idle" not in event_desc: 
      priority += [rel_ctx]
  if priority: 
    return random.choice(priority)
  return None


def _should_react(persona, retrieved, personas): 
  """
  Determines what form of reaction the persona should exihibit given the 
  retrieved values. 
  INPUT
    persona: Current <Persona> instance whose action we are determining. 
    retrieved: A dictionary of <ConceptNode> that were retrieved from the 
               the persona's associative memory. This dictionary takes the
               following form: 
               dictionary[event.description] = 
                 {["curr_event"] = <ConceptNode>, 
                  ["events"] = [<ConceptNode>, ...], 
                  ["thoughts"] = [<ConceptNode>, ...] }
    personas: A dictionary that contains all persona names as keys, and the 
              <Persona> instance as values. 
  """
  def lets_talk(init_persona, target_persona, retrieved):
    if (not target_persona.scratch.act_address 
        or not target_persona.scratch.act_description
        or not init_persona.scratch.act_address
        or not init_persona.scratch.act_description): 
      return False

    if ("sleeping" in target_persona.scratch.act_description 
        or "sleeping" in init_persona.scratch.act_description): 
      return False

    if init_persona.scratch.curr_time.hour == 23: 
      return False

    if "<waiting>" in target_persona.scratch.act_address: 
      return False

    if (target_persona.scratch.chatting_with 
      or init_persona.scratch.chatting_with): 
      return False

    if (target_persona.name in init_persona.scratch.chatting_with_buffer): 
      if init_persona.scratch.chatting_with_buffer[target_persona.name] > 0: 
        return False

    # Social attention budget (quota-era suppressor, 2026-10-05 reversal):
    # banal-event ignore gate now defaults OFF — laya grades talk decisions
    # at ~100ms, so the LLM-budget rationale is gone. Set
    # GA_POIG_IGNORE_FLOOR back to 3 (and restart) to restore old behavior.
    _poig_floor = int(os.environ.get("GA_POIG_IGNORE_FLOOR", "0"))
    _curr = retrieved.get("curr_event") if isinstance(retrieved, dict) else None
    if _curr is not None and getattr(_curr, "poignancy", 10) < _poig_floor:
      return False

    # Cap concurrent social-decision LLM calls (see _SOCIAL_THINK_SEM note).
    with _SOCIAL_THINK_SEM:
      if generate_decide_to_talk(init_persona, target_persona, retrieved): 
        return True

    return False

  def lets_react(init_persona, target_persona, retrieved): 
    if (not target_persona.scratch.act_address 
        or not target_persona.scratch.act_description
        or not init_persona.scratch.act_address
        or not init_persona.scratch.act_description): 
      return False

    if ("sleeping" in target_persona.scratch.act_description 
        or "sleeping" in init_persona.scratch.act_description): 
      return False

    # return False
    if init_persona.scratch.curr_time.hour == 23: 
      return False

    if "waiting" in target_persona.scratch.act_description: 
      return False
    if init_persona.scratch.planned_path == []:
      return False

    if (init_persona.scratch.act_address 
        != target_persona.scratch.act_address): 
      return False

    react_mode = generate_decide_to_react(init_persona, 
                                          target_persona, retrieved)

    if react_mode == "1": 
      wait_until = ((target_persona.scratch.act_start_time 
        + datetime.timedelta(minutes=target_persona.scratch.act_duration - 1))
        .strftime("%B %d, %Y, %H:%M:%S"))
      return f"wait: {wait_until}"
    elif react_mode == "2":
      return False
      return "do other things"
    else:
      return False #"keep" 

  # If the persona is chatting right now, default to no reaction 
  if persona.scratch.chatting_with: 
    return False
  if "<waiting>" in persona.scratch.act_address: 
    return False

  # Recall that retrieved takes the following form: 
  # dictionary {["curr_event"] = <ConceptNode>, 
  #             ["events"] = [<ConceptNode>, ...], 
  #             ["thoughts"] = [<ConceptNode>, ...]}
  curr_event = retrieved["curr_event"]

  if curr_event.subject in personas:
    # this is a persona event.
    if lets_talk(persona, personas[curr_event.subject], retrieved):
      return f"chat with {curr_event.subject}"
    react_mode = lets_react(persona, personas[curr_event.subject],
                            retrieved)
    return react_mode
  return False


def _create_react(persona, inserted_act, inserted_act_dur,
                  act_address, act_event, chatting_with, chat, chatting_with_buffer,
                  chatting_end_time, 
                  act_pronunciatio, act_obj_description, act_obj_pronunciatio, 
                  act_obj_event, act_start_time=None): 
  p = persona 

  min_sum = 0
  for i in range (p.scratch.get_f_daily_schedule_hourly_org_index()): 
    min_sum += p.scratch.f_daily_schedule_hourly_org[i][1]
  start_hour = int (min_sum/60)

  if (p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] >= 120):
    end_hour = start_hour + p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1]/60

  elif (p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] + 
      p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()+1][1]): 
    end_hour = start_hour + ((p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()][1] + 
              p.scratch.f_daily_schedule_hourly_org[p.scratch.get_f_daily_schedule_hourly_org_index()+1][1])/60)

  else: 
    end_hour = start_hour + 2
  end_hour = int(end_hour)

  dur_sum = 0
  count = 0 
  start_index = None
  end_index = None
  for act, dur in p.scratch.f_daily_schedule: 
    if dur_sum >= start_hour * 60 and start_index == None:
      start_index = count
    if dur_sum >= end_hour * 60 and end_index == None: 
      end_index = count
    dur_sum += dur
    count += 1

  _fix = chat_replan_fix_enabled()
  ret = generate_new_decomp_schedule(
      p, inserted_act, inserted_act_dur, start_hour, end_hour,
      convo_note=(inserted_act if (_fix and chatting_with) else None))
  if _fix:
    # FIX A (duration sanity), corrected: cap absurd subtasks, but keep the
    # spliced window the SAME LENGTH as the slice it replaces. The old
    # clamp(parent_dur=inserted_act_dur) rescaled the whole multi-hour
    # window down to the chat's few minutes, so every chat deleted ~40 min
    # of schedule and slid the rest of the day earlier (agents ate dinner
    # at 10 AM and slept by 3 PM; they missed the 5 PM Valentine's party).
    _slice = p.scratch.f_daily_schedule[start_index:end_index]
    _span = sum(int(d) for _, d in _slice)
    _llm_total = sum(int(d) for _, d in ret)
    ret = fit_schedule_to_span(clamp_subtask_durations(ret), _span)
  else:
    _slice = p.scratch.f_daily_schedule[start_index:end_index]
    _span = sum(int(d) for _, d in _slice)
    _llm_total = sum(int(d) for _, d in ret)
    ret = clamp_subtask_durations(ret, parent_dur=inserted_act_dur)
  # A/B telemetry: how much of the replanned window survived the splice.
  print(f"[ChatReplan] {p.name} mode={'fix' if _fix else 'legacy'} "
        f"chat={inserted_act_dur}m window={start_hour}-{end_hour}h "
        f"slice_before={_span}m llm_total={_llm_total}m "
        f"slice_after={sum(int(d) for _, d in ret)}m "
        f"sched_total={sum(int(d) for _, d in p.scratch.f_daily_schedule) - _span + sum(int(d) for _, d in ret)}m",
        flush=True)
  p.scratch.f_daily_schedule[start_index:end_index] = ret
  p.scratch.add_new_action(act_address,
                           inserted_act_dur,
                           inserted_act,
                           act_pronunciatio,
                           act_event,
                           chatting_with,
                           chat,
                           chatting_with_buffer,
                           chatting_end_time,
                           act_obj_description,
                           act_obj_pronunciatio,
                           act_obj_event,
                           act_start_time)


def _chat_react(maze, persona, focused_event, reaction_mode, personas):
  # There are two personas -- the persona who is initiating the conversation
  # and the persona who is the target. We get the persona instances here. 
  init_persona = persona
  target_persona = personas[reaction_mode[9:].strip()]
  curr_personas = [init_persona, target_persona]

  # Actually creating the conversation here. 
  convo, duration_min = generate_convo(maze, init_persona, target_persona)
  convo_summary = generate_convo_summary(init_persona, convo)
  _journal_conversation(init_persona, target_persona, convo, convo_summary,
                        duration_min)
  inserted_act = convo_summary
  inserted_act_dur = duration_min

  act_start_time = target_persona.scratch.act_start_time

  curr_time = target_persona.scratch.curr_time
  if curr_time.second != 0: 
    temp_curr_time = curr_time + datetime.timedelta(seconds=60 - curr_time.second)
    chatting_end_time = temp_curr_time + datetime.timedelta(minutes=inserted_act_dur)
  else: 
    chatting_end_time = curr_time + datetime.timedelta(minutes=inserted_act_dur)

  for role, p in [("init", init_persona), ("target", target_persona)]: 
    if role == "init": 
      act_address = f"<persona> {target_persona.name}"
      act_event = (p.name, "chat with", target_persona.name)
      chatting_with = target_persona.name
      # Re-chat cooldown (quota-era suppressor, 2026-10-05 reversal): 800
      # muted a pair for most of a sim-day after ONE chat — brutal for
      # small-registry towns. Env-overridable, default now 120 (1 sim-hour).
      chatting_with_buffer = {}
      chatting_with_buffer[target_persona.name] = int(
          os.environ.get("GA_RECHAT_COOLDOWN_STEPS", "120"))
    elif role == "target":
      act_address = f"<persona> {init_persona.name}"
      act_event = (p.name, "chat with", init_persona.name)
      chatting_with = init_persona.name
      chatting_with_buffer = {}
      chatting_with_buffer[init_persona.name] = int(
          os.environ.get("GA_RECHAT_COOLDOWN_STEPS", "120"))

    act_pronunciatio = "💬" 
    act_obj_description = None
    act_obj_pronunciatio = None
    act_obj_event = (None, None, None)

    _create_react(p, inserted_act, inserted_act_dur,
      act_address, act_event, chatting_with, convo, chatting_with_buffer, chatting_end_time,
      act_pronunciatio, act_obj_description, act_obj_pronunciatio,
      act_obj_event, act_start_time)

  # Phase 3.3b: Resource sharing memory injection
  # If conversation mentions resources, inject a memory for the listener
  _inject_resource_sharing_memories(maze, init_persona, target_persona, convo)


def _inject_resource_sharing_memories(maze, init_persona, target_persona, convo):
  """
  After a conversation ends, if resource-related keywords appear in the conversation,
  inject a memory node for the *other* agent so they're aware of the shared info.

  Phase 3.3b: Social resource awareness in conversations
  """
  if not convo:
    return

  resource_keywords = ["out of", "empty", "no coffee", "no eggs", "grocery", "fridge",
                       "running low", "store", "restocked", "supplies", "ran out",
                       "need to buy", "shopping", "market", "cafe", "breakfast"]

  # Build full conversation text
  all_utt = ""
  for row in convo:
    speaker = row[0]
    utt = row[1]
    all_utt += f"{speaker}: {utt}\n"

  all_utt_lower = all_utt.lower()

  # Check if any resource keywords appear
  if not any(kw in all_utt_lower for kw in resource_keywords):
    return

  # For each utterance, check if it contains resource keywords and inject memory for listener
  for row in convo:
    speaker_name = row[0]
    utterance = row[1]
    utt_lower = utterance.lower()

    # Check for resource-related keywords in this utterance
    found_keywords = [kw for kw in resource_keywords if kw in utt_lower]
    if not found_keywords:
      continue

    # Determine the listener (the other persona)
    if speaker_name == init_persona.name:
      listener = target_persona
    elif speaker_name == target_persona.name:
      listener = init_persona
    else:
      continue

    # Create a memory for the listener about what the speaker mentioned
    try:
      curr_time = listener.scratch.curr_time
      expiration = curr_time + datetime.timedelta(days=7)

      # Build a concise memory description
      keyword_str = ", ".join(found_keywords[:2])  # Limit to 2 keywords
      memory_desc = f"{speaker_name} mentioned something about {keyword_str} during conversation"

      s = speaker_name
      p = "mentioned"
      o = keyword_str

      keywords = set([speaker_name.lower(), "conversation", "resource"] + found_keywords[:2])
      poignancy = 4  # Moderate importance

      # Simple embedding key
      embedding_key = f"resource_chat_{speaker_name}_{listener.name}_{curr_time.strftime('%Y%m%d%H%M%S')}"
      embedding_pair = (embedding_key, [0.0] * 1536)

      listener.a_mem.add_event(
        curr_time, expiration,
        s, p, o,
        memory_desc, keywords, poignancy,
        embedding_pair, []
      )

      print(f"[ResourceSharing] {listener.name} remembers: {memory_desc}")

    except Exception as e:
      print(f"[ResourceSharing] Error injecting memory: {e}")
      continue


def _wait_react(persona, reaction_mode): 
  p = persona

  inserted_act = f'waiting to start {p.scratch.act_description.split("(")[-1][:-1]}'
  end_time = datetime.datetime.strptime(reaction_mode[6:].strip(), "%B %d, %Y, %H:%M:%S")
  inserted_act_dur = (end_time.minute + end_time.hour * 60) - (p.scratch.curr_time.minute + p.scratch.curr_time.hour * 60) + 1

  act_address = f"<waiting> {p.scratch.curr_tile[0]} {p.scratch.curr_tile[1]}"
  act_event = (p.name, "waiting to start", p.scratch.act_description.split("(")[-1][:-1])
  chatting_with = None
  chat = None
  chatting_with_buffer = None
  chatting_end_time = None

  act_pronunciatio = "⌛" 
  act_obj_description = None
  act_obj_pronunciatio = None
  act_obj_event = (None, None, None)

  _create_react(p, inserted_act, inserted_act_dur,
    act_address, act_event, chatting_with, chat, chatting_with_buffer, chatting_end_time,
    act_pronunciatio, act_obj_description, act_obj_pronunciatio, act_obj_event)


def plan(persona, maze, personas, new_day, retrieved): 
  """
  Main cognitive function of the chain. It takes the retrieved memory and 
  perception, as well as the maze and the first day state to conduct both 
  the long term and short term planning for the persona. 

  INPUT: 
    maze: Current <Maze> instance of the world. 
    personas: A dictionary that contains all persona names as keys, and the 
              Persona instance as values. 
    new_day: This can take one of the three values. 
      1) <Boolean> False -- It is not a "new day" cycle (if it is, we would
         need to call the long term planning sequence for the persona). 
      2) <String> "First day" -- It is literally the start of a simulation,
         so not only is it a new day, but also it is the first day. 
      2) <String> "New day" -- It is a new day. 
    retrieved: dictionary of dictionary. The first layer specifies an event,
               while the latter layer specifies the "curr_event", "events", 
               and "thoughts" that are relevant.
  OUTPUT 
    The target action address of the persona (persona.scratch.act_address).
  """ 
  # PART 1: Generate the hourly schedule.
  if new_day:
    _long_term_planning(persona, new_day, maze)

  # PART 2: If the current action has expired, we want to create a new plan.
  if persona.scratch.act_check_finished(): 
    _determine_action(persona, maze)

  # PART 3: If you perceived an event that needs to be responded to (saw 
  # another persona), and retrieved relevant information. 
  # Step 1: Retrieved may have multiple events represented in it. The first 
  #         job here is to determine which of the events we want to focus 
  #         on for the persona. 
  #         <focused_event> takes the form of a dictionary like this: 
  #         dictionary {["curr_event"] = <ConceptNode>, 
  #                     ["events"] = [<ConceptNode>, ...], 
  #                     ["thoughts"] = [<ConceptNode>, ...]}
  focused_event = False
  if retrieved.keys(): 
    focused_event = _choose_retrieved(persona, retrieved)
  
  # Step 2: Once we choose an event, we need to determine whether the
  # persona will take any actions for the perceived event. There are
  # three possible modes of reaction returned by _should_react.
  #   a) "chat with {target_persona.name}"
  #   b) "react"
  #   c) False
  # FIX (vector 1): reaction decisions read other personas' live state and
  # conversations mutate BOTH personas — serialize on CONVO_LOCK under the
  # pooled-cognition step loop.
  # PERF (Oct 21 2026, n68): the lock used to span the WHOLE conversation —
  # ~24 serial cloud calls (5-10 min) during which every other persona with
  # a focused event blocked, freezing the step barrier. Now the lock covers
  # only the decision + claiming both speakers (CHAT_BUSY); generation runs
  # outside it, so independent pairs/reactions proceed in parallel.
  from concurrency_utils import CONVO_LOCK, CHAT_BUSY
  if focused_event:
    chat_pair = None
    with CONVO_LOCK:
      if persona.name not in CHAT_BUSY:
        reaction_mode = _should_react(persona, focused_event, personas)
        if reaction_mode:
          if reaction_mode[:9] == "chat with":
            target_name = reaction_mode[9:].strip()
            if target_name not in CHAT_BUSY:
              chat_pair = (persona.name, target_name)
              CHAT_BUSY.update(chat_pair)
          elif reaction_mode[:4] == "wait":
            _wait_react(persona, reaction_mode)
    if chat_pair:
      from concurrency_utils import async_chat_enabled, PENDING_CHATS
      if async_chat_enabled():
        # Queued, not generated: reverie starts it after this step's pool
        # and freezes both speakers until it is written back (the pair
        # stays claimed in CHAT_BUSY until then).
        with CONVO_LOCK:
          PENDING_CHATS.append(
              (_chat_react, (maze, persona, focused_event, reaction_mode, personas),
               chat_pair))
      else:
        try:
          _chat_react(maze, persona, focused_event, reaction_mode, personas)
        finally:
          with CONVO_LOCK:
            CHAT_BUSY.difference_update(chat_pair)

  # Step 3: Chat-related state clean up. 
  # If the persona is not chatting with anyone, we clean up any of the 
  # chat-related states here. 
  if persona.scratch.act_event[1] != "chat with":
    persona.scratch.stash_unreflected_chat()
    persona.scratch.chatting_with = None
    persona.scratch.chat = None
    persona.scratch.chatting_end_time = None
  # We want to make sure that the persona does not keep conversing with each
  # other in an infinite loop. So, chatting_with_buffer maintains a form of 
  # buffer that makes the persona wait from talking to the same target 
  # immediately after chatting once. We keep track of the buffer value here. 
  curr_persona_chat_buffer = persona.scratch.chatting_with_buffer
  for persona_name, buffer_count in curr_persona_chat_buffer.items():
    if persona_name != persona.scratch.chatting_with: 
      persona.scratch.chatting_with_buffer[persona_name] -= 1

  return persona.scratch.act_address













































 
