"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: plan.py
Description: This defines the "Plan" module for generative agents. 
"""
import datetime
import math
import random 
import re
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
_SOCIAL_THINK_SEM = threading.Semaphore(2)
sys.path.append('../../')

from global_methods import *
from persona.prompt_template.run_gpt_prompt import *
from persona.cognitive_modules.retrieve import *
from persona.cognitive_modules.converse import *

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

  try:
    x = run_gpt_prompt_pronunciatio(act_desp, persona)[0]
  except:
    x = "🙂"

  if not x:
    return "🙂"
  return x


def _parse_event_triple(action_description, persona_name):
  """Parse (subject, predicate, object) from action description via regex.
  Returns tuple or None if parsing fails."""
  import re
  desc = action_description.strip()
  if "(" in desc:
    desc = desc.split("(")[-1].split(")")[0]

  # Common verb stems to avoid bad stemming
  VERB_MAP = {
    "cooking": "cook", "eating": "eat", "sleeping": "sleeping",
    "reading": "read", "writing": "write", "working": "work",
    "studying": "study", "walking": "walk", "running": "run",
    "shopping": "shop", "cleaning": "clean", "painting": "paint",
    "drawing": "draw", "listening": "listen", "watching": "watch",
    "drinking": "drink", "brewing": "brew", "teaching": "teach",
    "playing": "play", "sitting": "sit", "standing": "stand",
    "waiting": "wait", "talking": "talk", "chatting": "chat",
    "showering": "shower", "bathing": "bathe", "exercising": "exercise",
    "gardening": "garden", "organizing": "organize", "typing": "type",
    "browsing": "browse", "preparing": "prepare", "making": "make",
  }

  # Pattern: "<Name> is <verb>ing <rest>"
  m = re.match(r'^(.+?)\s+is\s+(\w+ing)\b\s*(.*?)\.?$', desc, re.IGNORECASE)
  if m:
    subject = m.group(1).strip()
    verb_ing = m.group(2).strip().lower()
    obj = m.group(3).strip()
    verb = VERB_MAP.get(verb_ing, verb_ing)
    if not obj:
      return (subject, "is", verb_ing)
    return (subject, verb, obj)

  # Pattern: "<Name> is <adjective/state>"
  m = re.match(r'^(.+?)\s+is\s+(\w+)\.?$', desc, re.IGNORECASE)
  if m:
    return (m.group(1).strip(), "is", m.group(2).strip())

  return None


def generate_action_event_triple(act_desp, persona):
  if debug: print ("GNS FUNCTION: <generate_action_event_triple>")
  parsed = _parse_event_triple(act_desp, persona.name)
  if parsed:
    return parsed
  return run_gpt_prompt_event_triple(act_desp, persona)[0]


def generate_act_obj_desc(act_game_object, act_desp, persona): 
  if debug: print ("GNS FUNCTION: <generate_act_obj_desc>")
  result = run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona)
  if result is None:
    return f"{act_game_object} is idle"
  return result[0]


def generate_act_obj_event_triple(act_game_object, act_obj_desc, persona): 
  if debug: print ("GNS FUNCTION: <generate_act_obj_event_triple>")
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

  if debug: print ("GNS FUNCTION: <generate_convo>")
  return convo, convo_length


def generate_convo_summary(persona, convo): 
  convo_summary = run_gpt_prompt_summarize_conversation(persona, convo)[0]
  return convo_summary


def generate_decide_to_talk(init_persona, target_persona, retrieved): 
  # Rizzo reflex fast path (Sep 2026 redesign): typed yes/no choice,
  # ~130ms, zero generated tokens. Falls open to the legacy LLM path.
  try:
    from persona.prompt_template.rizzo_scoring import rizzo_decide_talk
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_about = last_chat.description if last_chat else None
    ans, conf = rizzo_decide_talk(
        init_persona.scratch.name, init_persona.scratch.act_description,
        target_persona.name, target_persona.scratch.act_description,
        last_chat_about=last_about,
        needs=getattr(init_persona.scratch, 'needs', None))
    if ans is not None:
      if debug: print (f"[RIZZO] decide_to_talk -> {ans} (conf {conf:.2f})")
      return ans == "yes"
  except Exception as _rizzo_err:
    print(f"[RIZZO] talk fast path error ({_rizzo_err.__class__.__name__}: "
          f"{_rizzo_err}) — falling back to legacy generation")

  x =run_gpt_prompt_decide_to_talk(init_persona, target_persona, retrieved)[0]
  if debug: print ("GNS FUNCTION: <generate_decide_to_talk>")

  if x == "yes": 
    return True
  else: 
    return False


def generate_decide_to_react(init_persona, target_persona, retrieved): 
  if debug: print ("GNS FUNCTION: <generate_decide_to_react>")

  # Rizzo reflex fast path (Sep 2026 redesign): typed wait/continue choice.
  # NOTE: this replaces an unvalidated legacy path — the live decide_to_react
  # prompts offered only 2 options but asked for "three options", so the LLM
  # answered the nonexistent Option 3 on 81% of calls (runtime collapsed it
  # to False). Rizzo always answers a valid option or punts (None, None).
  try:
    from persona.prompt_template.rizzo_scoring import rizzo_decide_react
    # Layer 0: capacity-1 facilities (bathroom/shower) are exclusive — say so
    # in the prompt so the typed decision has the facts it needs. Check both
    # the address and the activity text (sim addresses can mismatch).
    _act = ((init_persona.scratch.act_address or "") + " "
            + (init_persona.scratch.act_description or "")).lower()
    _excl = any(w in _act for w in ("bathroom", "shower", "toilet"))
    ans, conf = rizzo_decide_react(
        init_persona.scratch.name, init_persona.scratch.act_description,
        target_persona.name, target_persona.scratch.act_description,
        venue=init_persona.scratch.act_address,
        exclusive=_excl)
    if ans is not None:
      if debug: print (f"[RIZZO] decide_to_react -> {ans} (conf {conf:.2f})")
      return ans
  except Exception as _rizzo_err:
    print(f"[RIZZO] react fast path error ({_rizzo_err.__class__.__name__}: "
          f"{_rizzo_err}) — falling back to legacy generation")

  return run_gpt_prompt_decide_to_react(init_persona, target_persona, retrieved)[0]


def generate_new_decomp_schedule(persona, inserted_act, inserted_act_dur,  start_hour, end_hour): 
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
                                            inserted_act_dur)[0]


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

  # Based on the daily_req, we create an hourly schedule for the persona, 
  # which is a list of todo items with a time duration (in minutes) that 
  # add up to 24 hours.
  persona.scratch.f_daily_schedule = generate_hourly_schedule(persona, 
                                                              wake_up_hour)
  persona.scratch.f_daily_schedule_hourly_org = (persona.scratch
                                                   .f_daily_schedule[:])


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
  # Check resource goals first — these take priority over normal replanning
  # FIX (needs-driven urgency): needs_critical was a dead threshold — nothing
  # ever read it. When a need crosses the critical line, inject an urgent
  # resource goal so the persona actually ACTS on its body instead of
  # following the daily plan while desperate.
  if hasattr(persona.scratch, "needs") and persona.scratch.needs:
    critical = getattr(persona.scratch, "needs_critical", 20)
    urgent_map = {
      "bladder": "go to the nearest bathroom to relieve themselves",
      "hygiene": "take a shower in the bathroom",
      "hunger": "eat a meal or get food from Hobbs Cafe",
      "energy": "go home and take a rest",
      "hydration": "get a drink of water",
    }
    for need, goal in urgent_map.items():
      val = persona.scratch.needs.get(need, 100)
      if val < critical:
        # Don't stack duplicates of the same urgent goal
        if goal not in persona.scratch.resource_goals:
          persona.scratch.resource_goals.insert(0, goal)
        break  # one urgent need at a time — highest urgency order above

  # PROACTIVE GROCERY SHOPPING (Sep 2026 economy revival): inject a
  # shopping goal before hunger goes critical, with a cooldown, so money
  # actually circulates (22/25 agents had never spent a dollar).
  try:
    _needs = getattr(persona.scratch, "needs", None)
    _hunger = _needs.get("hunger", 100) if isinstance(_needs, dict) else 100
    _wallet = getattr(persona.scratch, "wallet", 0)
    _last_shop = getattr(persona.scratch, "last_grocery_shop", None)
    _hours = None
    if _last_shop is not None:
      try:
        _hours = ((persona.scratch.curr_time - _last_shop).total_seconds() / 3600.0)
      except Exception:
        _hours = None
    _cooldown_ok = (_last_shop is None) or (_hours is None) or (_hours >= 6)
    _shop_goal = "buy groceries at the Harvey Oak Supply Store"
    _already = _shop_goal in getattr(persona.scratch, "resource_goals", [])
    if (_hunger < 45 and _cooldown_ok and _wallet >= 30
        and not _already and hasattr(persona.scratch, "resource_goals")):
      persona.scratch.resource_goals.insert(0, _shop_goal)
      print(f"[Economy] {persona.name} heading to the store "
            f"(hunger {_hunger:.0f}, wallet ${_wallet:.0f})")
  except Exception:
    pass  # economy must never crash the plan loop

  if hasattr(persona.scratch, "resource_goals") and persona.scratch.resource_goals:
    resource_goal = persona.scratch.resource_goals.pop(0)
    print(f"[ResourceGoal] {persona.name} pursuing: {resource_goal}")

    # Use the resource goal as the action description
    act_desp = resource_goal
    act_dura = 30  # 30 minutes for resource-driven actions

    # Determine the target location based on the goal
    act_world = maze.access_tile(persona.scratch.curr_tile)["world"]
    act_sector = generate_action_sector(act_desp, persona, maze)
    act_arena = generate_action_arena(act_desp, persona, maze, act_world, act_sector)

    # Strip any LLM artifacts from arena name
    import re as _re
    act_arena = _re.sub(r'^Answer:\s*', '', act_arena, flags=_re.IGNORECASE).strip()
    act_arena = act_arena.lstrip("{[(\"'").rstrip("}])\"'").strip()

    act_address = f"{act_world}:{act_sector}:{act_arena}"
    act_game_object = generate_action_game_object(act_desp, act_address, persona, maze)

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
  act_world = maze.access_tile(persona.scratch.curr_tile)["world"]
  # act_sector = maze.access_tile(persona.scratch.curr_tile)["sector"]
  act_sector = generate_action_sector(act_desp, persona, maze)
  act_arena = generate_action_arena(act_desp, persona, maze, act_world, act_sector)
  # Strip any LLM artifacts from arena name
  import re as _re
  act_arena = _re.sub(r'^Answer:\s*', '', act_arena, flags=_re.IGNORECASE).strip()
  act_arena = act_arena.lstrip("{[(\"'").rstrip("}])\"'").strip()
  act_address = f"{act_world}:{act_sector}:{act_arena}"
  act_game_object = generate_action_game_object(act_desp, act_address,
                                                persona, maze)
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

    # Social attention budget: banal events (low poignancy) never reach the
    # LLM — the decide prompt embeds the same context the poig score graded.
    _curr = retrieved.get("curr_event") if isinstance(retrieved, dict) else None
    if _curr is not None and getattr(_curr, "poignancy", 10) < 3:
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

  ret = generate_new_decomp_schedule(p, inserted_act, inserted_act_dur,
                                       start_hour, end_hour)
  # FIX A (duration sanity): LLM decomp output may exceed the parent block.
  # Clamp the spliced block to the intended reaction duration so a chat or
  # nap reaction can't inflate into a multi-hour freeze.
  ret = clamp_subtask_durations(ret, parent_dur=inserted_act_dur)
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
      chatting_with_buffer = {}
      chatting_with_buffer[target_persona.name] = 800
    elif role == "target": 
      act_address = f"<persona> {init_persona.name}"
      act_event = (p.name, "chat with", init_persona.name)
      chatting_with = init_persona.name
      chatting_with_buffer = {}
      chatting_with_buffer[init_persona.name] = 800

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
  from concurrency_utils import CONVO_LOCK
  if focused_event:
    with CONVO_LOCK:
      reaction_mode = _should_react(persona, focused_event, personas)
      if reaction_mode:
        # If we do want to chat, then we generate conversation
        if reaction_mode[:9] == "chat with":
          _chat_react(maze, persona, focused_event, reaction_mode, personas)
        elif reaction_mode[:4] == "wait":
          _wait_react(persona, reaction_mode)

  # Step 3: Chat-related state clean up. 
  # If the persona is not chatting with anyone, we clean up any of the 
  # chat-related states here. 
  if persona.scratch.act_event[1] != "chat with":
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













































 
