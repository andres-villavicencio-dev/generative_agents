"""Schedule-as-a-tool: the agent edits its own agenda with real tool calls.

A tool loop runs when something worth recording may have happened (a chat or
a notice that states a clock time). Each turn the model emits ONE
JSON call; the system executes it, feeds the result back, and asks again:

    check_schedule  -> the agenda and the rest of today's plan
    add_event       -> record an event (going=true only if the agent decides to go)
    modify_event    -> change time/place/title, or decide to go after all
    cancel_event    -> drop or decline, with a reason
    done            -> finish

Whether to FOLLOW the schedule or do something else is decided later, at each
block boundary, by the free choice (plan._free_choice: Laya with LLM fallback),
which is shown the clock, the agenda and any event that is about to start.

Failure policy: at most MAX_TURNS turns; one repair prompt for malformed JSON;
on a second malformed reply or an unavailable model the loop stops and the
agenda is left as it was (the legacy chat-commitment pin still runs).
Every call is traced to <GA_TRACE_DIR>/agenda.jsonl.
"""
import json

import agenda as ag

MAX_TURNS = 4

TOOLS = {
  "check_schedule": {"args": {}, "doc": "show your agenda and the rest of today's plan"},
  "add_event": {"args": {"title": "str", "place": "str", "date": "YYYY-MM-DD, today or tomorrow; leave out if nobody said which day",
                         "start": "17:00 or 5pm", "end": "19:00 or 7pm (optional)",
                         "going": "true only if you decided to attend, else false"},
                "doc": "record an event you heard about or arranged"},
  "modify_event": {"args": {"id": "int", "title": "str?", "place": "str?", "date": "str?",
                            "start": "str?", "end": "str?",
                            "going": "true if you now decide to attend (optional)"},
                   "doc": "change an event on your agenda"},
  "cancel_event": {"args": {"id": "int", "reason": "str"}, "doc": "drop or decline an event"},
  "done": {"args": {}, "doc": "finish; use it at once if nothing needs recording"},
}

def has_temporal_cue(text):
  """Cheap gate: the text states a clock time ("at 5", "5:30 pm", "noon"...),
  the same test the commitment pin uses. No topic words.

  A day word alone does not open the loop. An event cannot be recorded without
  a start time, and on the n75/n76 call log the day-only loops (818 of 1,376:
  "how are you today", "you may", "tomorrow"...) produced 88 events with an
  invented time against 3 with a stated one. The agent is told to ask for the
  day and time instead (agenda.ASK_WHEN_UNSURE)."""
  from persona.prompt_template.run_gpt_prompt import has_time_cue
  return has_time_cue(text)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_call(text):
  """First JSON object in <text> as (tool, args), or (None, error)."""
  s = str(text or "")
  start = s.find("{")
  while start != -1:
    depth = 0
    for i in range(start, len(s)):
      depth += (s[i] == "{") - (s[i] == "}")
      if depth == 0:
        try:
          obj = json.loads(s[start:i + 1])
        except Exception:
          break
        if isinstance(obj, dict):
          name = obj.get("tool") or obj.get("name") or obj.get("function")
          args = obj.get("args") if "args" in obj else obj.get("arguments", {})
          if isinstance(args, str):
            try:
              args = json.loads(args)
            except Exception:
              args = {}
          if name in TOOLS and isinstance(args, dict):
            return name, args
          return None, f"unknown tool {name!r}; use one of {', '.join(TOOLS)}"
        break
    start = s.find("{", start + 1)
  return None, "reply with one JSON object like {\"tool\": \"done\", \"args\": {}}"


def _truthy(v):
  return v is True or str(v).strip().lower() in ("true", "yes", "1", "y")


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
def _plan_today(scratch, now, limit=6):
  sched = getattr(scratch, "f_daily_schedule", None) or []
  now_min = now.hour * 60 + now.minute
  t, rows = 0, []
  for task, d in sched:
    if t + d > now_min and len(rows) < limit:
      rows.append(f"{max(t, now_min) // 60:02d}:{max(t, now_min) % 60:02d} {str(task)[:60]}")
    t += d
  return rows


def execute(persona, name, args, maze=None):
  """Run one tool call. Returns (ok, result_text)."""
  s = persona.scratch
  now = s.curr_time
  try:
    if name == "check_schedule":
      view = ag.render(s, now, persona, maze, limit=8) or "(nothing on your agenda)"
      plan = "\n".join(_plan_today(s, now)) or "(nothing planned)"
      return True, f"Agenda:\n{view}\nRest of today's plan:\n{plan}"
    if name == "add_event":
      date = args.get("date")
      if not date:
        # nobody said which day: the next time that hour comes (an event that
        # already started is not "today" just because the chat did not say)
        c = ag.parse_clock(args.get("start"))
        date = "tomorrow" if c is not None and c <= now.hour * 60 + now.minute else "today"
      past = ag.start_in_past(date, args.get("start"), now)
      if past:
        return False, past
      ev = ag.add_event(s, args.get("title", ""), args.get("place", ""), date,
                        args.get("start"), args.get("end"), source="chat",
                        status="going" if _truthy(args.get("going")) else "known", now=now)
      if ev is None:
        return False, "Error: need a title, a date (YYYY-MM-DD, today, tomorrow) and a start time like 17:00."
      if ev["status"] == "going":
        ag.overlay(s, now)
      return True, "Recorded: " + ag.describe(ev, now)
    if name == "modify_event":
      old = ag.get_event(s, args.get("id"))
      if old is None:
        return False, f"Error: no event #{args.get('id')} on your agenda."
      given = lambda k: args.get(k) not in (None, "")
      if given("date") and ag._norm_date(args["date"], now) is None:
        return False, f"Error: cannot read the date {args['date']!r}; use YYYY-MM-DD, today or tomorrow."
      for k in ("start", "end"):
        if given(k) and ag.parse_clock(args[k]) is None:
          return False, f"Error: cannot read the {k} time {args[k]!r}; use a time like 17:00."
      if given("date") or given("start"):
        # same rule as add_event: an event cannot be moved to a time already gone
        start = ag.parse_clock(args["start"]) if given("start") else old["start"]
        past = ag.start_in_past(args["date"] if given("date") else old["date"],
                                f"{start // 60:02d}:{start % 60:02d}", now)
        if past:
          return False, past
      fields = {k: v for k, v in args.items() if k not in ("id", "going")}
      if given("going"):
        if _truthy(args["going"]):
          fields["status"] = "going"
        elif old["status"] == "going":
          fields["status"] = "known"          # no longer planning to go; cancel_event declines
      ev = ag.modify_event(s, args.get("id"), **fields)
      if ev["status"] == "going":
        ag.overlay(s, now)
      return True, "Updated: " + ag.describe(ev, now)
    if name == "cancel_event":
      ev = ag.cancel_event(s, args.get("id"), args.get("reason", ""))
      if ev is None:
        return False, f"Error: no event #{args.get('id')} on your agenda."
      return True, f"Declined #{ev['id']} {ev['title']}."
    if name == "done":
      return True, "ok"
  except Exception as e:                                   # never crash the sim
    return False, f"Error: {type(e).__name__}: {e}"
  return False, f"Error: unknown tool {name!r}."


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def _prompt(persona, trigger, history, maze):
  s = persona.scratch
  now = s.curr_time
  tools = "\n".join(
    f"- {n}: {json.dumps(t['args']) if t['args'] else '{}'} -> {t['doc']}" for n, t in TOOLS.items())
  view = ag.render(s, now, persona, maze, limit=5) or "(empty)"
  hist = "".join(f"\nYou called {c['tool']} {json.dumps(c['args'])}\nResult: {c['result']}"
                 for c in history)
  return (
    f"{s.get_str_iss()}\n"
    f"Right now it is {now.strftime('%A %Y-%m-%d %H:%M')}.\n"
    f"Your agenda:\n{view}\n\n"
    f"{trigger}\n\n"
    "You keep your own agenda with tools. Reply with exactly ONE JSON object: "
    '{"tool": "<name>", "args": {...}}\nTools:\n' + tools + "\n"
    "Record an event only if its day and time were actually stated; never guess them. "
    "If you are not sure when it is, do not record it (reply done) and ask the person next "
    "time you talk. If nothing needs "
    'recording, reply {"tool": "done", "args": {}}.' + hist + "\nYour next JSON:")


def _default_ask(prompt):
  """One JSON-constrained-by-prompt call to the local chat model (free_form, no
  schema: the project's rule is never to force a schema on multi-line text)."""
  from persona.prompt_template.gpt_structure import safe_generate_response
  gp = {"engine": "text-davinci-002", "max_tokens": 160, "temperature": 0.2, "top_p": 1,
        "stream": False, "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  return safe_generate_response(prompt, gp, 1, None, lambda r, prompt="": bool(str(r).strip()),
                                lambda r, prompt="": str(r), free_form=True, cache=False)


def run_tool_loop(persona, trigger, maze=None, ask=None, max_turns=MAX_TURNS, kind="chat"):
  """Run the loop. Returns {"calls": [...], "changed": bool, "ended": str}."""
  ask = ask or _default_ask
  s = persona.scratch
  history, changed, bad, ended = [], False, 0, "cap"
  prompt_trigger = trigger
  for turn in range(max_turns):
    try:
      reply = ask(_prompt(persona, prompt_trigger, history, maze))
    except Exception as e:
      ended = f"llm_error:{type(e).__name__}"
      break
    if reply is None:
      ended = "llm_unavailable"
      break
    name, args = parse_call(reply)
    if name is None:
      bad += 1
      ag.trace({"t": s.curr_time.strftime("%Y-%m-%d %H:%M"), "who": persona.name, "kind": kind,
                "malformed": str(reply)[:160], "why": args})
      if bad >= 2:
        ended = "malformed"
        break
      prompt_trigger = trigger + f"\nYour last reply was not usable: {args}."
      continue
    ok, result = execute(persona, name, args, maze)
    call = {"tool": name, "args": args, "ok": ok, "result": result[:300]}
    history.append(call)
    if ok and name in ("add_event", "modify_event", "cancel_event"):
      changed = True
    ag.trace({"t": s.curr_time.strftime("%Y-%m-%d %H:%M"), "who": persona.name, "kind": kind,
              "tool": name, "args": args, "ok": ok, "result": result[:200]})
    if name == "done":
      ended = "done"
      break
  return {"calls": history, "changed": changed, "ended": ended}


def chat_tick(persona, transcript, maze=None, ask=None):
  """After a conversation: let the agent record/adjust agenda items. No-op
  unless GA_AGENDA is on and the transcript states a clock time."""
  if not ag.enabled() or not str(transcript or "").strip():
    return None
  shown = str(transcript).strip()[-1500:]       # gate on what the model will be shown
  if not has_temporal_cue(shown):
    return None
  trigger = ("You just finished this conversation:\n" + shown +
             "\nDid it make you aware of an event, or did you agree to something at a specific "
             "date and time? Decide whether to add it to your agenda, and whether you will go.")
  return run_tool_loop(persona, trigger, maze, ask, kind="chat")


def notice_tick(persona, text, maze=None, ask=None):
  """After the agent reads something posted in the world (a notice board, a
  flyer): same tool loop, different trigger. No-op without a clock time."""
  shown = str(text or "").strip()[-1200:]
  if not ag.enabled() or not shown or not has_temporal_cue(shown):
    return None
  trigger = ("You just read this notice:\n" + shown +
             "\nDoes it announce an event or a deadline worth putting on your agenda, "
             "and will you go?")
  return run_tool_loop(persona, trigger, maze, ask, kind="notice")


def plan_tick(persona, maze=None, ask=None):
  """At daily planning: the agent's own notes may name events it committed to
  or is hosting (a plan written yesterday, a bootstrap storyline). Same loop."""
  s = persona.scratch
  if not ag.enabled():
    return None
  notes = "\n".join(x for x in (str(getattr(s, "currently", "") or ""),
                                 str(getattr(s, "daily_plan_req", "") or "")) if x.strip())
  notes = notes[-1500:]
  if not has_temporal_cue(notes):
    return None
  trigger = ("You are planning your day. Your own notes about what is coming up:\n" + notes +
             "\nIs there an event with a date and time that you are hosting or committed to? "
             "Put it on your agenda (going=true if you will be there).")
  return run_tool_loop(persona, trigger, maze, ask, kind="plan")
