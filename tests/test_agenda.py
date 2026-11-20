import datetime, json, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
import agenda as ag

D = datetime.datetime


def sc(now=D(2023, 2, 13, 9, 0), sched=None):
  s = types.SimpleNamespace(curr_time=now, agenda=[], f_daily_schedule=sched or [["sleep", 420], ["work", 1020]])
  return s


def party(s, **kw):
  args = dict(title="Valentine's Day party", place="Hobbs Cafe", date="2023-02-14",
              start="5pm", end="7pm", source="chat", status="known")
  args.update(kw)
  return ag.add_event(s, **args)


def test_parse_clock_and_dates():
  assert ag.parse_clock("5pm") == 1020 and ag.parse_clock("17:30") == 1050 and ag.parse_clock(17) == 1020
  assert ag.parse_clock("12 am") == 0 and ag.parse_clock("garbage") is None
  now = D(2023, 2, 13, 9)
  assert ag._norm_date("tomorrow", now) == "2023-02-14"
  assert ag._norm_date("Feb 14", now) == "2023-02-14"
  assert ag._norm_date("Tuesday", now) == "2023-02-14"
  assert ag._norm_date("nonsense", now) is None


def test_add_rejects_unreadable():
  s = sc()
  assert ag.add_event(s, "x", date="whenever", start="5pm") is None
  assert ag.add_event(s, "x", date="tomorrow", start="late") is None
  assert s.agenda == []


def test_add_dedupes_same_event_heard_twice():
  s = sc()
  a = party(s)
  b = party(s, title="Isabella's Valentine party", start="17:00", status="going", note="invited by Isabella")
  assert a is b and len(s.agenda) == 1 and a["status"] == "going" and a["note"]
  c = party(s, title="Book club", place="Library", start="5pm")
  assert c is not a and len(s.agenda) == 2 and c["id"] == 2


def test_modify_cancel_and_ids():
  s = sc()
  e = party(s, status="going")
  ag.modify_event(s, f"#{e['id']}", start="6pm", end="8pm")
  assert (e["start"], e["end"]) == (1080, 1200)
  ag.cancel_event(s, e["id"], "feeling unwell")
  assert e["status"] == "declined" and e["note"] == "feeling unwell"
  assert ag.modify_event(s, 99, title="x") is None


def test_upcoming_render_and_due_for_party():
  s = sc(D(2023, 2, 14, 16, 25))
  party(s, status="going")
  rows = ag.upcoming(s)
  assert rows[0][0] == 35
  text = ag.render(s)
  assert "Valentine's Day party at Hobbs Cafe" in text and "starts in 35 min" in text and "[going]" in text
  assert ag.time_context(s).startswith("It is Tuesday 16:25.")
  assert ag.due(s) == []                       # default travel 8 + slack 10 = 18 < 35
  s.curr_time = D(2023, 2, 14, 16, 45)
  assert [e["id"] for e in ag.due(s)] == [1]
  s.curr_time = D(2023, 2, 14, 17, 30)
  assert "under way" in ag.render(s) and ag.due(s)
  s.curr_time = D(2023, 2, 14, 19, 5)
  assert ag.upcoming(s) == []


def test_declined_and_mark_past():
  s = sc(D(2023, 2, 14, 20, 0))
  a = party(s, status="going"); b = party(s, title="Book club", place="Library", status="known")
  ag.mark_past(s)
  assert a["status"] == "done" and b["status"] == "missed"
  s2 = sc(D(2023, 2, 13, 9)); e = party(s2); ag.cancel_event(s2, e["id"])
  assert ag.upcoming(s2) == []


def test_overlay_pins_going_event_once_and_survives_regeneration():
  s = sc(D(2023, 2, 14, 9, 0))
  party(s, status="going")
  assert ag.overlay(s) == 1
  assert sum(d for _, d in s.f_daily_schedule) == 1440
  t = 0
  for task, d in s.f_daily_schedule:
    if "Valentine" in task: assert (t, d) == (1020, 120)
    t += d
  assert ag.overlay(s) == 0                    # idempotent: decomposed items are not clobbered
  s.f_daily_schedule = [["sleep", 420], ["work", 1020]]   # daily regeneration
  ag.reset_overlay(s)
  assert ag.overlay(s) == 1 and any("Valentine" in x[0] for x in s.f_daily_schedule)


def test_overlay_ignores_known_declined_and_other_days():
  s = sc(D(2023, 2, 13, 9, 0))
  party(s, status="going")                     # tomorrow's party: not pinned today
  party(s, title="Dentist", date="2023-02-13", start="2pm", status="known")
  assert ag.overlay(s) == 0


def test_travel_minutes_uses_maze_distance():
  maze = types.SimpleNamespace(address_tiles={"the Ville:Hobbs Cafe:cafe:counter": {(10, 10), (40, 10)}})
  p = types.SimpleNamespace(scratch=types.SimpleNamespace(curr_tile=(40, 40)))
  assert ag.travel_minutes(p, "Hobbs Cafe", maze) == 5 + 3      # 30 tiles * 10 s = 5 min + slack
  assert ag.travel_minutes(p, "Mars Base", maze) == 8


def test_scratch_roundtrip_and_old_save(tmp_path):
  from persona.memory_structures.scratch import Scratch
  src = os.path.join(os.path.dirname(__file__), "..", "environment", "frontend_server", "storage",
                     "base_the_ville_n55", "personas", "Isabella Rodriguez", "bootstrap_memory", "scratch.json")
  s = Scratch(src)
  assert s.agenda == []                        # a save without the field loads as empty
  s.curr_time = s.act_start_time = D(2023, 2, 13, 9)
  party(s, status="going")
  f = tmp_path / "scratch.json"
  s.save(str(f))
  d = json.load(open(f))
  assert d["agenda"][0]["title"].startswith("Valentine")
  assert Scratch(str(f)).agenda == d["agenda"]


# ---- free choice sees the agenda -----------------------------------------
def _fc_persona(now, sched):
  import types as T
  sc = T.SimpleNamespace(
    curr_time=now, f_daily_schedule=sched, needs={"hunger": 80}, resource_goals=[],
    chatting_with=None, agenda=[], curr_tile=(0, 0), act_address="the Ville:Home:main room:desk",
    get_f_daily_schedule_index=lambda: 1)
  return T.SimpleNamespace(name="Ann Lee", scratch=sc)


def test_free_choice_offers_due_event_and_attending_pins_it(monkeypatch):
  import persona.cognitive_modules.plan as plan
  import persona.prompt_template.run_gpt_prompt as rgp
  now = D(2023, 2, 14, 16, 45)
  p = _fc_persona(now, [["sleep", 420], ["read at home", 1020]])
  party(p.scratch, status="known")
  seen = {}

  def fake(persona, item, nxt, pulls):
    seen["pulls"] = list(pulls)
    return {"choice": 1, "activity": "", "minutes": 30, "why": "invited"}
  monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity", fake)
  monkeypatch.setenv("GA_TRACE_DIR", os.path.join(os.path.dirname(__file__), "_tmp_trace"))
  out = plan._free_choice(p)
  assert seen["pulls"] == ["attending Valentine's Day party at Hobbs Cafe"]
  assert out["choice"] == 0 and out["pulls"] == []
  ev = p.scratch.agenda[0]
  assert ev["status"] == "going" and ev["pinned"] == "2023-02-14"
  assert any("Valentine" in x[0] for x in p.scratch.f_daily_schedule)
  assert p.scratch.choice_stats["event"] == 1


def test_free_choice_declining_leaves_event_known(monkeypatch):
  import persona.cognitive_modules.plan as plan
  import persona.prompt_template.run_gpt_prompt as rgp
  p = _fc_persona(D(2023, 2, 14, 16, 45), [["sleep", 420], ["read at home", 1020]])
  party(p.scratch, status="known")
  monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity",
                      lambda persona, i, n, pulls: {"choice": 0, "activity": "", "minutes": 30, "why": "tired"})
  monkeypatch.setenv("GA_TRACE_DIR", os.path.join(os.path.dirname(__file__), "_tmp_trace"))
  out = plan._free_choice(p)
  assert out["choice"] == 0 and p.scratch.agenda[0]["status"] == "known"
  assert p.scratch.f_daily_schedule == [["sleep", 420], ["read at home", 1020]]


def test_no_agenda_means_unchanged_behaviour(monkeypatch):
  import persona.cognitive_modules.plan as plan
  import persona.prompt_template.run_gpt_prompt as rgp
  p = _fc_persona(D(2023, 2, 14, 10, 0), [["sleep", 420], ["read at home", 1020]])
  party(p.scratch, status="known")             # 7 hours away: not due
  seen = {}
  def fake(persona, i, n, pulls):
    seen["p"] = list(pulls)
    return None
  monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity", fake)
  assert plan._free_choice(p) is None and seen["p"] == []


def test_choose_prompt_carries_agenda_and_laya_can_parse_it(monkeypatch):
  import persona.prompt_template.run_gpt_prompt as rgp
  from persona.prompt_template import laya_choose
  now = D(2023, 2, 14, 16, 45)
  p = _fc_persona(now, [["read at home", 1440]])
  p.a_mem = types.SimpleNamespace(seq_event=[])
  for name in ("get_str_iss", "get_str_name", "get_str_firstname", "get_str_needs_summary", "get_str_financial_summary"):
    setattr(p.scratch, name, (lambda n=name: {"get_str_name": "Ann Lee", "get_str_firstname": "Ann"}.get(n, "")))
  party(p.scratch, status="known")
  p.scratch._agenda_pull_notes = {"attending Valentine's Day party at Hobbs Cafe":
                                  ag.describe(p.scratch.agenda[0], now)}
  captured = {}
  monkeypatch.setenv("GA_CHOICE_LAYA", "0")
  monkeypatch.setattr(rgp, "safe_generate_response",
                      lambda prompt, *a, **k: captured.setdefault("prompt", prompt) and None)
  rgp.run_gpt_prompt_choose_activity(p, "read at home", "", ["attending Valentine's Day party at Hobbs Cafe"])
  pr = captured["prompt"]
  assert "agenda" in pr and "starts in 15 min" in pr and "going needs no special reason" in pr
  state, qs, keys = laya_choose.parse_prompt(pr, "Ann Lee")
  assert keys == ["plan", "pull1", "own"] and "Valentine" in qs["choose"]["criteria"]["pull1"]
  assert "starts in 15 min" in state["situation"]


def test_event_already_in_plan_is_not_offered_again(monkeypatch):
  import persona.cognitive_modules.plan as plan
  import persona.prompt_template.run_gpt_prompt as rgp
  p = _fc_persona(D(2023, 2, 14, 17, 20), [["sleep", 420], ["Valentine's Day party at Hobbs Cafe", 600]])
  party(p.scratch, status="going")
  seen = {}
  def fake(persona, i, n, pulls):
    seen["p"] = list(pulls)
    return None
  monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity", fake)
  plan._free_choice(p)
  assert seen["p"] == []


def test_stated_date_wins_over_weekday_name():
  """n76: 'Saturday, February 15' was read as the next Saturday (Feb 18)."""
  now = D(2023, 2, 13, 9, 0)                                   # a Monday
  assert ag._norm_date("Saturday, February 15", now) == "2023-02-15"
  assert ag._norm_date("Saturday, February 17, 2023", now) == "2023-02-17"
  assert ag._norm_date("Monday, February 13", now) == "2023-02-13"
  assert ag._norm_date("Tue Feb 14th", now) == "2023-02-14"
  assert ag._norm_date("Friday, 2023-02-17", now) == "2023-02-17"
  # a weekday alone, or with something that is not a date, still means the next such day
  assert ag._norm_date("Saturday", now) == "2023-02-18"
  assert ag._norm_date("Thurs.", now) == "2023-02-16"
  assert ag._norm_date("Saturday evening", now) == "2023-02-18"
  # words that merely start like a weekday are not one
  assert ag._norm_date("sunny afternoon", now) is None
  assert ag._norm_date("monthly meeting", now) is None


def _choice_rows(tmp_path):
  with open(tmp_path / "agenda.jsonl") as f:
    return [json.loads(l) for l in f if '"choice"' in l]


def test_trace_keeps_the_deciders_answer_when_an_event_is_taken(monkeypatch, tmp_path):
  """The attended event is run as the plan (out['choice'] becomes 0); the trace
  must still say the decider chose the event, not 'follow the plan'."""
  import persona.cognitive_modules.plan as plan
  import persona.prompt_template.run_gpt_prompt as rgp
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  p = _fc_persona(D(2023, 2, 14, 16, 45), [["sleep", 420], ["read at home", 1020]])
  party(p.scratch, status="known")
  monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity",
                      lambda persona, i, n, pulls: {"choice": 1, "activity": "", "minutes": 30, "why": "invited"})
  out = plan._free_choice(p)
  assert out["choice"] == 0                                    # runtime behaviour unchanged
  row = _choice_rows(tmp_path)[-1]
  assert row["choice"] == 1 and row["took"] == "event" and row["attended"] == p.scratch.agenda[0]["id"]


def test_trace_records_follow_the_plan_as_plan(monkeypatch, tmp_path):
  import persona.cognitive_modules.plan as plan
  import persona.prompt_template.run_gpt_prompt as rgp
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  p = _fc_persona(D(2023, 2, 14, 16, 45), [["sleep", 420], ["read at home", 1020]])
  party(p.scratch, status="known")
  monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity",
                      lambda persona, i, n, pulls: {"choice": 0, "activity": "", "minutes": 30, "why": "tired"})
  plan._free_choice(p)
  row = _choice_rows(tmp_path)[-1]
  assert row["choice"] == 0 and row["took"] == "plan" and row["attended"] is None


def test_hearing_again_does_not_undo_a_decision():
  """A repeat mention arrives as "known": it must not revive a declined event
  nor downgrade one the agent is going to; an explicit "going" still wins."""
  s = sc()
  e = party(s, status="going")
  assert party(s, status="known") is e and e["status"] == "going"
  ag.cancel_event(s, e["id"], "feeling unwell")
  assert party(s, status="known") is e and e["status"] == "declined" and len(s.agenda) == 1
  assert ag.upcoming(s) == []                                  # not offered again
  assert party(s, status="going") is e and e["status"] == "going"   # changed their mind


# --- cancel / move: the block leaves the schedule ---------------------------
BASE = [["sleep", 420], ["work", 1020]]


def club(s, **kw):
  args = dict(title="Book club", place="Library", date="today", start="17:00", end="18:00", status="going")
  args.update(kw)
  ev = ag.add_event(s, **args)
  ag.overlay(s)
  return ev


def blocks(s, word="Book club"):
  return [x for x in s.f_daily_schedule if word in x[0]]


def test_cancel_takes_the_block_out_of_todays_schedule():
  """Before: only the 'pinned' flag was cleared, the agent still went."""
  s = sc()
  e = club(s)
  assert blocks(s) == [["Book club at Library", 60]]
  ag.cancel_event(s, e["id"], "tired")
  assert s.f_daily_schedule == BASE and "pin" not in e and "pinned" not in e


def test_moving_an_event_leaves_one_block_at_the_new_time():
  """Before: the old block stayed and the event was in the schedule twice."""
  s = sc()
  e = club(s)
  ag.modify_event(s, e["id"], start="19:00", end="20:00")
  ag.overlay(s)
  assert s.f_daily_schedule == [["sleep", 420], ["work", 720], ["Book club at Library", 60], ["work", 240]]
  ag.modify_event(s, e["id"], date="tomorrow")           # moved off today: nothing left
  ag.overlay(s)
  assert s.f_daily_schedule == BASE
  assert sum(d for _, d in s.f_daily_schedule) == 1440


def test_what_was_planned_under_the_event_comes_back():
  s = sc(sched=[["sleep", 420], ["work", 570], ["paint", 30], ["cook dinner", 40], ["eat", 380]])
  before = [list(x) for x in s.f_daily_schedule]
  e = club(s)                                            # 17:00-18:00: all of cooking, 20 min of eating
  assert e["pin"]["replaced"] == [["cook dinner", 40], ["eat", 20]]
  ag.cancel_event(s, e["id"])
  assert s.f_daily_schedule == before


def test_cancel_after_a_replan_that_kept_the_block():
  """reset_overlay + overlay runs after every chat replan; the second pin
  finds its own block in the schedule and must not record it as the original."""
  s = sc()
  e = club(s)
  ag.reset_overlay(s)
  ag.overlay(s)
  assert e["pin"]["replaced"] == [["work", 60]]
  ag.cancel_event(s, e["id"])
  assert s.f_daily_schedule == BASE


def test_cancel_one_of_two_overlapping_events():
  s = sc()
  a = club(s, end="19:00")
  b = club(s, title="Dentist", place="Clinic", start="18:00", end="19:00")
  ag.cancel_event(s, a["id"])
  assert blocks(s) == [] and blocks(s, "Dentist") == [["Dentist at Clinic", 60]]
  ag.cancel_event(s, b["id"])                            # must not bring the book club back
  assert s.f_daily_schedule == BASE


def test_a_rebuilt_schedule_is_left_alone():
  """The schedule was regenerated (new day's plan): the old record no longer
  describes it, cancelling must not write yesterday's pieces into it."""
  s = sc()
  e = club(s)
  s.curr_time = D(2023, 2, 14, 9, 0)
  s.f_daily_schedule = [["sleep", 480], ["garden", 960]]
  ag.cancel_event(s, e["id"])
  assert s.f_daily_schedule == [["sleep", 480], ["garden", 960]]


# --- asking the other speaker for the missing day/time ----------------------
def chat(*lines):
  return [list(x) for x in lines]


def test_states_clock_time():
  for t in ("party from 5 to 7", "see you at 5", "come around six", "5:30pm", "at noon", "until 9"):
    assert ag.states_clock_time(t), t
  for t in ("let's meet tonight", "this afternoon", "on Tuesday", "around 3 people came",
            "from Monday to Friday", "How are you today?"):
    assert not ag.states_clock_time(t), t


def test_ask_when_the_other_names_an_event_only_they_have():
  host, guest = sc(), sc()
  party(host, status="going")
  c = chat(("Isabella", "I'm getting everything ready for the Valentine's Day party at Hobbs Cafe!"))
  hint = ag.ask_hint(c, "Ann", guest, host)
  assert "Isabella" in hint and "which day and what time" in hint
  assert "17" not in hint and "5" not in hint                     # the answer is never given away
  party(guest)                                                    # I already have it: nothing to ask
  assert ag.ask_hint(c, "Ann", guest, host) == ""


def test_no_ask_when_a_time_was_said_or_i_already_asked():
  host, guest = sc(), sc()
  party(host, status="going")
  said = chat(("Isabella", "Come to the Valentine's Day party tomorrow from 5 to 7!"))
  assert ag.ask_hint(said, "Ann", guest, host) == ""
  asked = chat(("Isabella", "I'm planning the Valentine's Day party."), ("Ann", "Lovely! When is it?"),
               ("Isabella", "The Valentine's Day party will be great."))
  assert ag.ask_hint(asked, "Ann", guest, host) == ""
  mine = chat(("Ann", "I heard about the Valentine's Day party."))    # my own line: not my turn to ask
  assert ag.ask_hint(mine, "Ann", guest, host) == ""
  assert ag.ask_hint([], "Ann", guest, host) == ""


def test_ask_when_something_is_proposed_for_a_coming_day():
  assert "what time" in ag.ask_hint(chat(("Bob", "Let's have lunch tomorrow.")), "Ann")
  assert "Friday" in ag.ask_hint(chat(("Bob", "Why don't you come by on Friday?")), "Ann")
  for line in ("How was last Tuesday?", "I'll send you the report tomorrow.", "It's a Monday, so we should check.",
               "What are you having for dinner tonight?", "Let's have lunch tomorrow at noon."):
    assert ag.ask_hint(chat(("Bob", line)), "Ann") == "", line


def test_answer_with_the_agenda_when_asked_when():
  host = sc()
  party(host, status="going")
  c = chat(("Isabella", "I'm preparing the Valentine's Day party."), ("Ann", "Sounds fun! When is it?"))
  hint = ag.answer_hint(c, "Isabella", host)
  assert "Tuesday February 14, tomorrow 17:00-19:00" in hint and "Hobbs Cafe" in hint and "Ann" in hint
  assert ag.answer_hint(c, "Isabella", sc()) == ""                 # nothing on my agenda: no line
  other = chat(("Isabella", "I'm repainting the fence."), ("Ann", "When did you start?"))
  assert ag.answer_hint(other, "Isabella", host) == ""             # the question is about something else


def test_chat_hints_answer_first_and_switch(monkeypatch):
  host, guest = sc(), sc()
  party(host, status="going")
  c = chat(("Isabella", "I'm preparing the Valentine's Day party."), ("Ann", "When is it?"))
  assert "On your agenda" in ag.chat_hints(c, "Isabella", host, guest)
  assert "brought up" in ag.chat_hints(c[:1], "Ann", guest, host)
  monkeypatch.setenv("GA_AGENDA_ASK", "0")
  assert ag.chat_hints(c[:1], "Ann", guest, host) == ""


def test_the_hint_reaches_the_chat_prompt_right_after_the_conversation(monkeypatch):
  import persona.prompt_template.run_gpt_prompt as rgp
  seen = {}
  monkeypatch.setattr(rgp, "generate_prompt", lambda inp, tpl: seen.setdefault("input", inp) and "PROMPT")
  monkeypatch.setattr(rgp, "ChatGPT_safe_generate_response_OLD",
                      lambda *a, **k: {"utterance": "ok", "end": False})

  def who(name):
    s = sc()
    s.name, s.curr_tile, s.act_address = name, (0, 0), ""
    s.get_str_iss = lambda: f"Name: {name}"
    return types.SimpleNamespace(name=name, scratch=s, a_mem=types.SimpleNamespace(seq_chat=[]))
  maze = types.SimpleNamespace(access_tile=lambda t: {"sector": "Hobbs Cafe", "arena": "cafe"}, address_tiles={})
  ann, isa = who("Ann Lee"), who("Isabella Rodriguez")
  party(isa.scratch, status="going")
  c = chat(("Isabella Rodriguez", "I'm getting ready for the Valentine's Day party!"))
  rgp.run_gpt_generate_iterative_chat_utt(maze, ann, isa, {}, "ctx", c)
  convo = seen["input"][8]
  assert convo.startswith("Isabella Rodriguez: I'm getting ready")
  assert convo.rstrip().endswith("(and where, if you do not know).") and "which day and what time" in convo
  seen.clear()                                             # Isabella's own turn: nothing to ask
  rgp.run_gpt_generate_iterative_chat_utt(maze, isa, ann, {}, "ctx", c)
  assert "which day" not in seen["input"][8]
  seen.clear()                                             # asked: Isabella gets her agenda line
  rgp.run_gpt_generate_iterative_chat_utt(maze, isa, ann, {}, "ctx", c + chat(("Ann Lee", "Lovely, when is it?")))
  assert "Tuesday February 14, tomorrow 17:00-19:00" in seen["input"][8]


def test_utterance_prompt_lists_each_memory_once_and_no_empty_past_context(monkeypatch):
  import persona.prompt_template.run_gpt_prompt as rgp
  seen = {}
  monkeypatch.setattr(rgp, "generate_prompt", lambda inp, tpl: seen.setdefault("input", inp) and "PROMPT")
  monkeypatch.setattr(rgp, "ChatGPT_safe_generate_response_OLD",
                      lambda *a, **k: {"utterance": "ok", "end": False})

  def who(name):
    s = sc()
    s.name, s.curr_tile, s.act_address = name, (0, 0), ""
    s.get_str_iss = lambda: f"Name: {name}"
    return types.SimpleNamespace(name=name, scratch=s, a_mem=types.SimpleNamespace(seq_chat=[]))
  maze = types.SimpleNamespace(access_tile=lambda t: {"sector": "Hobbs Cafe", "arena": "cafe"}, address_tiles={})
  N = lambda i, d: types.SimpleNamespace(node_id=f"node_{i}", description=d)
  a, b = N(1, "Ann likes the cafe's croissants"), N(2, "Isabella is planning a party")
  retrieved = {"rel": [a, b], "Isabella": [b, a], "topic": [N(3, "Ann likes the cafe's croissants.")]}
  rgp.run_gpt_generate_iterative_chat_utt(maze, who("Ann Lee"), who("Isabella Rodriguez"), retrieved, "ctx", [])
  assert seen["input"][2] == "- Ann likes the cafe's croissants\n- Isabella is planning a party\n"
  assert seen["input"][3] == ""
  tpl = open(os.path.join(os.path.dirname(rgp.__file__), "v3_ChatGPT", "iterative_convo_v1.txt")).read()
  filled = tpl.split("<commentblockmarker>###</commentblockmarker>")[1].replace("!<INPUT 3>!", "")
  assert "Past Context" not in filled and "PART 2. \nCurrent Location" in filled
