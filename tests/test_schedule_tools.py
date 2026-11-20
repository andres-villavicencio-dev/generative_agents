import datetime, json, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
import agenda as ag
import schedule_tools as st

D = datetime.datetime
os.environ.setdefault("GA_TRACE_DIR", os.path.join(os.path.dirname(__file__), "_tmp_trace_tools"))


def persona(now=D(2023, 2, 13, 9, 30)):
  sc = types.SimpleNamespace(curr_time=now, agenda=[], f_daily_schedule=[["sleep", 420], ["work", 1020]],
                             get_str_iss=lambda: "Name: Ann Lee", curr_tile=(0, 0))
  return types.SimpleNamespace(name="Ann Lee", scratch=sc)


def scripted(*replies):
  it = iter(replies); prompts = []
  def ask(prompt):
    prompts.append(prompt)
    return next(it, '{"tool": "done", "args": {}}')
  ask.prompts = prompts
  return ask


ADD = json.dumps({"tool": "add_event", "args": {"title": "Valentine's party", "place": "Hobbs Cafe",
                  "date": "tomorrow", "start": "17:00", "end": "19:00", "going": True}})


def test_parse_call_variants():
  assert st.parse_call('{"tool": "done", "args": {}}') == ("done", {})
  assert st.parse_call('Sure!\n```json\n{"tool": "cancel_event", "args": {"id": 2, "reason": "sick"}}\n```')[0] == "cancel_event"
  assert st.parse_call('{"name": "check_schedule", "arguments": "{}"}') == ("check_schedule", {})
  name, err = st.parse_call('{"tool": "teleport"}')
  assert name is None and "unknown tool" in err
  assert st.parse_call("no json here")[0] is None


def test_add_then_done_records_event_and_feeds_results_back():
  p = persona()
  ask = scripted('{"tool": "check_schedule", "args": {}}', ADD, '{"tool": "done", "args": {}}')
  out = st.run_tool_loop(p, "You just talked about a party.", ask=ask)
  assert [c["tool"] for c in out["calls"]] == ["check_schedule", "add_event", "done"]
  assert out["changed"] and out["ended"] == "done"
  ev = p.scratch.agenda[0]
  assert ev["date"] == "2023-02-14" and ev["status"] == "going" and ev["start"] == 1020
  assert "Recorded:" in ask.prompts[2] and "Agenda:" in ask.prompts[1]      # results reach the next turn


def test_going_today_pins_into_schedule_immediately():
  p = persona()
  call = json.dumps({"tool": "add_event", "args": {"title": "Book club", "place": "Library",
                     "date": "today", "start": "5pm", "going": "yes"}})
  st.run_tool_loop(p, "x", ask=scripted(call, '{"tool": "done"}'))
  assert any("Book club" in x[0] for x in p.scratch.f_daily_schedule)
  assert sum(d for _, d in p.scratch.f_daily_schedule) == 1440


def test_bad_args_return_error_then_agent_can_retry():
  p = persona()
  bad = '{"tool": "add_event", "args": {"title": "x", "date": "someday", "start": "late"}}'
  out = st.run_tool_loop(p, "x", ask=scripted(bad, ADD, '{"tool": "done", "args": {}}'))
  assert out["calls"][0]["ok"] is False and out["calls"][1]["ok"] is True and len(p.scratch.agenda) == 1


def test_malformed_gets_one_repair_then_stops():
  p = persona()
  ask = scripted("blah", "still blah")
  out = st.run_tool_loop(p, "x", ask=ask)
  assert out["ended"] == "malformed" and out["calls"] == [] and p.scratch.agenda == []
  assert "not usable" in ask.prompts[1]
  ask2 = scripted("blah", '{"tool": "done", "args": {}}')
  assert st.run_tool_loop(persona(), "x", ask=ask2)["ended"] == "done"      # the repair worked


def test_turn_cap_and_llm_failures_are_safe():
  p = persona()
  out = st.run_tool_loop(p, "x", ask=scripted(*['{"tool": "check_schedule", "args": {}}'] * 9))
  assert len(out["calls"]) == st.MAX_TURNS and out["ended"] == "cap"
  assert st.run_tool_loop(persona(), "x", ask=lambda pr: None)["ended"] == "llm_unavailable"
  def boom(pr): raise RuntimeError("down")
  assert st.run_tool_loop(persona(), "x", ask=boom)["ended"].startswith("llm_error")


def test_modify_and_cancel():
  p = persona()
  st.run_tool_loop(p, "x", ask=scripted(ADD, '{"tool": "modify_event", "args": {"id": 1, "start": "18:00", "end": "20:00"}}',
                                        '{"tool": "cancel_event", "args": {"id": 1, "reason": "family dinner"}}', '{"tool":"done"}'))
  ev = p.scratch.agenda[0]
  assert ev["start"] == 1080 and ev["status"] == "declined" and ev["note"] == "family dinner"
  out = st.run_tool_loop(p, "x", ask=scripted('{"tool": "cancel_event", "args": {"id": 9}}', '{"tool":"done"}'))
  assert out["calls"][0]["ok"] is False


def test_chat_tick_gates_and_runs():
  p = persona()
  never = scripted()
  assert st.chat_tick(p, "We talked about the weather and my cat.", ask=never) is None and never.prompts == []
  assert st.chat_tick(p, "", ask=never) is None
  ask = scripted(ADD, '{"tool": "done", "args": {}}')
  out = st.chat_tick(p, "Isabella: Come to my party tomorrow from 5 to 7 pm at Hobbs Cafe!\nAnn: I'll be there!", ask=ask)
  assert out["changed"] and "Come to my party" in ask.prompts[0] and len(p.scratch.agenda) == 1


def test_chat_tick_respects_flag(monkeypatch):
  monkeypatch.setenv("GA_AGENDA", "0")
  assert st.chat_tick(persona(), "see you tomorrow at 5 pm", ask=scripted()) is None


def test_missing_date_means_next_occurrence_of_that_hour():
  # 17:19 chat that says "from 5 to 7 pm" without a day: 5 pm already passed -> tomorrow
  p = persona(D(2023, 2, 13, 17, 19))
  call = json.dumps({"tool": "add_event", "args": {"title": "Valentine's party", "place": "Hobbs Cafe",
                     "start": "5pm", "end": "7pm", "going": True}})
  st.run_tool_loop(p, "x", ask=scripted(call, '{"tool":"done"}'))
  assert p.scratch.agenda[0]["date"] == "2023-02-14"
  p2 = persona(D(2023, 2, 13, 9, 0))                    # morning: 5 pm is still ahead -> today
  st.run_tool_loop(p2, "x", ask=scripted(call, '{"tool":"done"}'))
  assert p2.scratch.agenda[0]["date"] == "2023-02-13"


def test_plan_tick_reads_own_notes_and_hosts():
  p = persona(D(2023, 2, 13, 0, 5))
  p.scratch.currently = "Isabella is hosting a Valentine's Day party at Hobbs Cafe on February 14th from 5pm to 7pm."
  p.scratch.daily_plan_req = "run the cafe"
  call = json.dumps({"tool": "add_event", "args": {"title": "Valentine's Day party", "place": "Hobbs Cafe",
                     "date": "2023-02-14", "start": "17:00", "end": "19:00", "going": True}})
  ask = scripted(call, '{"tool": "done"}')
  out = st.plan_tick(p, ask=ask)
  assert out["changed"] and "hosting a Valentine" in ask.prompts[0] and p.scratch.agenda[0]["date"] == "2023-02-14"
  q = persona(); q.scratch.currently = "likes gardening"; q.scratch.daily_plan_req = "water the plants"
  assert st.plan_tick(q, ask=scripted()) is None            # no time/day in the notes: no LLM call


def test_notice_tick():
  p = persona()
  out = st.notice_tick(p, "NOTICE: town meeting on Friday at 6 pm in the library hall.",
                       ask=scripted('{"tool": "add_event", "args": {"title": "Town meeting", "place": "library hall", "date": "friday", "start": "18:00", "going": false}}', '{"tool":"done"}'))
  assert out["changed"] and p.scratch.agenda[0]["status"] == "known" and p.scratch.agenda[0]["date"] == "2023-02-17"
  assert st.notice_tick(p, "Fresh bread available", ask=scripted()) is None


def test_past_start_is_rejected_and_agent_can_correct():
  # the n75 failure: 11:04 chat -> party recorded at 08:00 "today"
  p = persona(D(2023, 2, 13, 11, 4))
  bad = json.dumps({"tool": "add_event", "args": {"title": "Party", "place": "Hobbs Cafe",
                    "date": "2023-02-13", "start": "08:00", "going": True}})
  ask = scripted(bad, '{"tool":"done"}')
  r = st.run_tool_loop(p, "x", ask=ask)
  assert p.scratch.agenda == [] and not r["changed"]
  assert "already in the past" in r["calls"][0]["result"] and not r["calls"][0]["ok"]
  # corrected retry for a future time is accepted
  good = json.dumps({"tool": "add_event", "args": {"title": "Party", "place": "Hobbs Cafe",
                     "date": "2023-02-14", "start": "17:00", "going": True}})
  st.run_tool_loop(p, "x", ask=scripted(bad, good, '{"tool":"done"}'))
  assert len(p.scratch.agenda) == 1 and p.scratch.agenda[0]["date"] == "2023-02-14"
  # started 14 minutes ago is also past; same minute and later today are fine
  assert ag.start_in_past("today", "22:00", D(2023, 2, 13, 22, 14))
  assert ag.start_in_past("today", "22:14", D(2023, 2, 13, 22, 14, 30)) is None
  assert ag.start_in_past("today", "23:00", D(2023, 2, 13, 22, 14)) is None
  assert ag.start_in_past("garbage", "xx", D(2023, 2, 13, 22, 14)) is None


def test_prompts_tell_agent_to_ask_instead_of_guessing():
  p = persona()
  txt = st._prompt(p, "You just finished this conversation: ...", [], None)
  assert "never guess" in txt and "ask the person" in txt
  assert "ask them" in ag.ASK_WHEN_UNSURE and "instead of guessing" in ag.ASK_WHEN_UNSURE


def test_gate_needs_a_clock_time_not_a_day_word():
  """n75/n76: day words opened 59% of the loops ("how are you today", "you
  may", "I will march on") and those produced invented times."""
  for chat in ("How are you today?", "You may sit here.", "I will march on.", "See you tomorrow!",
               "Come to my party on Tuesday!", "I'm inviting everyone this weekend."):
    assert not st.has_temporal_cue(chat), chat
    assert st.chat_tick(persona(), chat, ask=scripted()) is None
  for chat in ("Come to my party tomorrow from 5 to 7 pm", "see you at 5:30", "let's meet at noon",
               "the meeting is at three o'clock"):
    assert st.has_temporal_cue(chat), chat


def test_gate_looks_only_at_the_text_the_model_is_shown():
  ask = scripted()
  early = "Ann: see you at 5 pm.\n" + "Bob: nice weather we are having, is it not?\n" * 60
  assert st.chat_tick(persona(), early, ask=ask) is None and ask.prompts == []   # the time was cut off
  late = "Bob: nice weather we are having, is it not?\n" * 60 + "Ann: see you at 5 pm."
  assert st.chat_tick(persona(), late, ask=ask) is not None and "5 pm" in ask.prompts[0]


def call(tool, **args):
  return json.dumps({"tool": tool, "args": args})


def test_modify_cannot_move_an_event_into_the_past():
  p = persona(D(2023, 2, 13, 9, 30))
  st.run_tool_loop(p, "x", ask=scripted(ADD))
  ev = p.scratch.agenda[0]
  before = dict(ev)
  for args in ({"date": "2023-02-12"}, {"date": "today", "start": "08:00"}, {"date": "Sunday, February 12"}):
    ok, msg = st.execute(p, "modify_event", dict(args, id=1))
    assert ok is False and "in the past" in msg and ev == before, args
  ok, _ = st.execute(p, "modify_event", {"id": 1, "date": "today", "start": "18:00", "end": "20:00"})
  assert ok and (ev["date"], ev["start"], ev["end"]) == ("2023-02-13", 1080, 1200)
  # changing only the place of an event under way is not a move into the past
  p.scratch.curr_time = D(2023, 2, 13, 18, 30)
  ok, _ = st.execute(p, "modify_event", {"id": 1, "place": "The Rose and Crown Pub"})
  assert ok and ev["place"] == "The Rose and Crown Pub"


def test_modify_reports_unreadable_values_instead_of_saying_updated():
  p = persona()
  st.run_tool_loop(p, "x", ask=scripted(ADD))
  before = dict(p.scratch.agenda[0])
  for args in ({"date": "someday"}, {"start": "late"}, {"end": "whenever"}):
    ok, msg = st.execute(p, "modify_event", dict(args, id=1))
    assert ok is False and "cannot read" in msg and p.scratch.agenda[0] == before, args


def test_modify_going_changes_the_decision():
  p = persona()
  st.execute(p, "add_event", {"title": "Book club", "place": "Library", "date": "today",
                              "start": "17:00", "going": False})
  ev = p.scratch.agenda[0]
  assert ev["status"] == "known" and "pinned" not in ev
  ok, msg = st.execute(p, "modify_event", {"id": 1, "going": "true"})
  assert ok and ev["status"] == "going" and "[going]" in msg
  assert ev["pinned"] == "2023-02-13" and any("Book club" in x[0] for x in p.scratch.f_daily_schedule)
  st.execute(p, "modify_event", {"id": 1, "going": False})
  assert ev["status"] == "known"
  st.execute(p, "cancel_event", {"id": 1, "reason": "tired"})
  st.execute(p, "modify_event", {"id": 1, "going": False})       # false never revives a declined event
  assert ev["status"] == "declined"
  assert "going" in st._prompt(p, "x", [], None).split("modify_event")[1].split("cancel_event")[0]


def test_tools_take_a_dropped_event_out_of_the_schedule():
  base = [["sleep", 420], ["work", 1020]]
  add = {"title": "Book club", "place": "Library", "date": "today", "start": "17:00", "going": True}
  for drop in (("cancel_event", {"id": 1, "reason": "tired"}), ("modify_event", {"id": 1, "going": False}),
               ("modify_event", {"id": 1, "date": "tomorrow"})):
    p = persona()
    st.execute(p, "add_event", dict(add))
    assert ["Book club at Library", 60] in p.scratch.f_daily_schedule
    ok, _ = st.execute(p, *drop)
    assert ok and p.scratch.f_daily_schedule == base, drop
  p = persona()
  st.execute(p, "add_event", dict(add))
  st.execute(p, "modify_event", {"id": 1, "start": "19:00", "end": "20:00"})
  assert [x for x in p.scratch.f_daily_schedule if "Book club" in x[0]] == [["Book club at Library", 60]]
  assert p.scratch.f_daily_schedule[1] == ["work", 720]            # the old 17:00 slot is work again
