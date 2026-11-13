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
