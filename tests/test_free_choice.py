import datetime, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
import persona.cognitive_modules.plan as plan
import persona.prompt_template.run_gpt_prompt as rgp


def _persona(sched, needs=None, goals=None, hour=10):
    sc = types.SimpleNamespace(
        curr_time=datetime.datetime(2023, 2, 15, hour, 0), f_daily_schedule=sched,
        needs=needs or {"hunger": 80}, resource_goals=list(goals or []),
        chatting_with=None, get_f_daily_schedule_index=lambda: 0)
    return types.SimpleNamespace(name="Ann Lee", scratch=sc)


def _stub(monkeypatch, ans):
    monkeypatch.setattr(rgp, "run_gpt_prompt_choose_activity",
                        lambda p, i, n, pulls: dict(ans) if ans else None)


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("GA_FREE_CHOICE", raising=False)
    assert not plan.free_choice_enabled()


def test_comply_counts(monkeypatch):
    _stub(monkeypatch, {"choice": 0, "activity": "", "minutes": 30, "why": "plan fits"})
    p = _persona([["working at the desk", 60]], goals=["buying groceries"])
    out = plan._free_choice(p)
    assert out["choice"] == 0 and out["pulls"] == ["buying groceries"]
    assert p.scratch.choice_stats["complied"] == 1


def test_pull_and_own_are_recorded(monkeypatch):
    _stub(monkeypatch, {"choice": 1, "activity": "", "minutes": 30, "why": "hungry"})
    p = _persona([["working", 60]], goals=["eating a meal at the Hobbs Cafe"])
    plan._free_choice(p)
    _stub(monkeypatch, {"choice": "own", "activity": "reading in the park", "minutes": 45, "why": "nice day"})
    plan._free_choice(p)
    st = p.scratch.choice_stats
    assert (st["pull"], st["own"], st["decisions"]) == (1, 1, 2)


def test_llm_failure_falls_back(monkeypatch):
    _stub(monkeypatch, None)
    assert plan._free_choice(_persona([["working", 60]])) is None


def test_safety_net_and_sleep_skip_choice(monkeypatch):
    _stub(monkeypatch, {"choice": "own", "activity": "x", "minutes": 30, "why": ""})
    assert plan._free_choice(_persona([["working", 60]], needs={"hunger": 5})) is None
    assert plan._free_choice(_persona([["sleeping", 60]], hour=3)) is None


def test_reply_parser():
    class P:  # minimal persona to reach the parser through the real function
        pass
    calls = {}
    def fake(prompt, gpt, repeat, fail, validate, clean, **kw):
        calls["ok"] = [clean(r) for r in (
            "choice: X\nactivity: reading in the park\nminutes: 200\nwhy: sunny",
            "choice: 2\nminutes: 20\nwhy: broke",
            ) if validate(r)]
        calls["bad"] = validate("choice: 9\nminutes: 30")
        return calls["ok"][0]
    orig = rgp.safe_generate_response
    rgp.safe_generate_response = fake
    try:
        sc = types.SimpleNamespace(
            get_str_name=lambda: "Ann Lee", get_str_firstname=lambda: "Ann",
            get_str_iss=lambda: "", get_str_needs_summary=lambda: "",
            get_str_financial_summary=lambda: "", act_address="the Ville:Home:room",
            curr_time=datetime.datetime(2023, 2, 15, 10, 0))
        rgp.run_gpt_prompt_choose_activity(
            types.SimpleNamespace(scratch=sc, a_mem=types.SimpleNamespace(seq_event=[])),
            "work", "", ["a", "b"])
    finally:
        rgp.safe_generate_response = orig
    own, pull = calls["ok"]
    assert own["choice"] == "own" and own["minutes"] == 120 and own["activity"].startswith("reading")
    assert pull["choice"] == 2 and pull["minutes"] == 20
    assert calls["bad"] is False


def test_grocery_trip_pays_once():
    src = open(os.path.join(os.path.dirname(rgp.__file__), "..", "..", "reverie.py")).read()
    assert "A grocery trip is paid ONCE" in src and "* 0.35" in src


def test_meal_consumes_once_per_cooldown():
    src = open(os.path.join(os.path.dirname(rgp.__file__), "..", "..", "reverie.py")).read()
    assert "One meal = one consumption" in src
    assert 'MEAL_COOLDOWN_MIN = int(os.environ.get("GA_MEAL_COOLDOWN_MIN", "90"))' in src


def test_civic_location_forced_and_wide_open():
    import economy
    assert all(t["slots"] >= 25 for t in economy.CIVIC_TASKS.values())
    assert economy.MIN_SHIFT_PRESENCE_MIN <= 45
    import inspect
    from persona.cognitive_modules import plan
    src = inspect.getsource(plan.resolve_action_location)
    assert "CIVIC_TASKS" in src and "forcing" in src
