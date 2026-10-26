"""GA_JOBS: wages need on-site presence; employees get a shift goal.

Run: venv/bin/python -m pytest tests/test_jobs.py -q
"""
import datetime
import os
import sys
import tempfile
import types

BACKEND = os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server")
sys.path.insert(0, BACKEND)

import economy  # noqa: E402
from economy import EconomyManager  # noqa: E402

PUB = "the Ville:The Rose and Crown Pub:pub:shelf"


def _persona(name):
    sc = types.SimpleNamespace(name=name, wallet=100.0, financial_stress=0.0,
                               learned="", job=None, act_address="")
    return types.SimpleNamespace(name=name, scratch=sc)


def _econ():
    town = {n: _persona(n) for n in ("Arthur Burton", "Klaus Mueller", "Maria Lopez")}
    e = EconomyManager(tempfile.mkdtemp())
    e.attach(llm=lambda p: "NO", personas=town, bulletin=lambda t: None)
    e.state["employees"]["pub"] = ["Klaus Mueller", "Maria Lopez"]
    e.state["daily_intake"]["pub"] = 100.0
    return e, town


def _work(e, town, who, minutes):
    t = datetime.datetime(2023, 2, 15, 10, 0, 0)
    e.record_presence(t)
    for who_ in who:
        town[who_].scratch.act_address = PUB
    for _ in range(minutes):
        t += datetime.timedelta(minutes=1)
        e.record_presence(t)


def test_wage_requires_presence(monkeypatch):
    monkeypatch.setenv("GA_JOBS", "1")
    e, town = _econ()
    _work(e, town, ["Klaus Mueller"], 90)        # Maria never shows up
    e.pay_employees(datetime.datetime(2023, 2, 15, 22, 0, 0))
    assert town["Klaus Mueller"].scratch.wallet > 100.0
    assert town["Maria Lopez"].scratch.wallet == 100.0


def test_short_visit_is_not_enough(monkeypatch):
    monkeypatch.setenv("GA_JOBS", "1")
    e, town = _econ()
    _work(e, town, ["Klaus Mueller"], 30)
    e.pay_employees(datetime.datetime(2023, 2, 15, 22, 0, 0))
    assert town["Klaus Mueller"].scratch.wallet == 100.0


def test_presence_outside_shift_does_not_count(monkeypatch):
    monkeypatch.setenv("GA_JOBS", "1")
    e, town = _econ()
    t = datetime.datetime(2023, 2, 15, 16, 0, 0)
    town["Klaus Mueller"].scratch.act_address = PUB
    e.record_presence(t)
    for m in range(120):
        e.record_presence(t + datetime.timedelta(minutes=m + 1))
    e.pay_employees(datetime.datetime(2023, 2, 15, 22, 0, 0))
    assert town["Klaus Mueller"].scratch.wallet == 100.0


def test_flag_off_keeps_legacy_pay(monkeypatch):
    monkeypatch.setenv("GA_JOBS", "0")
    e, town = _econ()
    e.pay_employees(datetime.datetime(2023, 2, 15, 22, 0, 0))
    assert town["Klaus Mueller"].scratch.wallet > 100.0
    assert town["Maria Lopez"].scratch.wallet > 100.0


def test_broke_employee_gets_one_staff_meal_a_day(monkeypatch):
    monkeypatch.setenv("GA_JOBS", "1")
    e, town = _econ()
    k = town["Klaus Mueller"].scratch
    k.wallet, k.needs = 1.4, {"hunger": 20}
    town["Maria Lopez"].scratch.needs = {"hunger": 20}   # rich: wallet 100
    _work(e, town, ["Klaus Mueller", "Maria Lopez"], 60)
    assert k.needs["hunger"] == 75
    assert town["Maria Lopez"].scratch.needs["hunger"] == 20
    k.needs["hunger"] = 10
    _work(e, town, ["Klaus Mueller"], 5)                 # same day: no second meal
    assert k.needs["hunger"] == 10
