"""GA_COMMUNITY: tax -> fund -> community work + dividend.

Run: venv/bin/python -m pytest tests/test_community.py -q
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

PARK = "the Ville:Johnson Park:park:bench"
NAMES = ("Arthur Burton", "Isabella Rodriguez", "Klaus Mueller", "Maria Lopez")


def _econ(tmp=None):
    town = {}
    for n in NAMES:
        sc = types.SimpleNamespace(name=n, wallet=10.0, financial_stress=0.0,
                                   learned="", job=None, act_address="",
                                   needs={"hunger": 20})
        town[n] = types.SimpleNamespace(name=n, scratch=sc)
    e = EconomyManager(tmp or tempfile.mkdtemp())
    e.attach(llm=lambda p: "NO", personas=town, bulletin=lambda t: None)
    e.state["fund"]["seeded"] = True          # tests start from an empty fund
    return e, town


def _work(e, who, address, minutes, hour=10):
    t = datetime.datetime(2023, 2, 15, hour, 0, 0)
    e.record_presence(t)
    for w in who:
        w.scratch.act_address = address
    for _ in range(minutes):
        t += datetime.timedelta(minutes=1)
        e.record_presence(t)


def test_off_when_disabled(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "0")
    e, _ = _econ()
    e.credit_sale("cafe", "coffee", 1, "Klaus Mueller", 100.0)
    assert e.state["fund"]["balance"] == 0.0


def test_sales_tax_goes_to_fund_and_till_gets_net(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, _ = _econ()
    till = e.state["tills"]["cafe"]
    e.credit_sale("cafe", "coffee", 1, "Klaus Mueller", 100.0)
    tax = 100.0 * economy.FUND_SALES_TAX
    assert e.state["fund"]["balance"] == tax
    assert e.state["tills"]["cafe"] == till + 100.0 - tax
    assert e.state["daily_intake"]["cafe"] == 100.0


def test_draw_tax_and_conservation(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, town = _econ()
    e.state["tills"]["cafe"] = 500.0
    before = (town["Isabella Rodriguez"].scratch.wallet
              + e.state["tills"]["cafe"] + e.state["fund"]["balance"])
    e._owner_draws(datetime.datetime(2023, 2, 15, 22, 0, 0))
    after = (town["Isabella Rodriguez"].scratch.wallet
             + e.state["tills"]["cafe"] + e.state["fund"]["balance"])
    assert e.state["fund"]["balance"] > 0
    assert abs(before - after) < 0.02


def test_fund_persists(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    tmp = tempfile.mkdtemp()
    e, _ = _econ(tmp)
    e.credit_sale("cafe", "coffee", 1, "Klaus Mueller", 100.0)
    e.save()
    assert EconomyManager(tmp).state["fund"]["balance"] == 100.0 * economy.FUND_SALES_TAX


def test_community_pay_needs_presence_and_respects_fund(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, town = _econ()
    day = datetime.datetime(2023, 2, 15, 9, 0, 0)
    assert e.pick_civic_task("Klaus Mueller", day) is not None
    e.state["civic"]["Maria Lopez"] = "park"
    e.state["fund"]["balance"] = 100.0
    e.state["civic"]["Klaus Mueller"] = "park"
    _work(e, [town["Klaus Mueller"]], PARK, 90)      # Maria never shows up
    e.pay_community(datetime.datetime(2023, 2, 15, 22, 0, 0))
    assert town["Klaus Mueller"].scratch.wallet > 10.0
    assert town["Maria Lopez"].scratch.wallet == 10.0
    assert e.state["fund"]["balance"] < 100.0


def test_short_fund_pays_pro_rata_never_overdraws(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, town = _econ()
    e.state["civic"]["Klaus Mueller"] = "park"
    e.state["fund"]["balance"] = 3.0
    _work(e, [town["Klaus Mueller"]], PARK, 120)
    e.pay_community(datetime.datetime(2023, 2, 15, 22, 0, 0))
    assert e.state["fund"]["balance"] == 0.0
    assert town["Klaus Mueller"].scratch.wallet == 13.0


def test_dividend_capped_and_never_exceeds_fund(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, town = _econ()
    e.state["fund"]["balance"] = 1000.0
    each = e.pay_dividend(datetime.datetime(2023, 2, 20, 6, 0, 0))  # Monday
    assert each == economy.FUND_DIVIDEND_CAP
    assert e.state["fund"]["balance"] == 1000.0 - each * len(NAMES)
    assert e.pay_dividend(datetime.datetime(2023, 2, 20, 7, 0, 0)) == 0.0


def test_new_fund_is_seeded_once(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, _ = _econ()
    e.state["fund"].update(seeded=False, balance=0.0, **{"in": 0.0})
    assert e._fund()["balance"] == economy.FUND_SEED
    assert e._fund()["balance"] == economy.FUND_SEED


def test_owners_and_employees_get_no_civic_task(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, _ = _econ()
    e.state["employees"]["cafe"] = ["Klaus Mueller"]
    t = datetime.datetime(2023, 2, 15, 9, 0, 0)
    assert e.pick_civic_task("Isabella Rodriguez", t) is None   # cafe owner
    assert e.pick_civic_task("Klaus Mueller", t) is None        # employee
    assert e.pick_civic_task("Maria Lopez", t) is not None


def test_tax_on_tiny_sales_is_not_rounded_away(monkeypatch):
    monkeypatch.setenv("GA_COMMUNITY", "1")
    e, _ = _econ()
    till = e.state["tills"]["cafe"]
    for _ in range(200):                      # 200 x $0.10 sales = $20; 5% = half a cent each
        e.credit_sale("cafe", "coffee", 1, "Klaus Mueller", 0.10)
    assert e.state["fund"]["balance"] == 1.0  # 5% of $20, carried across sales
    assert abs(e.state["tills"]["cafe"] - (till + 19.0)) < 0.011
