import os, sys, types, datetime, tempfile
os.environ["GA_DUTIES"] = "1"
os.environ["GA_COMMUNITY"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
import importlib
import duties as D
importlib.reload(D)
import economy as E
import duty_world as W

import pytest


@pytest.fixture(autouse=True)
def _duties_on(monkeypatch):
    monkeypatch.setattr(D, "ENABLED", True)


T0 = datetime.datetime(2023, 2, 14, 10, 0, 0)


def t(m):
    return T0 + datetime.timedelta(minutes=m)


class Scratch:
    def __init__(self):
        self.wallet = 50.0
        self.chatting_with = None
        self.act_description = "idle"
        self.act_address = ""
        self.act_duration = 0
        self.act_start_time = T0
        self.curr_tile = (0, 0)
        self.financial_stress = 0.0
        self.finished = True
        self.started = []
        self.curr_time = T0

    def act_check_finished(self):
        return self.finished

    def add_new_action(self, addr, dur, desc, *a):
        self.started.append((addr, dur, desc))
        self.act_address, self.act_description, self.act_duration = addr, desc, dur
        self.finished = False


class Maze:
    def __init__(self):
        self.address_tiles = {"the Ville:Hobbs Cafe:cafe:counter": {(5, 5)},
                              "the Ville:Hobbs Cafe:cafe:cafe customer seating": {(6, 5)}}
        self.pos = {(5, 5): ("cafe", "counter"), (6, 5): ("cafe", "cafe customer seating"), (0, 0): ("street", "")}

    def access_tile(self, tile):
        a, o = self.pos.get(tuple(tile), ("", ""))
        return {"arena": a, "game_object": o}


def make():
    econ = E.EconomyManager(tempfile.mkdtemp())
    ps = {n: types.SimpleNamespace(name=n, scratch=Scratch()) for n in ("Isabella Rodriguez", "Ana")}
    econ.attach(personas=ps)
    return econ, ps, Maze()


def test_owner_serves_a_sale_at_the_counter_and_is_paid_from_till():
    econ, ps, maze = make()
    econ.credit_sale("cafe", "pastries", 1, "Ana", 3.0)
    till0 = econ.state["tills"]["cafe"]
    owner = ps["Isabella Rodriguez"].scratch
    W.step(econ, ps, maze, t(0), None)
    assert owner.started and "serving a customer" in owner.started[0][2]
    owner.curr_tile = (5, 5)                       # walks to the counter
    done = []
    for m in range(1, 8):
        done += W.step(econ, ps, maze, t(m), None)
    assert [c["ok"] for c in done] == [True]
    assert econ.state["tills"]["cafe"] < till0           # wage came out of the till
    assert any(r["kind"] == "duty" for r in econ.state["pay_log"])
    assert owner.wallet > 50.0


def test_standing_elsewhere_earns_nothing():
    econ, ps, maze = make()
    econ.credit_sale("cafe", "pastries", 1, "Ana", 3.0)
    owner = ps["Isabella Rodriguez"].scratch
    W.step(econ, ps, maze, t(0), None)
    done = []
    for m in range(1, 15):
        done += W.step(econ, ps, maze, t(m), None)   # never leaves (0,0)
    assert done == [] and owner.wallet == 50.0


def test_no_presence_wage_for_employees_under_duties():
    econ, ps, maze = make()
    econ.state["employees"]["cafe"] = ["Ana"]
    ps["Ana"].scratch.job = {"venue": "cafe"}
    econ.state["daily_intake"]["cafe"] = 200.0
    econ._presence = {"Ana": 120.0}
    econ.pay_employees(datetime.datetime(2023, 2, 14, 22, 5))
    assert not any(r["kind"] == "wage" for r in econ.state["pay_log"])


def test_backlog_makes_the_owner_post_an_opening_without_llm():
    econ, ps, maze = make()
    for _ in range(12):
        econ.credit_sale("cafe", "pastries", 1, "Ana", 3.0)
    owner = ps["Isabella Rodriguez"].scratch
    owner.finished = False                         # busy: cannot work the backlog off
    W.step(econ, ps, maze, t(0), None)
    econ.consider_hiring(datetime.datetime(2023, 2, 15, 7, 5))
    assert "cafe" in econ.state["postings"]
