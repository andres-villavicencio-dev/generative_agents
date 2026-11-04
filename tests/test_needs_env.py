import os, sys, datetime, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
os.environ["GA_NEEDS_LOG"] = "0"
import needs_env as N
import persona.prompt_template.gpt_structure as _gs
_gs.get_embedding = lambda *a, **k: [0.0] * 768   # never hit the live Ollama queue from tests

NOW = datetime.datetime(2023, 2, 13, 12, 0)

def mk(needs=None, obj="", arena="", addr="the Ville:Hobbs Cafe:cafe:counter"):
    sc = types.SimpleNamespace(needs=dict(needs or {"hunger": 30, "hydration": 30, "hygiene": 30, "bladder": 30,
                                                    "comfort": 30, "energy": 30, "stimulation": 30, "social": 30}),
                               act_address=addr, curr_tile=(0, 0), chatting_with=None)
    p = types.SimpleNamespace(name="Test Person", scratch=sc)
    maze = types.SimpleNamespace(access_tile=lambda t: {"sector": "x", "arena": arena, "game_object": obj})
    return p, maze

def test_keyword_intent_word_boundary():
    assert N.keyword_intent("eating breakfast") == "eat"
    assert N.keyword_intent("visiting the park") == "none"          # 'sit' inside 'visit'
    assert N.keyword_intent("restocking the shelves") == "none"      # 'rest' inside 'restock'
    assert N.keyword_intent("buying bread") == "none"                # 'read' inside 'bread'
    assert N.keyword_intent("taking a shower") == "wash"
    assert N.keyword_intent("using the bathroom") == "toilet"
    assert N.keyword_intent("painting a landscape") == "engage"

def test_no_food_no_relief():
    p, m = mk()
    out = N.apply_relief(p, m, "eating breakfast", NOW)
    assert out == {} and p.scratch.needs["hunger"] == 30

def test_real_food_relieves_once_per_ledger():
    p, m = mk()
    N.note_consumed(p, "eggs")
    out = N.apply_relief(p, m, "eating breakfast", NOW)
    assert out["hunger"] == 50 and p.scratch.needs["hunger"] == 80
    out2 = N.apply_relief(p, m, "eating breakfast: pour milk", NOW)   # sub-task, nothing consumed
    assert out2 == {} and p.scratch.needs["hunger"] == 80

def test_toilet_needs_bathroom():
    p, m = mk(arena="kitchen")
    assert N.apply_relief(p, m, "using the bathroom", NOW) == {}
    p, m = mk(arena="common bathroom", obj="toilet")
    assert N.apply_relief(p, m, "using the bathroom", NOW)["bladder"] == 80
    assert N.apply_relief(p, m, "using the bathroom", NOW + datetime.timedelta(minutes=10)) == {}   # cooldown
    assert N.apply_relief(p, m, "using the bathroom", NOW + datetime.timedelta(minutes=50))["bladder"] == 80

def test_shower_needs_hot_water_or_fixture():
    p, m = mk(arena="bathroom", obj="shower")
    assert N.apply_relief(p, m, "taking a shower", NOW)["hygiene"] == 70
    p, m = mk(arena="bedroom", obj="bed")
    assert N.apply_relief(p, m, "taking a shower", NOW) == {}
    p, m = mk(arena="bedroom", obj="bed")
    N.note_consumed(p, "level")
    assert N.apply_relief(p, m, "taking a shower", NOW)["hygiene"] == 70

def test_rest_needs_seat_and_work_needs_object():
    p, m = mk(obj="shelves")
    assert N.apply_relief(p, m, "relaxing", NOW) == {}
    p, m = mk(obj="sofa")
    assert "comfort" in N.apply_relief(p, m, "relaxing on the sofa", NOW)
    p, m = mk(obj="computer desk")
    assert "stimulation" in N.apply_relief(p, m, "studying for the exam", NOW)

def test_tick_equilibrium_and_social():
    assert N.tick_value("hunger", 100, False, False, 1.0) < 100
    assert N.tick_value("energy", 50, True, False, 1.0) > 50
    assert N.tick_value("social", 50, False, True, 1.0) > 50
    assert N.tick_value("social", 50, False, False, 1.0) < 50
    # a neglected need reaches the critical threshold (20) within the target window
    v, ticks = 100.0, 0
    while v > 20:
        v = N.tick_value("hunger", v, False, False, 1.0); ticks += 1
    assert 12 <= ticks / 360 <= 20          # hours awake

def test_laya_off_uses_keywords():
    p, m = mk()
    assert N.get_intent(p, "eating lunch", "x")[1] == "keyword"


# ---- need-zero crises ----------------------------------------------------------------------------
def crisis_persona(need, wallet=100.0, sleeping=False):
    p, m = mk()
    sc = p.scratch
    sc.wallet = wallet
    sc.act_description = "sleeping" if sleeping else "working"
    sc.financial_stress = 0.0
    sc.actions = []
    def add_new_action(addr, dur, desc, pron, ev, cw, ch, buf, cet, od, op, oe):
        sc.act_address, sc.act_duration, sc.act_description, sc.act_pronunciatio = addr, dur, desc, pron
        sc.act_path_set = False
    sc.add_new_action = add_new_action
    sc.needs[need] = 0.0
    mem = []
    p.a_mem = types.SimpleNamespace(add_event=lambda *a: mem.append(a))
    p.mem = mem
    m.address_tiles = {N.CLINIC_BED: {(5, 5), (6, 5)}}
    econ = types.SimpleNamespace(state={}, fund={"balance": 0.0, "in": 0.0}, log=[])
    econ._fund = lambda: econ.fund
    econ._log_pay = lambda kind, who, amt, v, note=None: econ.log.append((kind, who, amt))
    return p, m, econ

def test_energy_zero_hospitalises_charges_and_remembers():
    os.environ["GA_NEEDS_V2"] = "1"; N.ENABLED = True
    p, m, econ = crisis_persona("energy", wallet=100)
    fired = N.check_crises(p, m, econ, NOW)
    assert len(fired) == 1 and fired[0]["need"] == "energy"
    s = p.scratch
    assert s.planned_path == [(5, 5)] and s.act_path_set is True            # transported to the clinic bed
    assert s.act_address == N.CLINIC_BED and s.act_duration == 150
    assert s.needs["energy"] == 70
    assert s.wallet == 85.0 and econ.fund["balance"] == 15.0 and econ.log == [("fee", "Test Person", 15.0)]
    assert len(p.mem) == 1 and p.mem[0][7] == 9                              # poignancy 9, unpleasant event
    assert N.check_crises(p, m, econ, NOW + datetime.timedelta(minutes=30)) == []   # no double-billing while treated

def test_fee_never_exceeds_wallet():
    N.ENABLED = True
    p, m, econ = crisis_persona("hunger", wallet=3.0)
    N.check_crises(p, m, econ, NOW)
    assert p.scratch.wallet == 0.0 and econ.fund["balance"] == 3.0

def test_personal_crises_have_no_hospital():
    N.ENABLED = True
    p, m, econ = crisis_persona("bladder")
    N.check_crises(p, m, econ, NOW)
    s = p.scratch
    assert s.needs["bladder"] == 65 and s.needs["hygiene"] == 0.0     # hygiene 30 - 40 clamps at 0
    assert getattr(s, "planned_path", None) is None and s.wallet == 100.0
    assert len(p.mem) == 1 and p.mem[0][7] == 8

def test_sleeping_agent_cannot_hit_energy_crisis_and_chatting_defers():
    N.ENABLED = True
    p, m, econ = crisis_persona("energy", sleeping=True)
    assert N.check_crises(p, m, econ, NOW) == []
    p, m, econ = crisis_persona("social")
    p.scratch.chatting_with = "Someone"
    assert N.check_crises(p, m, econ, NOW) == []

def test_every_need_but_status_has_a_crisis_that_restores():
    assert set(N.CRISIS) == {"energy", "hunger", "hydration", "bladder", "hygiene", "social", "comfort", "stimulation"}
    for need, spec in N.CRISIS.items():
        assert spec["restore"] > 20 and spec["poign"] >= 7


def test_toilet_relief_waits_until_agent_reaches_the_fixture():
    p, m = mk(arena="main room", obj="desk")
    p.scratch.act_description = "using the bathroom at home"
    assert N.apply_relief(p, m, "using the bathroom at home", NOW) == {}
    assert N.recheck(p, m, NOW) == {} and p.scratch.needs["bladder"] == 30      # still at the desk
    m.access_tile = lambda t: {"sector": "x", "arena": "bathroom", "game_object": "toilet"}
    out = N.recheck(p, m, NOW)                                                  # walked to the toilet
    assert out["bladder"] == 80 and p.scratch.needs["bladder"] == 100
    assert N.recheck(p, m, NOW) == {}                                           # only once


def test_recheck_dropped_when_action_changes():
    p, m = mk(arena="main room", obj="desk")
    p.scratch.act_description = "using the bathroom at home"
    N.apply_relief(p, m, "using the bathroom at home", NOW)
    p.scratch.act_description = "reading a book"
    m.access_tile = lambda t: {"sector": "x", "arena": "bathroom", "game_object": "toilet"}
    assert N.recheck(p, m, NOW) == {} and p.scratch.needs["bladder"] == 30


def test_action_already_running_at_resume_is_still_relieved():
    p, m = mk(arena="bathroom", obj="toilet")          # no apply_relief ever ran for this action (sim resumed)
    p.scratch.act_description = "using the bathroom at home"
    assert N.recheck(p, m, NOW)["bladder"] == 80 and p.scratch.needs["bladder"] == 100
    assert N.recheck(p, m, NOW) == {}                  # once per action


def test_toilet_action_wraps_up_after_relief():
    p, m = mk(arena="bathroom", obj="toilet")
    p.scratch.act_description = "using the bathroom at home"
    p.scratch.act_start_time = NOW - datetime.timedelta(minutes=4)
    p.scratch.act_duration = 30
    N.recheck(p, m, NOW)
    assert p.scratch.act_duration == 4 + N.WRAP_UP_MIN


def test_rest_action_is_not_cut_short():
    p, m = mk(arena="room", obj="sofa")
    p.scratch.act_description = "resting on the sofa"
    p.scratch.act_start_time = NOW - datetime.timedelta(minutes=4)
    p.scratch.act_duration = 30
    N.recheck(p, m, NOW)
    assert p.scratch.act_duration == 30


def test_drink_water_at_kitchen_park_and_cafe_but_not_in_bed():
    for arena, obj, ok in [("kitchen", "cooking area", True), ("park", "park garden", True),
                           ("cafe", "behind the cafe counter", True), ("bedroom 1", "bed", False),
                           ("bathroom", "toilet", False), ("bathroom", "bathroom sink", True)]:
        p, m = mk(arena=arena, obj=obj)
        p.scratch.act_description = "drinking a glass of water"
        out = N.apply_relief(p, m, "drinking a glass of water", NOW)
        assert bool(out) == ok, (arena, obj, out)


def test_bottled_water_purchase_counts_and_drink_waits_for_a_tap():
    p, m = mk(arena="store", obj="grocery store shelf")
    N.note_consumed(p, "water")
    assert N.apply_relief(p, m, "buying a bottle of water", NOW)["hydration"] == 30.0   # bought bottled water counts
    p, m = mk(arena="bedroom 1", obj="bed")
    p.scratch.act_description = "drinking a glass of water"
    assert N.apply_relief(p, m, "drinking a glass of water", NOW) == {}
    m.access_tile = lambda t: {"sector": "x", "arena": "kitchen", "game_object": "cooking area"}
    assert N.recheck(p, m, NOW)["hydration"] == 25.0
