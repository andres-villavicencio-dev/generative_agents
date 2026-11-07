import os, sys, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
import duties as D

T0 = datetime.datetime(2023, 2, 14, 10, 0, 0)


def t(m):
    return T0 + datetime.timedelta(minutes=m)


class FakeStock:
    def __init__(self, counter=2, back=30, mx=20):
        self.counter, self.back, self.mx = counter, back, mx

    def counter_low(self, vid):
        return vid == "cafe" and self.counter < self.mx * D.COUNTER_LOW_FRACTION

    def backroom_has(self, vid):
        return vid == "cafe" and self.back > 0

    def refill(self, vid):
        n = min(self.back, self.mx - self.counter)
        self.counter += n
        self.back -= n
        return n


def work(state, name, arena, obj, start, minutes, stock=None):
    out = None
    for m in range(0, minutes + 1):
        r = D.tick_worker(state, name, arena, obj, t(start + m), stock)
        out = out or r
    return out


def test_no_work_no_duties():
    s = D.new_state()
    assert D.generate(s, FakeStock(counter=20)) == []
    assert D.open_for(s) == []


def test_visits_dirty_tables_and_spawn_cleaning():
    s = D.new_state()
    for _ in range(3):
        D.note_visit(s, "cafe")
    ids = D.generate(s)
    assert len(ids) == 1 and s["open"][ids[0]]["type"] == "cafe.clean_table"
    # generating again does not duplicate
    assert D.generate(s) == []


def test_sale_queues_order_and_is_served_and_paid():
    s = D.new_state()
    D.note_sale(s, "cafe")
    (did,) = D.generate(s)
    assert D.claim(s, "Ana", did, t(0))
    # standing at the wrong place earns nothing
    assert work(s, "Ana", "kitchen", "sink", 0, 10) is None
    r = work(s, "Ana", "cafe", "counter", 12, 5)
    assert r and r["ok"] and r["pay"] == 0.60 and r["worker"] == "Ana"
    assert s["venues"]["cafe"]["orders"] == 0.0 and s["done"] == 1


def test_presence_alone_does_not_pay():
    s = D.new_state()
    D.note_sale(s, "cafe")
    (did,) = D.generate(s)
    D.claim(s, "Ana", did, t(0))
    # present at the right object but for less than the required minutes
    assert work(s, "Ana", "cafe", "counter", 0, 1) is None
    assert did in s["open"]


def test_leaving_resets_the_work_timer():
    s = D.new_state()
    D.note_visit(s, "cafe"); D.note_visit(s, "cafe"); D.note_visit(s, "cafe")
    (did,) = D.generate(s)
    D.claim(s, "Ana", did, t(0))
    D.tick_worker(s, "Ana", "cafe", "table", t(0))
    D.tick_worker(s, "Ana", "cafe", "table", t(2))      # 2 min on site, needs 3
    for m in range(3, 12):
        D.tick_worker(s, "Ana", "street", "", t(m))     # wandered off
    assert s["open"][did]["on_site_min"] == 0.0


def test_restock_moves_stock_and_pays_per_unit():
    s = D.new_state()
    st = FakeStock(counter=2, back=30, mx=20)
    (did,) = D.generate(s, st)
    assert s["open"][did]["type"] == "cafe.refill_counter"
    D.claim(s, "Bo", did, t(0))
    r = work(s, "Bo", "cafe", "counter", 0, 6, st)
    assert r["ok"] and r["units"] == 18 and st.counter == 20 and st.back == 12
    assert abs(r["pay"] - (0.40 + 0.04 * 18)) < 1e-6


def test_restock_without_back_room_is_not_generated():
    s = D.new_state()
    assert D.generate(s, FakeStock(counter=0, back=0)) == []


def test_two_workers_cannot_claim_same_duty_or_hold_two():
    s = D.new_state()
    D.note_sale(s, "cafe"); D.note_sale(s, "cafe")
    a, b = D.generate(s)
    assert D.claim(s, "Ana", a, t(0))
    assert D.claim(s, "Bo", a, t(0)) is None
    assert D.claim(s, "Ana", b, t(0)) is None    # one duty at a time


def test_stale_claim_is_released():
    s = D.new_state()
    D.note_sale(s, "cafe")
    (did,) = D.generate(s)
    D.claim(s, "Ana", did, t(0))
    assert D.release_stale(s, t(D.CLAIM_TTL_MIN + 5)) == 1
    assert s["open"][did]["claimed_by"] is None


def test_background_work_accumulates_for_civic_posts():
    s = D.new_state()
    D.accumulate(s, t(0))
    D.accumulate(s, t(60))
    D.accumulate(s, t(120))
    types = {s["open"][i]["type"] for i in D.generate(s)}
    assert "park.pick_litter" in types


def test_backlog_value_for_hiring():
    s = D.new_state()
    assert D.demand_value(s, "cafe") == 0.0
    for _ in range(4):
        D.note_sale(s, "cafe")
    D.generate(s)
    assert D.demand_value(s, "cafe") > 0


def test_clinic_patient_creates_treatment_duty():
    s = D.new_state()
    assert D.generate(s) == []
    D.note_sale(s, "clinic")
    ids = D.generate(s)
    assert [s["open"][i]["type"] for i in ids] == ["clinic.treat_patient"]
