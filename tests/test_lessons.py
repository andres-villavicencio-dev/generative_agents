"""Outcome memory (lessons): record / expire / render / persist.

Run: venv/bin/python -m pytest tests/test_lessons.py -q
"""
import datetime
import os
import sys
import types

BACKEND = os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server")
sys.path.insert(0, BACKEND)

from persona.cognitive_modules import lessons  # noqa: E402
from persona.memory_structures.scratch import Scratch  # noqa: E402

T0 = datetime.datetime(2023, 2, 14, 9, 0, 0)
H = datetime.timedelta(hours=1)


def _s():
    return types.SimpleNamespace()


def test_record_is_new_then_repeat():
    s = _s()
    assert lessons.record(s, "home_stock", "bread", "sold_out", T0) is True
    assert lessons.record(s, "home_stock", "bread", "sold_out", T0 + H) is False
    assert lessons.live(s, T0 + 2 * H)[0]["count"] == 2


def test_expiry_and_refresh():
    s = _s()
    lessons.record(s, "store_stock", "milk", "cant_afford", T0)      # 12 h TTL
    assert lessons.live(s, T0 + 11 * H)
    assert not lessons.live(s, T0 + 13 * H)
    assert lessons.record(s, "store_stock", "milk", "cant_afford", T0 + 13 * H) is True


def test_filters_and_clear():
    s = _s()
    lessons.record(s, "home_stock", "eggs", "sold_out", T0)
    lessons.record(s, "store_stock", "milk", "cant_afford", T0)
    assert len(lessons.live(s, T0, kind="home_stock")) == 1
    assert len(lessons.live(s, T0, reason="cant_afford")) == 1
    lessons.clear(s, "home_stock", "eggs")
    assert not lessons.live(s, T0, kind="home_stock")


def test_render_distinguishes_broke_from_sold_out():
    s = _s()
    lessons.record(s, "home_stock", "bread", "sold_out", T0)
    lessons.record(s, "store_stock", "milk", "cant_afford", T0)
    txt = lessons.render(s, T0)
    assert "run out of bread" in txt and "could not afford milk" in txt
    assert lessons.render(_s(), T0) == ""


def test_cap_keeps_book_bounded():
    s = _s()
    for i in range(60):
        lessons.record(s, "store_stock", f"item{i}", "sold_out", T0 + datetime.timedelta(minutes=i))
    assert len(s.lessons) <= 40


def test_persists_through_scratch(tmp_path):
    base = os.path.join(os.path.dirname(__file__), "..", "environment", "frontend_server",
                        "storage", "base_the_ville_n25", "personas", "Abigail Chen",
                        "bootstrap_memory", "scratch.json")
    s = Scratch(base)
    s.curr_time = s.act_start_time = T0
    lessons.record(s, "home_stock", "eggs", "sold_out", T0)
    out = str(tmp_path / "scratch.json")
    s.save(out)
    s2 = Scratch(out)
    assert lessons.live(s2, T0)[0]["target"] == "eggs"
