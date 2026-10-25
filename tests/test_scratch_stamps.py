"""Cooldown stamps survive Scratch.save/load (resume no longer resets them).

Run: venv/bin/python -m pytest tests/test_scratch_stamps.py -q
"""
import datetime
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server")
sys.path.insert(0, BACKEND)

from persona.memory_structures.scratch import Scratch  # noqa: E402

BASE = os.path.join(
    os.path.dirname(__file__), "..", "environment", "frontend_server", "storage",
    "base_the_ville_n25", "personas", "Abigail Chen", "bootstrap_memory", "scratch.json")


def test_stamps_round_trip(tmp_path):
    s = Scratch(BASE)
    s.curr_time = s.act_start_time = datetime.datetime(2023, 2, 14, 9, 30, 0)
    t = datetime.datetime(2023, 2, 14, 9, 30, 0)
    s.last_grocery_attempt = t
    s.last_grocery_shop = t + datetime.timedelta(hours=1)
    s.last_luxury_attempt = t + datetime.timedelta(days=1)
    s.last_status_thought_day = t.date()
    out = str(tmp_path / "scratch.json")
    s.save(out)

    s2 = Scratch(out)
    assert s2.last_grocery_attempt == t
    assert s2.last_grocery_shop == t + datetime.timedelta(hours=1)
    assert s2.last_luxury_attempt == t + datetime.timedelta(days=1)
    assert s2.last_status_thought_day == t.date()


def test_missing_stamps_stay_unset(tmp_path):
    s = Scratch(BASE)
    s.curr_time = s.act_start_time = datetime.datetime(2023, 2, 14, 9, 30, 0)
    out = str(tmp_path / "scratch.json")
    s.save(out)
    s2 = Scratch(out)
    assert getattr(s2, "last_grocery_attempt", None) is None
