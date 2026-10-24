"""Chat -> replan -> reflection fixes (GA_CHAT_REPLAN_LEGACY switches back).

Run: venv/bin/python -m pytest tests/test_chat_replan_fix.py -q
"""
import datetime
import os
import sys
import types

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server")
sys.path.insert(0, BACKEND)

from persona.prompt_template import run_gpt_prompt as rgp  # noqa: E402
from persona.cognitive_modules import reflect  # noqa: E402
from persona.memory_structures.scratch import Scratch  # noqa: E402


# --- Fix 1: window keeps its length -----------------------------------------

def _window(n_items=23, each=10):
    return [[f"task {i}", each] for i in range(n_items)]


def test_old_clamp_shrinks_window_to_chat_length():
    sched = _window()                      # 230 min
    out = rgp.clamp_subtask_durations(sched, parent_dur=8)
    assert sum(d for _, d in out) < 230 / 1.5          # the bug


@pytest.mark.parametrize("span", [120, 230, 300, 305])
def test_fit_preserves_span(span):
    out = rgp.fit_schedule_to_span(_window(), span)
    assert sum(d for _, d in out) == span
    assert all(d >= 5 and d % 5 == 0 for _, d in out)
    assert len(out) == 23


@pytest.mark.parametrize("n,span", [(23, 60), (12, 50), (8, 33), (3, 7), (5, 5), (4, 3)])
def test_fit_never_overshoots_small_span(n, span):
    out = rgp.fit_schedule_to_span(_window(n, 10), span)
    assert sum(d for _, d in out) == span
    assert all(d >= min(5, span) for _, d in out)


def test_fit_noop_when_already_exact():
    sched = _window(4, 15)
    assert rgp.fit_schedule_to_span(sched, 60) == sched


# --- Fix 3: conversation reflection fires once, however the clock moves -----

class _Persona:
    def __init__(self):
        self.scratch = Scratch.__new__(Scratch)
        s = self.scratch
        s.chatting_with = "Isabella Rodriguez"
        s.chat = [["Tom Moreno", "I'll come at five."], ["Isabella Rodriguez", "Great."]]
        s.chatting_end_time = datetime.datetime(2023, 2, 14, 8, 6, 0)
        s.curr_time = datetime.datetime(2023, 2, 14, 8, 0, 0)
        s.chatting_with_buffer = {}
        s.act_address = None


@pytest.fixture
def fired(monkeypatch):
    calls = []
    monkeypatch.setattr(reflect, "reflection_trigger", lambda p: False)
    monkeypatch.setattr(reflect, "_reflect_on_convo",
                        lambda p, w, c: calls.append((w, len(c))))
    monkeypatch.delenv("GA_CHAT_REPLAN_LEGACY", raising=False)
    return calls


def test_exact_tick_fires_once(fired):
    p = _Persona()
    p.scratch.curr_time = datetime.datetime(2023, 2, 14, 8, 5, 50)
    reflect.reflect(p)
    reflect.reflect(p)                     # same chat, second step: no repeat
    assert fired == [("Isabella Rodriguez", 2)]


def test_skipped_tick_still_fires(fired):
    p = _Persona()
    p.scratch.curr_time = datetime.datetime(2023, 2, 14, 8, 6, 40)  # jumped over
    reflect.reflect(p)
    assert len(fired) == 1


def test_chat_cleared_before_reflect_is_stashed(fired):
    p = _Persona()
    p.scratch.curr_time = datetime.datetime(2023, 2, 14, 8, 7, 0)
    p.scratch.stash_unreflected_chat()     # what add_new_action / plan() do
    p.scratch.chatting_with = p.scratch.chat = p.scratch.chatting_end_time = None
    reflect.reflect(p)
    reflect.reflect(p)
    assert fired == [("Isabella Rodriguez", 2)]


def test_legacy_misses_skipped_tick(fired, monkeypatch):
    monkeypatch.setenv("GA_CHAT_REPLAN_LEGACY", "1")
    p = _Persona()
    p.scratch.curr_time = datetime.datetime(2023, 2, 14, 8, 6, 40)
    reflect.reflect(p)
    assert fired == []                      # the bug
