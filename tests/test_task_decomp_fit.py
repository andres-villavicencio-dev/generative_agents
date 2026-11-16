"""Regression: task decomposition must never come back empty.

n76, Giorgio Rossi: an 84-minute block (an agenda event pinned "from now") was
decomposed into ONE 85-minute step. Nothing fitted, the list stayed empty and
`fin_output[-1]` raised IndexError on 66 consecutive ticks (agent frozen).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from persona.prompt_template.run_gpt_prompt import fit_decomp_to_duration as fit


def total(rows):
  return sum(d for _, d in rows)


def test_single_step_longer_than_block_is_clipped():
  out = fit([["checking the party schedule and confirming the guest list", 85]], 84, "party")
  assert out == [["checking the party schedule and confirming the guest list", 84]]


def test_first_step_too_long_with_more_steps_after():
  out = fit([["set up", 90], ["greet guests", 10]], 60, "party")
  assert out == [["set up", 60]]


def test_steps_that_fit_are_kept_and_last_takes_the_remainder():
  out = fit([["a", 20], ["b", 20], ["c", 30]], 60, "t")
  assert out == [["a", 20], ["b", 40]]
  out = fit([["a", 10], ["b", 10]], 60, "t")
  assert out == [["a", 10], ["b", 50]] and total(out) == 60


def test_exact_fit_is_unchanged():
  assert fit([["a", 30], ["b", 30]], 60, "t") == [["a", 30], ["b", 30]]


def test_unusable_output_falls_back_to_the_task_itself():
  for bad in ([], None, [["a", "soon"]], ["just text"], [["a", 0], ["b", -5]], [["a", 1, 2]]):
    assert fit(bad, 45, "read a book") == [["read a book", 45]]


def test_bad_rows_are_skipped_not_fatal():
  out = fit([["a", "x"], ["b", "15"], "junk", ["c", 15]], 40, "t")
  assert out == [["b", 15], ["c", 25]]
