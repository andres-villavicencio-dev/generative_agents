import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from persona.prompt_template.run_gpt_prompt import parse_commitments, pin_commitment

def tot(s): return sum(d for _, d in s)

BASE = [["sleep", 420], ["work", 300], ["lunch", 60], ["read", 300], ["dinner", 60], ["sleep", 300]]

def test_pin_preserves_total_and_position():
  out = pin_commitment(BASE, 17*60, 120, "attend party at Hobbs Cafe")
  assert tot(out) == 1440
  t = 0
  for task, d in out:
    if task.startswith("attend"): assert t == 1020 and d == 120
    t += d

def test_pin_splits_item():
  out = pin_commitment([["read", 600]], 120, 60, "x")
  assert out == [["read", 120], ["x", 60], ["read", 420]]

def test_pin_spanning_multiple_items():
  out = pin_commitment(BASE, 11*60+30, 120, "x")  # overlaps work and lunch
  assert tot(out) == 1440 and [d for t, d in out if t == "x"] == [120]

def test_pin_near_end_clamped():
  out = pin_commitment(BASE, 1438, 60, "x")
  assert tot(out) == 1440

def test_parse_variants():
  r = parse_commitments("- 17:00 | 120 | attend the party at Hobbs Cafe\n**5:30 PM** | 30 min | drop by (the cafe)")
  assert r[0] == (1020, 120, "attend the party at Hobbs Cafe")
  assert r[1][0] == 17*60+30 and r[1][1] == 30 and "(" not in r[1][2]

def test_parse_none_and_garbage():
  assert parse_commitments("NONE") == []
  assert parse_commitments("I think they agreed to come later.") == []

from persona.prompt_template.run_gpt_prompt import has_time_cue

def test_time_cue_gate():
  assert has_time_cue("I'll definitely stop by between five and seven")
  assert has_time_cue("see you at 5:30 PM")
  assert has_time_cue("Let's meet tonight")
  assert has_time_cue("at 3 then")
  assert not has_time_cue("Nice weather. I'll stop by later, and bring five apples.")
  assert not has_time_cue("Let's explore the water path together.")

from persona.prompt_template.run_gpt_prompt import hour_in_text

def test_hour_in_text():
  t = "I'll stop by between five and seven"
  assert hour_in_text(17*60, t) and hour_in_text(5*60, t)
  assert not hour_in_text(10*60+30, t)
  assert hour_in_text(17*60+45, "tea at 5:45 p.m.")
  assert hour_in_text(12*60, "let's meet at noon at 12") 
  assert not hour_in_text(17*60, "coffee later, maybe at ten")

from persona.prompt_template.run_gpt_prompt import accept_commitment

def test_accept_commitment():
  t = "I'll stop by between five and seven"
  assert accept_commitment(5*60, 9*60+12, t) == 17*60      # 05:00 already past -> PM
  assert accept_commitment(17*60, 9*60+12, t) == 17*60
  assert accept_commitment(17*60, 17*60+30, t) is None     # already past
  assert accept_commitment(10*60+30, 10*60+47, "meet at 10:30") is None  # past, not an early-AM slip
  assert accept_commitment(20*60, 9*60, t) is None         # hour never mentioned
