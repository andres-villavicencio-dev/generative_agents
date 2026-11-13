"""Regression: a planned timed event must survive daily-schedule compilation.

John Lin (n73) planned `attend the Isabella Rodriguez Valentine's Day party at
Hobbs Cafe from 5:00 to 7:00 pm`, but the compiled hourly schedule held
"have dinner" at 17:00 and "explore the kitchen" at 18:00: point anchors (one
explicit, one inferred from the previous item) overwrote the event's range.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from persona.cognitive_modules.plan import build_schedule_from_req

JOHN = [
  "wake up and complete the morning routine at 6:00 am",
  "eat breakfast at 7:00 am",
  "open the Willow Market and Pharmacy counter at 9:00 am and serve customers",
  "have lunch at 12:00 pm",
  "continue helping customers, comparing prices, and researching mayoral candidates from 1:00 to 5:00 pm",
  "have dinner at 5:00 pm",
  "attend the Isabella Rodriguez Valentine’s Day party at Hobbs Cafe from 5:00 to 7:00 pm",
  "have dinner around 5:30 pm and go to bed around 10:00 pm",
  "explore the kitchen in Willow Court 2 for the first time",
]


def test_inferred_followon_does_not_erase_event_range():
  # the untimed "explore the kitchen" used to land on 18:00 and wipe the party
  s = build_schedule_from_req(JOHN, 6)
  assert "party" in s[18].lower(), s[16:21]
  assert "explore" not in s[18].lower()


def test_work_block_still_yields_to_lunch_anchor():
  s = build_schedule_from_req(
    ["work at the store from 9:00 am to 5:00 pm", "have lunch at 12:00 pm"], 7)
  assert "lunch" in s[12].lower() and "work" in s[11].lower() and "work" in s[13].lower()


def test_untimed_item_goes_to_next_free_hour():
  s = build_schedule_from_req(
    ["attend the book club at the library from 5:00 to 7:00 pm", "tidy the garden shed"], 7)
  assert "book club" in s[17].lower() and "book club" in s[18].lower()
  assert any("garden" in x.lower() for x in s[19:22]), s[17:23]
