"""Tests for the deterministic event-triple parser (no LLM)."""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server",
                     "persona", "cognitive_modules", "event_triple_parser.py")
_spec = importlib.util.spec_from_file_location("event_triple_parser", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse = _mod.parse_event_triple

SAMPLES = [
  "John Lin is wakes up, gets out of bed, and opens the curtains.",
  "Ryan Park is identify the materials and resources needed.",
  "Adam Smith is Consider students’ interests, needs, and prior learning.",
  "Sam Moore is Creativity can be cultivated through diverse experiences and perspectives.",
  "Tamara Taylor is The closet currently contains no items.",
  "Tamara Taylor is brushes her teeth.",
  "Adam Smith is exploring a part of the ville they rarely visit.",
  "John Lin is conversing about Tamara Taylor and John Lin discussing his work.",
  "Francisco Lopez is gently while recharging.",
]


def test_no_sample_returns_none():
  for s in SAMPLES:
    r = parse(s, "X")
    assert r is not None and len(r) == 3 and all(isinstance(p, str) for p in r), s


def test_existing_patterns_unchanged():
  assert parse("Isabella is cooking breakfast.", "Isabella") == ("Isabella", "cook", "breakfast")
  assert parse("Jane is sleeping.", "Jane") == ("Jane", "is", "sleeping")
  assert parse("Adam Smith is exploring a part of the ville they rarely visit.", "A") == \
    ("Adam Smith", "explore", "a part of the ville they rarely visit")
  assert parse("Tom is happy.", "Tom") == ("Tom", "is", "happy")


def test_third_person_and_imperative():
  assert parse("Tamara Taylor is brushes her teeth.", "T") == ("Tamara Taylor", "brush", "her teeth")
  assert parse("John Lin is wakes up, gets out of bed, and opens the curtains.", "J") == \
    ("John Lin", "wake", "up")
  assert parse("Ryan Park is identify the materials and resources needed.", "R") == \
    ("Ryan Park", "identify", "the materials")


def test_fallbacks():
  assert parse("Tamara Taylor is The closet currently contains no items.", "T")[:2] == \
    ("Tamara Taylor", "is")
  assert parse("Francisco Lopez is gently while recharging.", "F")[:2] == ("Francisco Lopez", "is")
  assert parse("Sam Moore is Creativity can be cultivated through diverse experiences.", "S")[1] == "is"


def test_none_only_without_is():
  assert parse("no verb here", "x") is None
  assert parse("Bob is", "Bob") == ("Bob", "is", "idle")
  r = parse("Eddy is working (at the desk)", "Eddy")
  assert r is not None and r[0] == "Eddy"


def test_object_truncated():
  r = parse("A is " + "word " * 40, "A")
  assert len(r[2]) <= 60
