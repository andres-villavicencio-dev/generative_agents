import datetime, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
from persona.memory_structures import associative_memory as am

D = datetime.datetime
PAIR = lambda txt: (txt, [0.0] * 8)  # embedding pair placeholder


def mem():
  """Bare AssociativeMemory with just what add_thought and the gate touch."""
  m = object.__new__(am.AssociativeMemory)
  m.id_to_node = {}
  m.seq_thought = []
  m.kw_to_thought = {}
  m.kw_strength_thought = {}
  m.embeddings = {}
  return m


def add(m, txt, t, s="Wolfgang Schulz"):
  return m.add_thought(t, t + datetime.timedelta(days=30), s, "is", "idle",
                       txt, {"a"}, 5, PAIR(txt), None)


def test_identical_recent_thought_is_suppressed(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  assert add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 8)) is not None
  assert add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 12)) is None
  assert len(m.seq_thought) == 1


def test_cosmetic_rewrite_is_suppressed(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  t1 = ("Tamara Taylor is a flexible, multi-disciplinary professional who "
        "successfully balances a demanding Hobbs Cafe shift with household, "
        "creative, and family responsibilities")
  t2 = ("Tamara is a flexible multi-disciplinary professional who successfully "
        "balances a demanding Hobbs Cafe shift with household and creative "
        "responsibilities")
  assert add(m, t1, D(2023, 2, 13, 9), s="Tamara Taylor") is not None
  assert add(m, t2, D(2023, 2, 13, 15), s="Tamara Taylor") is None
  assert len(m.seq_thought) == 1


def test_distinct_thoughts_are_kept(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  assert add(m, "Wolfgang Schulz is studying chemistry for the exams", D(2023, 2, 13, 8)) is not None
  assert add(m, "Wolfgang Schulz is cooking oatmeal with fresh berries", D(2023, 2, 13, 9)) is not None
  assert add(m, "Wolfgang Schulz went for a morning run at the park", D(2023, 2, 13, 10)) is not None
  assert len(m.seq_thought) == 3


def test_lookback_expires(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  assert add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 8)) is not None
  # 25h later: outside the 24h window -> allowed again
  assert add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 14, 9)) is not None
  assert len(m.seq_thought) == 2


def test_plan_marker_is_exempt_and_ignored_as_target(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  p1 = "This is Wolfgang Schulz's plan for Monday February 13: wake up, eat, work."
  p2 = "This is Wolfgang Schulz's plan for Tuesday February 14: wake up, eat, work."
  assert add(m, p1, D(2023, 2, 13, 7)) is not None
  assert add(m, p2, D(2023, 2, 14, 7)) is not None   # near-identical but a marker
  # a normal thought similar to a stored plan marker is NOT compared against it
  assert add(m, "Wolfgang Schulz's plan for Tuesday involves waking and eating",
             D(2023, 2, 14, 8)) is not None


def test_flag_off_bypasses_the_gate(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  monkeypatch.setenv("GA_THOUGHT_DEDUP", "0")
  m = mem()
  assert add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 8)) is not None
  assert add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 12)) is not None
  assert len(m.seq_thought) == 2


def test_suppression_writes_trace(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 8))
  add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 12))
  f = tmp_path / "thought_dedup.jsonl"
  assert f.exists()
  assert "Wolfgang" in f.read_text()


def test_typo_here_is_here_is_gone():
  src = open(os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server",
                          "persona", "prompt_template", "run_gpt_prompt.py")).read()
  assert "Here is Here is" not in src
