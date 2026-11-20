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


# --- fixes after the n76 load check (Nov 20 2026) ---------------------------
def full_mem(tmp_path):
  d = tmp_path / "empty"
  d.mkdir()
  (d / "nodes.json").write_text("{}")
  (d / "embeddings.json").write_text("{}")
  (d / "kw_strength.json").write_text('{"kw_strength_event": {}, "kw_strength_thought": {}}')
  return am.AssociativeMemory(str(d))


def test_loading_a_save_keeps_every_thought_and_id(monkeypatch, tmp_path):
  """The gate ran inside the loader: n76 lost 18% of its thoughts on load and
  78,836 nodes were renumbered, so evidence links pointed at wrong memories."""
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  monkeypatch.setenv("GA_THOUGHT_DEDUP", "0")          # a save written before the gate
  m = full_mem(tmp_path)
  t = D(2023, 2, 13, 8)
  for i in range(3):
    add(m, "Wolfgang Schulz is sleeping", t + datetime.timedelta(hours=i))
  m.add_event(t, None, "Wolfgang Schulz", "is", "idle", "Wolfgang Schulz is idle",
              {"idle"}, 3, PAIR("Wolfgang Schulz is idle"), [])
  m.add_thought(t, None, "Wolfgang Schulz", "is", "tired", "Wolfgang Schulz is tired",
                {"tired"}, 4, PAIR("Wolfgang Schulz is tired"), ["node_4"])
  out = tmp_path / "save"
  out.mkdir()
  m.save(str(out))
  monkeypatch.setenv("GA_THOUGHT_DEDUP", "1")
  back = am.AssociativeMemory(str(out))
  assert len(back.seq_thought) == 4
  assert {k: v.description for k, v in back.id_to_node.items()} == \
         {k: v.description for k, v in m.id_to_node.items()}
  assert back.id_to_node["node_5"].filling == ["node_4"]
  assert back.id_to_node["node_4"].description == "Wolfgang Schulz is idle"
  assert not (tmp_path / "thought_dedup.jsonl").exists()


def test_numbers_and_times_make_thoughts_different(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  assert add(m, "Wolfgang Schulz will meet Maria at 5:00 on February 13", D(2023, 2, 13, 8))
  assert add(m, "Wolfgang Schulz will meet Maria at 7:00 on February 14", D(2023, 2, 13, 9))
  assert add(m, "Wolfgang Schulz will meet Maria at 7:00 on February 14.", D(2023, 2, 13, 10)) is None


def test_chat_planning_notes_are_never_rejected(monkeypatch, tmp_path):
  """They carry the chat's commitments: a new step shares most words with
  the previous note about the same person."""
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  a = "For Tony Stewart's planning: Tony Stewart needs to check Chad's blood pressure reading with the patient"
  b = "For Tony Stewart's planning: Tony Stewart needs to check Chad's blood pressure reading and ask the patient about a referral"
  assert add(m, a, D(2023, 2, 13, 8), s="Tony Stewart") and add(m, b, D(2023, 2, 13, 9), s="Tony Stewart")
  assert not m.is_dup_thought(D(2023, 2, 13, 10), b)


def test_is_dup_thought_answers_before_the_expensive_calls(monkeypatch, tmp_path):
  monkeypatch.setenv("GA_TRACE_DIR", str(tmp_path))
  m = mem()
  add(m, "Wolfgang Schulz is sleeping", D(2023, 2, 13, 8))
  assert m.is_dup_thought(D(2023, 2, 13, 9), "Wolfgang Schulz is sleeping")
  assert not m.is_dup_thought(D(2023, 2, 13, 9), "Wolfgang Schulz is cooking oatmeal")
  monkeypatch.setenv("GA_THOUGHT_DEDUP", "0")
  assert not m.is_dup_thought(D(2023, 2, 13, 9), "Wolfgang Schulz is sleeping")


def test_default_trace_dir_is_the_repo_ga_traces(monkeypatch):
  monkeypatch.delenv("GA_TRACE_DIR", raising=False)
  seen = {}
  monkeypatch.setattr(am.os, "makedirs", lambda p, exist_ok=False: seen.setdefault("dir", p))
  monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError()))
  am._trace_dedup(D(2023, 2, 13), "x", "a", "b")
  repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  assert os.path.abspath(seen["dir"]) == os.path.join(repo, "ga_traces")


def test_unique_nodes_drops_repeats_and_keeps_the_first():
  N = lambda i, d: types.SimpleNamespace(node_id=f"node_{i}", description=d)
  a, b = N(1, "Klaus is writing a paper about gentrification"), N(2, "Klaus went to the cafe at 9:00")
  c = N(3, "Klaus is writing a paper about gentrification.")
  d = N(4, "Klaus went to the cafe at 11:00")
  assert am.unique_nodes([a, b, a, c, d, b]) == [a, b, d]


def test_memo_gets_the_name_only_when_it_continues_the_sentence():
  import importlib
  rf = importlib.import_module("persona.cognitive_modules.reflect")
  k = "Klaus Mueller"
  assert rf._memo_with_name(k, "Klaus Mueller found the shift interesting.") == "Klaus Mueller found the shift interesting."
  assert rf._memo_with_name(k, "Rajiv Patel said he'd return.") == "Rajiv Patel said he'd return."
  assert rf._memo_with_name(k, "found the shift interesting.") == "Klaus Mueller found the shift interesting."
  assert rf._memo_with_name(k, '"thought the cake was great."') == "Klaus Mueller thought the cake was great."
