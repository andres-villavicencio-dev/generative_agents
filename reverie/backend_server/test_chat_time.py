"""Chat-time accounting test battery. Run: python test_chat_time.py
Covers: the module itself (charge/drain/thread-safety/env knobs), the
converse.py wiring (agent_chat_v2 charges per reply), and the reverie.py
step-loop drain (clock advances by sec_per_step + debt, meta curr_time
follows). No live backend needed — agent_chat_v2 is exercised with the
LLM helpers monkeypatched.
"""
import os
import sys
import datetime
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else: FAIL += 1
    print(f"  [{tag}] {name}" + (f" | {detail}" if detail else ""))

# ---------------------------------------------------------------- module
print("== chat_time module ==")
import chat_time as ct

ct.drain_tick()  # clean slate
check("gate default ON", ct.chat_time_enabled())

ct.charge_reply()
sec, utts = ct.peek()
check("charge_reply accumulates", (sec, utts) == (60.0, 1), f"{sec}s/{utts}")

ct.charge_reply(); ct.charge_reply()
sec, utts = ct.peek()
check("three replies = 180s", (sec, utts) == (180.0, 3), f"{sec}s/{utts}")

d, u = ct.drain_tick()
check("drain returns and resets", (d, u) == (180.0, 3))
d2, u2 = ct.drain_tick()
check("second drain is empty", (d2, u2) == (0.0, 0))

os.environ["GA_CHAT_TIME"] = "0"
ct.charge_reply()
check("GA_CHAT_TIME=0 opts out", ct.peek() == (0.0, 0))
del os.environ["GA_CHAT_TIME"]

os.environ["GA_CHAT_SECONDS_PER_REPLY"] = "30"
ct.charge_reply()
sec, _ = ct.peek()
check("custom per-reply seconds honored", sec == 30.0, f"{sec}s")
del os.environ["GA_CHAT_SECONDS_PER_REPLY"]
ct.drain_tick()

# thread safety: 8 threads x 25 charges, no lost updates
def hammer():
    for _ in range(25):
        ct.charge_reply()
threads = [threading.Thread(target=hammer) for _ in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
sec, utts = ct.peek()
check("thread-safe accounting", (sec, utts) == (12000.0, 200), f"{sec}s/{utts}")
ct.drain_tick()

# ---------------------------------------------------------------- converse wiring
print("== converse.py agent_chat_v2 wiring ==")
# Import the module WITHOUT triggering heavy imports: converse.py pulls the
# world tree, but its imports are all local-path tolerant (same package
# layout as the backend's cwd).
os.environ.setdefault("GA_CHAT_TIME", "1")
from persona.cognitive_modules import converse as conv
from persona.cognitive_modules import retrieve as retr

class FakeScratch:
    name = "Test A"
    act_description = "waiting outside Hobbs Cafe"
    curr_tile = (50, 50)

class FakePersona:
    scratch = FakeScratch()

class FakeMaze:
    def access_tile(self, tile):
        return {"events": []}

a, b = FakePersona(), FakePersona()
a.scratch.name, b.scratch.name = "Test A", "Test B"

# Monkeypatch the LLM-touching helpers converse.py calls
conv.new_retrieve = lambda p, fp, n=30: []
conv.generate_summarize_agent_relationship = lambda ip, tp, r: "friends"
conv.run_gpt_generate_iterative_chat_utt = (
    lambda maze, ip, tp, r, ctx, chat:
    ({"utterance": f"line {len(chat)}", "end": len(chat) >= 5},))

ct.drain_tick()
before = ct.peek()[0]
convo = conv.agent_chat_v2(FakeMaze(), a, b)
after, utts = ct.peek()
n_utts = len(convo)
check("every reply charged", utts == n_utts,
      f"charged {utts} vs convo len {n_utts}")
check("end flag stops charging", utts <= 12, f"{utts} charges (cap 6 rounds)")
check("convo content intact", all(row[1].startswith("line") for row in convo))
ct.drain_tick()

# ---------------------------------------------------------------- reverie drain
print("== reverie step-loop drain (unit-level) ==")
# Simulate the drain arithmetic the step loop performs.
sec_per_step = 10
ct.charge_reply(); ct.charge_reply(); ct.charge_reply()  # 3 replies = 180s
t0 = datetime.datetime(2023, 2, 13, 12, 0, 0)
debt, utts = ct.drain_tick()
t1 = t0 + datetime.timedelta(seconds=sec_per_step + debt)
expected = datetime.datetime(2023, 2, 13, 12, 3, 10)  # +3min10s
check("clock advances sec_per_step + debt", t1 == expected, f"{t1} == {expected}")
check("drain resets after use", ct.drain_tick() == (0.0, 0))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)