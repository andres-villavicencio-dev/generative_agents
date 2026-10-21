"""
File: concurrency_utils.py
Description: Shared threading primitives for the GA backend.

Vector 1 (parallel cognition) introduced a thread pool over per-persona
cognition in reverie.py. Conversations mutate BOTH personas' scratch state
and reaction decisions read other personas' state, so the whole
reaction phase (should_react -> chat_react) must serialize on CONVO_LOCK.
This module exists so both reverie.py and persona/cognitive_modules/plan.py
can import the SAME lock object without circular imports.
"""
import threading

# Serializes the conversation/reaction decision phase.
CONVO_LOCK = threading.Lock()

# Names of personas currently generating a conversation (guarded by
# CONVO_LOCK). Conversation generation runs OUTSIDE the lock; a busy persona
# can't be pulled into a second chat until its first one is written back.
CHAT_BUSY = set()

# Per-persona locks for finer-grained future use (currently unused).
_PERSONA_LOCKS = {}
_PERSONA_LOCKS_GUARD = threading.Lock()


def get_persona_lock(name):
    with _PERSONA_LOCKS_GUARD:
        if name not in _PERSONA_LOCKS:
            _PERSONA_LOCKS[name] = threading.Lock()
        return _PERSONA_LOCKS[name]

# ---------------------------------------------------------------------------
# Async conversations (Oct 21 2026, fresh25_1): a whole conversation (up to
# 12 utterances + summary + both re-plans) used to be generated INSIDE one
# step, so every other persona waited minutes at the step barrier. Now the
# step that decides a chat only QUEUES it (speakers stay claimed in
# CHAT_BUSY); after the step's worker pool finishes, reverie starts it on a
# background thread and both speakers are FROZEN (their move() is skipped —
# they stand still showing a chat bubble) until it is written back. Nothing
# else touches a frozen persona, so the background thread owns it.
# GA_ASYNC_CHAT=0 restores in-step generation.
import os as _os

PENDING_CHATS = []      # (fn, args, (init_name, target_name)) — guarded by CONVO_LOCK
ASYNC_FROZEN = {}       # persona name -> chat partner name
_CHAT_THREADS = []


def async_chat_enabled():
    return _os.environ.get("GA_ASYNC_CHAT", "1") == "1"


def start_pending_chats(set_context=None):
    """Called by reverie at the serial point after a step's pool: start every
    queued conversation on its own thread and freeze its speakers."""
    with CONVO_LOCK:
        jobs = PENDING_CHATS[:]
        del PENDING_CHATS[:]
        for _fn, _args, (a, b) in jobs:
            ASYNC_FROZEN[a] = b
            ASYNC_FROZEN[b] = a
    for fn, args, pair in jobs:
        def _run(fn=fn, args=args, pair=pair):
            try:
                if set_context:
                    set_context(pair[0])
                fn(*args)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[AsyncChat] {pair[0]} <-> {pair[1]} failed: {e}")
            finally:
                with CONVO_LOCK:
                    for n in pair:
                        ASYNC_FROZEN.pop(n, None)
                    CHAT_BUSY.difference_update(pair)
                print(f"[AsyncChat] {pair[0]} <-> {pair[1]} done")
        t = threading.Thread(target=_run, name=f"chat:{pair[0]}", daemon=True)
        _CHAT_THREADS.append(t)
        t.start()
        print(f"[AsyncChat] started {pair[0]} <-> {pair[1]}")
    _CHAT_THREADS[:] = [t for t in _CHAT_THREADS if t.is_alive()]


def join_chat_threads(timeout=900):
    """Block until running conversations are written back (checkpoint saves
    must not serialize a half-written persona)."""
    import time as _time
    deadline = _time.time() + timeout
    for t in list(_CHAT_THREADS):
        t.join(max(0.0, deadline - _time.time()))
    _CHAT_THREADS[:] = [t for t in _CHAT_THREADS if t.is_alive()]
