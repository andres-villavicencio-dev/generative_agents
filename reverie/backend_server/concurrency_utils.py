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

# Per-persona locks for finer-grained future use (currently unused).
_PERSONA_LOCKS = {}
_PERSONA_LOCKS_GUARD = threading.Lock()


def get_persona_lock(name):
    with _PERSONA_LOCKS_GUARD:
        if name not in _PERSONA_LOCKS:
            _PERSONA_LOCKS[name] = threading.Lock()
        return _PERSONA_LOCKS[name]