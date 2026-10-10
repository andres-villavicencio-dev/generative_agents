"""
Chat-time accounting (Oct 2026): conversations cost sim-time.

Real life: long conversations take time. The stock sim charges a chat
nothing while it happens — the world clock only moves once per step, and
one step is spent generating the whole conversation anyway. With the
worker-pool backend, a 6-round convo still freezes BOTH chatters for its
wall-clock duration while the rest of the town moves on a clock that
pretends the conversation was instant.

This module keeps a per-tick, thread-safe "chat-time debt": every
generated utterance charges CHAT_SECONDS_PER_REPLY (default 60 sim-sec =
1 sim-minute — about how long a real spoken reply takes). reverie.py
drains the debt when the step completes and advances the world clock by
sec_per_step + debt. The clock keeps moving during long conversations,
other agents' schedules reflect that the chat ate real time, and the
chatting pair's chatting_end_time already covers the conversational
block via the normal react path.

Env knobs:
  GA_CHAT_TIME=1     feature on (default ON; 0 opts out)
  GA_CHAT_SECONDS_PER_REPLY  sim-seconds charged per utterance (default 60)
"""

import os
import threading

_LOCK = threading.Lock()
_DEBT = 0.0          # sim-seconds of conversation time, this tick
_UTTERANCES = 0      # utterances charged, this tick (observability)


def chat_time_enabled():
    """Feature gate. Default ON (GA_CHAT_TIME=0 opts out)."""
    return os.environ.get("GA_CHAT_TIME", "1") != "0"


def _per_reply_seconds():
    try:
        return max(0, int(os.environ.get("GA_CHAT_SECONDS_PER_REPLY", "60")))
    except ValueError:
        return 60


def charge_reply(seconds=None):
    """Charge the world clock for one conversational reply.

    Called by agent_chat_v2 after EVERY generated utterance — 'every
    reply of an agent to its interlocutor allows the clock to continue'.
    Thread-safe: conversations for different pairs run on different
    worker threads inside the same tick.
    """
    global _DEBT, _UTTERANCES
    if not chat_time_enabled():
        return 0.0
    if seconds is None:
        seconds = _per_reply_seconds()
    with _LOCK:
        _DEBT += seconds
        _UTTERANCES += 1
    return seconds


def drain_tick():
    """Consume the accumulated chat-time debt for this tick.

    Returns (seconds, utterance_count) and resets the counters.
    Called ONCE by the reverie step loop, after all persona moves
    complete — the clock advance is a main-thread serial side effect.
    """
    global _DEBT, _UTTERANCES
    with _LOCK:
        debt, utts = _DEBT, _UTTERANCES
        _DEBT, _UTTERANCES = 0.0, 0
    return debt, utts


def peek():
    """Non-destructive read (tests, debug)."""
    with _LOCK:
        return _DEBT, _UTTERANCES