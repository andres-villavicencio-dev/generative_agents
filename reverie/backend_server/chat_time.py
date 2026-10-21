"""
Chat-time accounting: conversations take the PARTICIPANTS' time.

A conversation is an activity like any other: while two agents talk, their
turns pass in conversation — "chat with X" lasts as long as the exchange
would take, then they resume their day. The world clock is NOT touched; it
advances sec_per_step every step for everyone.

(Oct 22 2026: the first version charged every reply to a GLOBAL clock
"debt" that reverie added to the world clock, so the whole town jumped
forward while two agents talked — skipped hours, a clock running ahead of
the step count. Replies now set the conversation's duration instead, via
conversation_minutes() in plan.generate_convo.)

Env knobs:
  GA_CHAT_TIME=1     feature on (default ON; 0 = stock length estimate only)
  GA_CHAT_SECONDS_PER_REPLY  sim-seconds one spoken reply takes (default 60)
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


def conversation_minutes(n_replies):
    """How long a conversation of n replies takes its participants, in
    whole sim-minutes (0 when the feature is off)."""
    if not chat_time_enabled() or n_replies <= 0:
        return 0
    return -(-n_replies * _per_reply_seconds() // 60)   # ceil


def charge_reply(seconds=None):
    """Count one generated reply (observability only — the world clock is
    no longer charged; see the module docstring)."""
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
    """Read-and-reset this tick's reply counters: (seconds, utterances).
    Reverie logs them; it does NOT advance the world clock with them."""
    global _DEBT, _UTTERANCES
    with _LOCK:
        debt, utts = _DEBT, _UTTERANCES
        _DEBT, _UTTERANCES = 0.0, 0
    return debt, utts


def peek():
    """Non-destructive read (tests, debug)."""
    with _LOCK:
        return _DEBT, _UTTERANCES