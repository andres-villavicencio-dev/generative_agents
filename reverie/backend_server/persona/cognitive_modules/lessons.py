"""Outcome memory: what failed recently, and why (GA_LESSONS=1 enables).

Before this, a failed purchase / empty fridge became a one-off memory event and
a one-replan fallback goal; the next day's plan repeated the same attempt
(verify_1: 96% of 329 depletion events repeated a failure from the previous
24 sim-hours). A lesson is a short-lived fact keyed by (kind, target):

  home_stock   target=item   reason=sold_out     own fridge/pantry empty
  store_stock  target=item   reason=sold_out     shop shelf empty
  store_stock  target=item   reason=cant_afford  wallet < price

Lessons expire (`until`), stay in `scratch.lessons` (persisted by Scratch) and are
rendered into the daily-plan prompt. Pure functions over a scratch object so they
are trivially testable.
"""
import datetime
import os

_FMT = "%B %d, %Y, %H:%M:%S"
_TTL_H = {"sold_out": 24, "cant_afford": 12, "occupied": 2}
_MAX = 40


def enabled():
  return os.environ.get("GA_LESSONS", "0") == "1"


def _key(kind, target):
  return f"{kind}|{target}"


def record(scratch, kind, target, reason, now):
  """Store/refresh a lesson. Returns True if it was NEW (not already live)."""
  if now is None:
    return False
  book = getattr(scratch, "lessons", None)
  if book is None:
    book = scratch.lessons = {}
  k = _key(kind, target)
  old = book.get(k)
  fresh = old is None or _parse(old.get("until")) is None or _parse(old["until"]) <= now
  until = now + datetime.timedelta(hours=_TTL_H.get(reason, 12))
  book[k] = {"kind": kind, "target": target, "reason": reason,
             "count": (0 if fresh else old.get("count", 0)) + 1,
             "last": now.strftime(_FMT), "until": until.strftime(_FMT)}
  if len(book) > _MAX:  # drop the stalest
    for kk in sorted(book, key=lambda x: book[x]["last"])[:len(book) - _MAX]:
      del book[kk]
  return fresh


def _parse(s):
  try:
    return datetime.datetime.strptime(s, _FMT)
  except Exception:
    return None


def live(scratch, now, kind=None, reason=None):
  """Unexpired lessons (list of dicts), optionally filtered."""
  out = []
  for v in (getattr(scratch, "lessons", None) or {}).values():
    u = _parse(v.get("until"))
    if u is None or now is None or u <= now:
      continue
    if kind and v["kind"] != kind:
      continue
    if reason and v["reason"] != reason:
      continue
    out.append(v)
  return out


def clear(scratch, kind, target):
  (getattr(scratch, "lessons", None) or {}).pop(_key(kind, target), None)


def render(scratch, now, k=5):
  """Prompt text for the daily planner; '' when there is nothing live."""
  items = sorted(live(scratch, now), key=lambda v: -v["count"])[:k]
  if not items:
    return ""
  home = [v["target"] for v in items if v["kind"] == "home_stock"]
  broke = [v["target"] for v in items if v["reason"] == "cant_afford"]
  gone = [v["target"] for v in items
          if v["kind"] == "store_stock" and v["reason"] == "sold_out"]
  lines = []
  if home:
    lines.append(f"Your kitchen has run out of {', '.join(sorted(set(home)))}; "
                 "plan a grocery trip (or eat out) before you try to cook.")
  if broke:
    lines.append(f"You could not afford {', '.join(sorted(set(broke)))} recently; "
                 "earn money or choose cheaper food first.")
  if gone:
    lines.append(f"The shop was sold out of {', '.join(sorted(set(gone)))}; "
                 "try another shop.")
  return "Lessons from recent days: " + " ".join(lines)
