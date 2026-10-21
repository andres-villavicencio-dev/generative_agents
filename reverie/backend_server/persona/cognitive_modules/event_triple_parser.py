"""Import-light (subject, predicate, object) event-triple parser.

The triple is only a memory-index key, so a deterministic heuristic replaces
the ~13s LLM call (run_gpt_prompt_event_triple) for essentially every input.
"""
import re

# Common verb stems to avoid bad stemming
VERB_MAP = {
  "cooking": "cook", "eating": "eat", "sleeping": "sleeping",
  "reading": "read", "writing": "write", "working": "work",
  "studying": "study", "walking": "walk", "running": "run",
  "shopping": "shop", "cleaning": "clean", "painting": "paint",
  "drawing": "draw", "listening": "listen", "watching": "watch",
  "drinking": "drink", "brewing": "brew", "teaching": "teach",
  "playing": "play", "sitting": "sit", "standing": "stand",
  "waiting": "wait", "talking": "talk", "chatting": "chat",
  "showering": "shower", "bathing": "bathe", "exercising": "exercise",
  "gardening": "garden", "organizing": "organize", "typing": "type",
  "browsing": "browse", "preparing": "prepare", "making": "make",
  "buying": "buy", "having": "have", "getting": "get", "going": "go",
  "heading": "head", "spending": "spend", "exploring": "explore",
}

# First words that mark a noun phrase / adverb / modal clause, not a verb
_NON_VERB_FIRST = {
  "the", "a", "an", "this", "that", "these", "those", "his", "her", "their",
  "its", "my", "our", "your", "it", "he", "she", "they", "we", "there",
  "no", "not", "very", "also", "then", "now", "still", "just", "some",
  "each", "every", "all", "both", "many", "much", "more", "most",
}
_MODALS = {"can", "could", "will", "would", "may", "might", "must", "should",
           "shall", "is", "are", "was", "were", "has", "have", "had", "does",
           "do", "did", "currently"}

_MAX_OBJ = 60


def _stem_3rd(word):
  """Simple 3rd-person -s/-es stemmer: wakes->wake, brushes->brush."""
  w = word.lower()
  if len(w) > 3 and w.endswith("ies"):
    return w[:-3] + "y"
  if len(w) > 3 and re.search(r"(ss|sh|ch|x|z|o)es$", w):
    return w[:-2]
  if len(w) > 2 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
    return w[:-1]
  return w


def _clause(text):
  """Text up to the first comma / ' and ' / period, truncated."""
  text = re.split(r",|;|\s+and\s+|\.(?:\s|$)", text, maxsplit=1)[0]
  return text.strip(" .\t\n")[:_MAX_OBJ].strip()


def _parse(desc):
  # Pattern: "<Name> is <verb>ing <rest>"
  m = re.match(r'^(.+?)\s+is\s+(\w+ing)\b\s*(.*?)\.?$', desc, re.IGNORECASE)
  if m:
    subject = m.group(1).strip()
    verb_ing = m.group(2).strip().lower()
    obj = m.group(3).strip()
    verb = VERB_MAP.get(verb_ing, verb_ing)
    if not obj:
      return (subject, "is", verb_ing)
    return (subject, verb, obj)

  # Pattern: "<Name> is <adjective/state>"
  m = re.match(r'^(.+?)\s+is\s+(\w+)\.?$', desc, re.IGNORECASE)
  if m:
    return (m.group(1).strip(), "is", m.group(2).strip())

  m = re.match(r'^(.+?)\s+is(?:\s+(.*))?$', desc, re.IGNORECASE | re.DOTALL)
  if not m:
    return None
  subject = m.group(1).strip()
  rest = (m.group(2) or "").strip()
  if not rest:
    return (subject, "is", "idle")

  words = rest.split()
  first = re.sub(r"[^\w'-]", "", words[0]).lower()
  second = re.sub(r"[^\w'-]", "", words[1]).lower() if len(words) > 1 else ""
  # Third-person (wakes) / imperative (identify) / other verb form
  if (first and first.isalpha() and first not in _NON_VERB_FIRST
      and not first.endswith("ly") and second not in _MODALS
      and not (len(words) == 1 and rest.endswith("."))):
    verb = _stem_3rd(first)
    obj = _clause(" ".join(words[1:]))
    if not obj:
      return (subject, "is", verb)
    return (subject, verb, obj)

  # Final fallback: (subject, "is", first clause)
  return (subject, "is", _clause(rest) or rest[:_MAX_OBJ])


def parse_event_triple(action_description, persona_name=None):
  """Return (subject, predicate, object); None only if there is no ' is '."""
  desc = (action_description or "").strip()
  if not desc:
    return None
  tried = desc
  if "(" in desc:
    inner = desc.split("(")[-1].split(")")[0]
    res = _parse(inner.strip())
    if res:
      return res
    # Parenthetical had no " is ": drop it and parse the rest
    tried = re.sub(r"\s*\([^)]*\)\s*", " ", desc).strip()
  return _parse(tried)
