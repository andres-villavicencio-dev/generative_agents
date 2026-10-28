"""Retrieval cache for task decomposition (GA_DECOMP_CACHE=1).

72% of task_decomp LLM calls ask for a (normalised activity, duration) that an
earlier call already decomposed ("have lunch" x60 min was asked 313 times). A hit
returns a stored decomposition with the persona's first name substituted; a miss
goes to the LLM and the result is stored. Candidates must not mention other
people, and gendered pronouns only match personas of that gender."""
import json, os, random, re, threading

_LOCK = threading.Lock()
_CACHE = None
_PATH = os.environ.get("GA_DECOMP_CACHE_FILE") or os.path.join(
    os.path.dirname(__file__), "../../../../ga_traces/decomp_cache.json")
MAX_PER_KEY = 8
_PRON = {"m": re.compile(r"\b(he|his|him|himself)\b", re.I), "f": re.compile(r"\b(she|her|hers|herself)\b", re.I)}


def norm(task):
    t = re.sub(r"\(.*?\)", "", task or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", t)).strip()


def key(task, duration):
    return f"{norm(task)}|{int(duration)}"


def gender_of_text(text):
    m, f = len(_PRON["m"].findall(text)), len(_PRON["f"].findall(text))
    return "m" if m > f else "f" if f > m else "n"


def persona_gender(persona):
    s = persona.scratch
    return gender_of_text(" ".join(str(getattr(s, k, "")) for k in ("learned", "currently")))


def _load():
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            try:
                _CACHE = json.load(open(_PATH))
            except Exception:
                _CACHE = {}
    return _CACHE


def _clean_steps(steps, first, total):
    """None unless the steps sum to `total`, and mention nobody but {F}."""
    if not steps or sum(d for _, d in steps) != total:
        return None
    out = []
    for text, d in steps:
        if re.search(r"[^.\s](?:\s)[A-Z][a-z]+", text.replace(first, "")):   # another capitalised word mid-sentence
            return None
        out.append([text.replace(first, "{F}"), int(d)])
    return out


def lookup(persona, task, duration):
    c = _load().get(key(task, duration))
    if not c:
        return None
    g = persona_gender(persona)
    ok = [e for e in c if e["g"] in ("n", g)]
    if not ok:
        return None
    e = random.choice(ok)
    first = persona.scratch.get_str_firstname()
    return [[t.replace("{F}", first), d] for t, d in e["steps"]]


def add(persona, task, duration, steps):
    first = persona.scratch.get_str_firstname()
    st = _clean_steps([(s[0], int(s[1])) for s in steps if len(s) == 2], first, int(duration))
    if not st:
        return
    g = gender_of_text(" ".join(t for t, _ in st))
    if g != "n" and g != persona_gender(persona):
        return
    c = _load()
    with _LOCK:
        lst = c.setdefault(key(task, duration), [])
        if any(e["steps"] == st for e in lst) or len(lst) >= MAX_PER_KEY:
            return
        lst.append({"g": g, "steps": st})
        try:
            os.makedirs(os.path.dirname(os.path.abspath(_PATH)), exist_ok=True)
            json.dump(c, open(_PATH, "w"))
        except Exception:
            pass
