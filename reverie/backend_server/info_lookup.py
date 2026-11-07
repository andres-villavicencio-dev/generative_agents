"""Computer lookup tool: an agent seated at a computer who is searching/checking something
gets a true, current digest of the town's public information (job openings, paid work waiting,
shop stock trouble, the Community Fund). The result is written to their memory by the caller.

Mechanism only: what the agent does with the information (apply, shop, avoid) is its own decision.
"""
import os
import re

ENABLED = os.environ.get("GA_INFO_LOOKUP", "1") == "1"

_LOOKUP = re.compile(
    r"\b(look(ing)? up|search(ing)? (online|the web|for)|search bar|check(ing)? (the |their |his |her )?"
    r"(job|listing|opening|news|price|event|online|catalog|inventory|sales|schedule)s?|browse|browsing|"
    r"research(ing)?|job (search|hunt)(ing)?|online|internet|read(ing)? the news|find(ing)? out)\b", re.I)
_NOT = re.compile(r"^\s*(convers|chat|discuss|talk)", re.I)       # conversations merely mention these words
_DEVICE = re.compile(r"\b(phone|laptop|computer|online|internet|search bar|website)\b", re.I)


def matches(act_description):
    d = act_description or ""
    return bool(d) and not _NOT.search(d) and bool(_LOOKUP.search(d))


def has_device(maze, persona, act_description):
    """Everyone carries a phone; otherwise they must be at a computer."""
    if _DEVICE.search(act_description or ""):
        return True
    try:
        t = maze.access_tile(persona.scratch.curr_tile)
        return "computer" in (t.get("game_object") or "").lower()
    except Exception:
        return False


def digest(economy, name):
    """True facts, one short paragraph. Returns '' when there is nothing to report."""
    import economy as E
    bits = []
    try:
        opts = economy.board_options(name)
        if opts:
            bits.append("Openings: " + "; ".join(o[2] for o in opts[:4]) + ".")
        else:
            bits.append("No job openings are posted right now.")
    except Exception:
        pass
    try:
        import duties as D
        st = economy.state.get("duties") or {}
        by = {}
        for d in (st.get("open") or {}).values():
            if not d.get("claimed_by"):
                by[d["venue"]] = by.get(d["venue"], 0.0) + D.ALL_TYPES[d["type"]]["base"]
        top = sorted(by.items(), key=lambda kv: -kv[1])[:3]
        names = [(E.VENUES[v]["name"] if v in E.VENUES else v, round(x, 1)) for v, x in top if x > 0]
        if names:
            bits.append("Paid work is waiting at " + ", ".join(f"{n} (about ${x})" for n, x in names) + ".")
    except Exception:
        pass
    try:
        f = economy.state.get("fund") or {}
        if f.get("balance") is not None:
            bits.append(f"The Community Fund holds ${f['balance']:.0f}; sales tax is {(f.get('rate') or 0) * 100:.1f}%.")
    except Exception:
        pass
    return (f"{name} used the computer to look things up. " + " ".join(bits)) if bits else ""


def once_key(persona):
    s = persona.scratch
    return (s.act_description, str(getattr(s, "act_start_time", "")))
