# Copyright (c) 2026 Andus VU — MIT license
"""Voyager-inspired routine library + self-curriculum for generative agents.

Layer 2 of the agency stack: agents build PERSONAL routine libraries from
their own harvested action history (Voyager's verified-skill-library move,
translated to schedules), and a self-curriculum (Voyager's automatic
curriculum, translated to life-goals via an arena-coverage map per persona).
Both feed goals into the agentic-drives injection seam in plan.py.

Storage: storage/<sim>/personas/<Name>/routines.json — per-persona, forks
with the sim, survives checkpoints. Harvest is LLM-free (pure mining).

Env gates: GA_ROUTINES=1 (routine retrieval), GA_CURRICULUM=1 (coverage
goals). Both default OFF; fail-open everywhere.
"""
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

BACKEND = os.path.dirname(os.path.abspath(__file__))


def _env_on(flag):
    return os.environ.get(flag, "0") == "1"


# ---------------------------------------------------------------------------
# Harvest: build routines from REAL recorded history
# ---------------------------------------------------------------------------
def _sim_roster():
    sims = sorted(glob.glob(
        f'{BACKEND}/../../environment/frontend_server/storage/live3d_n*'))
    return sims[-1] if sims else None


def _movement_arenas(sim_dir, last_n=3):
    """Arenas visited in the newest movement files (cheap tail read)."""
    mv = os.path.join(sim_dir, 'movement')
    if not os.path.isdir(mv):
        return defaultdict(Counter)
    files = os.listdir(mv)

    def _k(f):
        try:
            return int(f[:-5])
        except Exception:
            return -1

    out = defaultdict(Counter)
    for f in sorted(files, key=_k)[-last_n:]:
        try:
            data = json.load(open(os.path.join(mv, f)))
        except Exception:
            continue
        for name, rec in data.get('persona', {}).items():
            desc = rec.get('description', '')
            if '@' not in desc:
                continue
            addr = desc.split('@', 1)[1].strip()
            if '<persona>' in addr:
                continue  # chat-pending actions anchor to '<persona> X', not a place
            parts = addr.split(':')
            # arena = 3rd path element (world:sector:arena:object);
            # fall back to the last element for short addresses
            arena = parts[2].strip() if len(parts) >= 3 else parts[-1].strip()
            out[name][arena] += 1
    return out


def harvest_routines(sim_dir=None, min_count=2):
    """Mine real movement history into per-persona routines:
    {name: [{name, arena, count}]}."""
    sim_dir = sim_dir or _sim_roster()
    if not sim_dir:
        return {}
    by_name = _movement_arenas(sim_dir, last_n=3)
    out = defaultdict(list)
    for name, arenas in by_name.items():
        for arena, count in arenas.most_common(6):
            if count >= min_count:
                out[name].append({
                    'name': f'at the {arena}',
                    'arena': arena,
                    'count': count,
                })
    return dict(out)


def write_routines(sim_dir=None):
    """Harvest + persist routines.json into each persona's storage dir.
    Atomic writes (tmp + os.replace) — the fork-integrity lesson."""
    sim_dir = sim_dir or _sim_roster()
    if not sim_dir:
        return 0
    routines = harvest_routines(sim_dir)
    n = 0
    for name, rs in routines.items():
        pdir = os.path.join(sim_dir, 'personas', name)
        if not os.path.isdir(pdir):
            continue
        path = os.path.join(pdir, 'routines.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'routines': rs}, f, indent=2)
        os.replace(tmp, path)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Retrieval: a routine as a plan fragment seed
# ---------------------------------------------------------------------------
def load_routines_for(persona):
    """Persona's routine library from its storage dir (routines.json), or
    []. Resolves the persona's sim folder from scratch.sim_folder (set by
    reverie for every live persona) with the roster-dir fallback."""
    try:
        folder = getattr(persona.scratch, 'sim_folder', None)
        if not folder:
            folder = _sim_roster()
        if not folder:
            return []
        path = os.path.join(folder, 'personas',
                            getattr(persona, 'name', ''), 'routines.json')
        if not os.path.isfile(path):
            return []
        data = json.load(open(path))
        return data.get('routines', [])
    except Exception:
        return []


def _arena_of(description):
    """Arena from an action description's '@ address' tail — 3rd path element
    (world:sector:arena:object), last element fallback, None for persona-
    anchors. THE canonical arena extractor: harvest and retrieval must agree."""
    if '@' not in description:
        return None
    addr = description.split('@', 1)[1].strip()
    if '<persona>' in addr:
        return None
    parts = addr.split(':')
    return (parts[2].strip() if len(parts) >= 3 else parts[-1].strip())


def _recent_arenas(persona, hours=20):
    """Arenas appearing in the persona's last-<hours> event memories."""
    import datetime
    seen = set()
    try:
        nodes = [n for n in persona.a_mem.id_to_node.values()
                 if n.type == 'event']
        day_start = persona.scratch.curr_time - datetime.timedelta(hours=hours)
        for n in nodes:
            if n.created and n.created >= day_start:
                a = _arena_of(n.description)
                if a:
                    seen.add(a)
    except Exception:
        pass
    return seen


def routine_goal(persona, maze=None):
    """Goal string drawn from the persona's OWN routine library, or None.
    Prefers routine arenas NOT visited in the last 20h (a routine refreshes
    life, it doesn't replay it)."""
    try:
        if not _env_on('GA_ROUTINES'):
            return None
        sc = getattr(persona, 'scratch', None)
        if sc is None:
            return None
        lib = getattr(sc, 'routines', None)
        if not lib:
            # lazy disk read: the harvest pass writes routines.json next to
            # scratch.json; old checkpoints won't have it in scratch yet.
            lib = load_routines_for(persona)
            if lib:
                sc.routines = lib
        if not lib:
            return None
        recent = _recent_arenas(persona)
        fresh = [r for r in lib if r['arena'] not in recent]
        pool = fresh or lib
        pool = sorted(pool, key=lambda r: -r['count'])[:3]
        pick = random.Random(f"routine-{sc.name}-{sc.curr_time}").choice(pool)
        return f"spending some time {pick['name']}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Self-curriculum: arena-coverage map -> exploration goals
# ---------------------------------------------------------------------------
def curriculum_goal(persona, maze=None):
    """Voyager's automatic curriculum, translated: ONE arena (not object)
    the persona has NEVER recorded visiting, as {'arena_name', 'address'},
    or None. Filters object-level rooms (bedrooms, bathrooms) and arenas
    belonging to personas not in this sim's roster."""
    try:
        if not _env_on('GA_CURRICULUM'):
            return None
        sc = getattr(persona, 'scratch', None)
        if sc is None or maze is None:
            return None
        seen = _recent_arenas(persona, hours=24 * 30)  # whole-life tail
        visited = set()
        for arena in seen:
            for addr in maze.address_tiles:
                parts = addr.split(':')
                if len(parts) >= 3 and parts[-1].strip().lower() == arena.lower():
                    visited.add(addr)
                    break
        roster = set()
        try:
            roster = set(getattr(persona, '_roster_names', []) or [])
        except Exception:
            roster = set()
        never = []
        for a in maze.address_tiles:
            if a in visited or len(a.split(':')) < 3:
                continue
            parts = a.split(':')
            sector, arena = parts[1], parts[2]
            # skip object-level rooms anchored to a specific person
            if re.search(r"\b(bedroom|bathroom|room)\b", arena, re.I):
                continue
            if roster and re.search(r"\b\w+(?:'s)?\s", arena) and not (
                    any(n.split()[0] in arena for n in roster)):
                continue
            never.append(a)
        if not never:
            return None
        # seed includes the sim date: a fresh proposal each sim-day,
        # still deterministic within a day (offline-testable)
        cand = random.Random(
            f"cur-{sc.name}-{getattr(sc, 'curr_time', None)}").choice(never)
        parts = cand.split(':')
        return {'arena_name': parts[2].strip() if len(parts) > 2 else cand,
                'sector': parts[1] if len(parts) > 1 else '',
                'address': cand}
    except Exception:
        return None