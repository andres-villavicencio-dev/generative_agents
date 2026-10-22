"""Per-family format validators + closed-form answer extraction for GA chat-lane calls.

Used both to drop malformed TEACHER outputs from the SFT set and to score student
outputs on the held-out split. validate(family, prompt, content) -> (ok, reason).
extract(family, prompt, content) -> normalised closed-form answer or None.
"""
import re

_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean(text):
    """What the sim's parser sees: think blocks stripped, light markdown removed."""
    t = _THINK.sub("", text or "")
    t = t.replace("</think>", "")
    if "<think>" in t:
        t = t.split("<think>")[0]
    t = re.sub(r"```[a-zA-Z]*", "", t)
    t = t.replace("**", "")
    return t.strip()


def family_of(rec):
    c = rec.get("caller")
    if c:
        return c
    p = rec.get("prompt") or ""
    if "sitting down at your personal computer" in p:
        return "none:work_session"
    if "plan today in broad-strokes" in p:
        return "none:broad_plan"
    return "none:other"


# ---------- location list parsing -------------------------------------------------
_PLACE_LINE = re.compile(r"^- (.+?): (.+)$")


def parse_places(prompt):
    """{area: {room: set(objects)}} from the action_location prompt's place list."""
    places = {}
    sec = prompt.split("can go", 1)[-1]
    for line in sec.splitlines():
        m = _PLACE_LINE.match(line.strip())
        if not m:
            continue
        area, rest = m.group(1).strip(), m.group(2)
        rooms = {}
        for part in re.split(r";\s*(?![^\[]*\])", rest):
            part = part.strip()
            if not part:
                continue
            mm = re.match(r"^(.*?)\s*\[(.*)\]\s*$", part)
            if mm:
                rooms[mm.group(1).strip()] = {o.strip() for o in mm.group(2).split(",") if o.strip()}
            else:
                rooms[part] = set()
        places[area] = rooms
    return places


def _one_line_output(t):
    lines = [l for l in t.splitlines() if l.strip()]
    if len(lines) != 1:
        return None
    m = re.match(r"^\s*output\s*:\s*(.*)$", lines[0])
    return m.group(1).strip() if m else None


def _yaml_output(t):
    """First `output:` value (single line or YAML list)."""
    m = re.search(r"^\s*output\s*:\s*(.*)$", t, re.M)
    if not m:
        return None
    return m.group(1).strip()


def _unq(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.strip()


def _braced_options(prompt, last=True):
    opts = re.findall(r"\{([^{}]*)\}", prompt)
    if not opts:
        return []
    o = opts[-1] if last else opts[0]
    return [x.strip() for x in o.split(",") if x.strip()]


def _total_minutes(prompt):
    m = re.findall(r"total duration in minutes:?\s*(\d+)", prompt)
    return int(m[-1]) if m else None


_DECOMP_LINE = re.compile(r"^\s*\d+\)\s*(.+?)\(duration in minutes:\s*(\d+),\s*minutes left:\s*(-?\d+)\)\s*$")
_SCHED_LINE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*[~–—-]\s*(\d{1,2}):(\d{2})\s*(?:--|—|–)\s*(.+)$")
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([AaPp])\.?\s*[Mm]\.?")


def validate(family, prompt, content):
    t = clean(content)
    if not t:
        return False, "empty"
    f = family
    if f == "run_gpt_prompt_action_location":
        v = _one_line_output(t)
        if v is None:
            return False, "not_one_output_line"
        parts = [p.strip() for p in _unq(v).split("|")]
        if len(parts) != 3:
            return False, "not_3_parts"
        a, r, o = parts
        places = parse_places(prompt)
        if a not in places:
            return False, "area_not_in_list"
        if r not in places[a]:
            return False, "room_not_in_area"
        objs = places[a][r]
        if objs and o not in objs:
            return False, "object_not_in_room"
        return True, "ok"
    if f in ("run_gpt_prompt_event_poignancy", "run_gpt_prompt_chat_poignancy"):
        v = _one_line_output(t)
        if v is None:
            return False, "not_one_output_line"
        try:
            n = int(_unq(v))
        except ValueError:
            return False, "not_int"
        return (1 <= n <= 10), ("ok" if 1 <= n <= 10 else "out_of_range")
    if f in ("run_gpt_prompt_summarize_conversation", "run_gpt_prompt_agent_chat_summarize_relationship",
             "run_gpt_prompt_memo_on_convo", "run_gpt_prompt_act_obj_desc", "run_gpt_prompt_pronunciatio"):
        v = _one_line_output(t)
        if v is None or not _unq(v):
            return False, "not_one_output_line"
        return True, "ok"
    if f == "run_gpt_prompt_focal_pt":
        if _yaml_output(t) is None and not re.search(r"^\s*(\d+[).]|-)\s*\S", t, re.M):
            return False, "no_output"
        qs = re.findall(r"^\s*(?:-|\d+[).])\s*\"?(.+?)\"?\s*$", t, re.M)
        v = _yaml_output(t) or ""
        if not qs and not v.strip("[]\" "):
            return False, "no_questions"
        return True, "ok"
    if f == "generate_one_utterance":
        mu = re.search(r"^\s*utterance\s*:\s*(.+)$", t, re.M)
        me = re.search(r"^\s*end\s*:\s*(true|false)\s*$", t, re.M | re.I)
        if not mu or not me:
            return False, "missing_utterance_or_end"
        extra = [l for l in t.splitlines() if l.strip() and not re.match(r"^\s*(utterance|end)\s*:", l)]
        if extra:
            return False, "extra_lines"
        return True, "ok"
    if f == "run_gpt_prompt_task_decomp":
        total = _total_minutes(prompt)
        lines = [l for l in t.splitlines() if re.match(r"^\s*\d+\)", l)]
        if not lines:
            return False, "no_numbered_lines"
        durs = []
        for l in lines:
            m = _DECOMP_LINE.match(l)
            if not m:
                return False, "bad_line_format"
            durs.append(int(m.group(2)))
        if total is not None and sum(durs) != total:
            return False, "sum_%d_vs_%d" % (sum(durs), total)
        return True, "ok"
    if f == "run_gpt_prompt_new_decomp_schedule":
        lines = [l for l in t.splitlines() if _SCHED_LINE.match(l.strip())]
        if len(lines) < 2:
            return False, "no_schedule_lines"
        prev = None
        for l in lines:
            m = _SCHED_LINE.match(l.strip())
            s = int(m.group(1)) * 60 + int(m.group(2))
            e = int(m.group(3)) * 60 + int(m.group(4))
            if e < s or (prev is not None and s < prev):
                return False, "non_monotone"
            prev = e
        return True, "ok"
    if f in ("run_gpt_prompt_event_triple", "run_gpt_prompt_act_obj_event_triple"):
        if not re.match(r"^\(\s*[^,()]+,\s*[^,()]+,\s*.+\)\s*$", t.splitlines()[0].strip()):
            return False, "not_triple"
        return True, "ok"
    if f in ("run_gpt_prompt_action_sector", "run_gpt_prompt_action_arena", "run_gpt_prompt_action_game_object"):
        v = re.sub(r"^Answer:\s*", "", t.splitlines()[0].strip(), flags=re.I).strip("{}").strip()
        opts = _braced_options(prompt)
        if not opts:
            return True, "no_options_parsed"
        return (v in opts), ("ok" if v in opts else "not_in_options")
    if f == "run_gpt_prompt_wake_up_hour":
        ok = bool(_TIME.search(t)) and len(t) < 80
        return ok, ("ok" if ok else "no_time_or_too_long")
    if f == "run_gpt_prompt_daily_plan":
        n = len(re.findall(r"^\s*\d+\)\s*\S", t, re.M))
        return (n >= 3), ("ok" if n >= 3 else "few_items")
    if f == "none:broad_plan":
        n = len(re.findall(r"(?:^|\s)\d+\.\s*\S", t))
        return (n >= 3), ("ok" if n >= 3 else "few_items")
    if f == "run_gpt_prompt_create_artifact":
        parts = t.strip().splitlines()[0].split("|||")
        if len(parts) != 4:
            return False, "not_4_fields"
        try:
            q = int(re.sub(r"[^\d]", "", parts[3]) or "x")
        except ValueError:
            return False, "bad_score"
        return (1 <= q <= 10), ("ok" if 1 <= q <= 10 else "score_out_of_range")
    if f == "run_gpt_prompt_insight_and_guidance":
        n = len(re.findall(r"\(because of [\d,\s]+\)", t))
        return (n >= 1), ("ok" if n >= 1 else "no_because")
    if f == "none:work_session":
        ok = bool(re.match(r"^\s*OP:\s*(OPEN|APPEND|REVISE|NEW_FILE|FINISH)\b", t)) and "CONTENT:" in t
        return ok, ("ok" if ok else "bad_op_format")
    # free text families
    return (len(t) >= 1), ("ok" if t else "empty")


def extract(family, prompt, content):
    """Closed-form answer for agreement scoring (None when not closed-form / unparseable)."""
    t = clean(content)
    f = family
    try:
        if f == "run_gpt_prompt_action_location":
            v = _one_line_output(t)
            return tuple(p.strip().lower() for p in _unq(v).split("|")) if v else None
        if f in ("run_gpt_prompt_event_poignancy", "run_gpt_prompt_chat_poignancy"):
            v = _one_line_output(t)
            return int(_unq(v)) if v else None
        if f in ("run_gpt_prompt_action_sector", "run_gpt_prompt_action_arena", "run_gpt_prompt_action_game_object"):
            return re.sub(r"^Answer:\s*", "", t.splitlines()[0].strip(), flags=re.I).strip("{}").strip().lower()
        if f == "run_gpt_prompt_wake_up_hour":
            m = _TIME.search(t)
            if not m:
                return None
            h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
            return h * 60 + int(m.group(2) or 0)
        if f in ("run_gpt_prompt_event_triple", "run_gpt_prompt_act_obj_event_triple"):
            return t.splitlines()[0].strip().lower()
        if f == "run_gpt_prompt_act_obj_desc":
            v = _one_line_output(t)
            return _unq(v).lower() if v else None
        if f == "generate_one_utterance":
            me = re.search(r"^\s*end\s*:\s*(true|false)\s*$", t, re.M | re.I)
            return me.group(1).lower() if me else None
    except Exception:  # noqa: BLE001
        return None
    return None


CLOSED_FORM = {"run_gpt_prompt_action_location", "run_gpt_prompt_event_poignancy",
               "run_gpt_prompt_chat_poignancy", "run_gpt_prompt_action_sector",
               "run_gpt_prompt_action_arena", "run_gpt_prompt_action_game_object",
               "run_gpt_prompt_wake_up_hour", "run_gpt_prompt_event_triple",
               "run_gpt_prompt_act_obj_event_triple", "run_gpt_prompt_act_obj_desc",
               "generate_one_utterance"}
