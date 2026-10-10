"""
ville_os.py — Per-persona simulated PC (Ville Dots, Phase 0+).

Every persona can own a small virtual workspace: a persistent file tree that
survives sim-days, checkpoints, and forks (it lives inside the sim storage
tree). Creative actions run bounded WORK SESSIONS against the workspace via
a file-op vocabulary (OPEN/APPEND/REVISE/NEW_FILE/FINISH) instead of only
accumulating artifact-progress minutes.

Design constraints (see docs/ville-dots-plan.md):
- No host contact: everything lives under the sim folder.
- One writer: the backend step loop (same single-writer rule as the browser
  vs headless-driver rule) — the Django PC viewer is read-only.
- All LLM calls route the existing artifact lane; the model callable is
  INJECTED so offline tests can pass a stub.

Storage layout (per persona):
  personas/<Name>/workspace/
    index.json        # file registry + session log (bounded)
    files/<fname>.txt # current text; .rN.txt = kept revisions (capped)

Scratch integration (Phase 1): scratch.active_projects list; scratch.py uses
hasattr-guarded save + .get() load, so absence in old checkpoints is safe.
"""

import json
import os
import re
import time

MAX_REVS = 3          # full-text revisions kept per file
MAX_CONTENT_CHARS = 1500   # clamp on one session's written text
MAX_TAIL_CHARS = 800       # how much of the current file the prompt sees
MAX_INDEX_SESSIONS = 200    # session log entries kept in index.json

VALID_OPS = ("OPEN", "APPEND", "REVISE", "NEW_FILE", "FINISH")

# Ops that change or create file text. OPEN is a read/no-op; FINISH flips status.
WRITING_OPS = ("APPEND", "REVISE", "NEW_FILE")

# --- PC action matcher ---------------------------------------------------
# Built from REAL movement phrasings harvested out of live3d_n45 (1,998
# unique descriptions; 320 creative-ish). Word-stem regex at word boundaries
# ONLY — literal substrings silently fail on real phrasing (matcher disease).
# Negative guards FIRST: reading/browsing/phone/email/lunch must never fire a
# work session (they're consumption or filler, not project work).
_PC_SESSION_PATTERNS = [
    # continue/resume/begin + writing/work — the dominant real phrasings:
    # "continue writing Whispers of the Whispering Pines",
    # "begin writing Whispers of Gentle Courage", "continue focused writing",
    # "resume animation project work ..."
    (re.compile(r"\b(continue|resume|begin|start|finish)\w*\b.*\bwrit(e|ing|es)\b"
               r"(?!.*\bletter\b)", re.IGNORECASE),),
    (re.compile(r"\b(continue|resume|begin|start|finish)\w*\b.*\b"
               r"(work|working)\b.*\bproject\b", re.IGNORECASE),),
    (re.compile(r"\bfocused writing\b|\bwriting session\b", re.IGNORECASE),),
    # generic write-stem with the big consumption guard (read a book etc. is
    # by far the most common action; 'write' inside 'rewind' is boundary-safe)
    (re.compile(r"\bwrit(e|ing|es)\b(?!.*\bletter\b)"
               r"(?!.*\bread)"
               r"(?!.*\breview)"
               r"(?!.*\bphone\b|\bbrowsing\b|\bemail)", re.IGNORECASE),),
    # sketching is project work (harvest: "calm sketching with music")
    (re.compile(r"\bsketch(ing|es)?\b", re.IGNORECASE),),
]
# Absolute negatives — never a PC session even if a positive matched above.
# conversing/chatting/discussing = conversation actions (their summaries often
# mention 'writing' — a real n51 false positive: 'conversing about ... planning
# a lesson that integrates ... narrative writing' fired PC sessions for two
# people standing in a corridor). The chat-protocol '@ <persona>' tail also
# marks conversation actions even if the verb is odd.
_PC_SESSION_NEGATIVES = re.compile(
    r"\b(read(s|ing)?|reading|browse|browsing|phone|email|text(ing)? on|"
    r"lunch|dinner|breakfast|nap|sleep|shower|eat|drink|"
    r"convers(e|ing|ation)|chat(ting)?|discuss(ing)?|talk(ing)?)\b"
    r"|@\s*<persona>", re.IGNORECASE)


def matches_pc_session(act_description):
    """True if this action should trigger a workspace work session."""
    if not act_description:
        return False
    d = act_description
    if _PC_SESSION_NEGATIVES.search(d):
        return False
    return any(p.search(d) for (p,) in _PC_SESSION_PATTERNS)


class WorkspaceError(Exception):
    pass


class WorkspaceStore:
    """Per-persona virtual file system. Lives at <sim>/personas/<Name>/workspace/."""

    def __init__(self, workspace_dir):
        self.dir = workspace_dir
        self.files_dir = os.path.join(workspace_dir, "files")
        self.index_path = os.path.join(workspace_dir, "index.json")
        self.index = {"files": {}, "sessions": []}
        os.makedirs(self.files_dir, exist_ok=True)
        self._load_index()

    # ---------- persistence ----------

    def _load_index(self):
        index_existed = False
        if os.path.exists(self.index_path):
            index_existed = True
            try:
                with open(self.index_path) as f:
                    self.index = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Corrupt/truncated index (non-atomic saves upstream): rebuild
                # the registry from what's on disk rather than dying at boot.
                self.index = {"files": {}, "sessions": []}
                index_existed = False
        if not index_existed:
            # Missing or corrupt index: rebuild the registry from disk so a
            # lost index never orphans existing workspace files.
            self.index = {"files": {}, "sessions": []}
            for fn in os.listdir(self.files_dir):
                base = re.sub(r"\.r\d+\.txt$", ".txt", fn)
                if base.endswith(".txt"):
                    self.index["files"].setdefault(
                        base, {"status": "draft", "revs": 0, "sessions": 0})
        self.index.setdefault("files", {})
        self.index.setdefault("sessions", [])

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.index, f, indent=1)
        os.replace(tmp, self.index_path)   # atomic — never a half-written index

    # ---------- file registry ----------

    def list_files(self):
        return dict(self.index["files"])

    def file_exists(self, name):
        return name in self.index["files"]

    def read(self, name):
        """Current text of a file ('' if missing)."""
        if not self.file_exists(name):
            return ""
        path = os.path.join(self.files_dir, name)
        if not os.path.exists(path):
            return ""
        with open(path) as f:
            return f.read()

    def _validate_name(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+(\.txt)?", name or ""):
            raise WorkspaceError(f"bad file name: {name!r}")

    def write(self, name, content, rev_bump=False):
        """Write current text. rev_bump=True keeps the previous text as a revision."""
        self._validate_name(name)
        if not name.endswith(".txt"):
            name += ".txt"
        path = os.path.join(self.files_dir, name)
        rec = self.index["files"].setdefault(
            name, {"status": "draft", "revs": 0, "sessions": 0})
        if rev_bump and os.path.exists(path):
            old = self.read(name)
            if old:
                rec["revs"] += 1
                rev_path = os.path.join(
                    self.files_dir,
                    name[:-4] + f".r{rec['revs']}.txt")
                with open(rev_path, "w") as f:
                    f.write(old)
                # Cap kept revisions: drop the oldest beyond MAX_REVS.
                oldest = rec["revs"] - MAX_REVS
                if oldest > 0:
                    drop = os.path.join(
                        self.files_dir, name[:-4] + f".r{oldest}.txt")
                    if os.path.exists(drop):
                        os.remove(drop)
        with open(path, "w") as f:
            f.write(content)
        rec["chars"] = len(content)
        rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_index()
        return name

    def append(self, name, content):
        cur = self.read(name)
        return self.write(name, cur + content)

    def set_status(self, name, status):
        if name in self.index["files"]:
            self.index["files"][name]["status"] = status
            self._save_index()

    # ---------- session log ----------

    def log_session(self, agent, op, file_name, chars, sim_time, note=""):
        self.index["sessions"].append({
            "agent": agent, "op": op, "file": file_name,
            "chars": chars, "sim_time": sim_time,
            "wall": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note[:200],
        })
        self.index["sessions"] = self.index["sessions"][-MAX_INDEX_SESSIONS:]
        self._save_index()

    # ---------- feedback inbox (Ville Dots Phase 4) ----------

    @property
    def inbox_path(self):
        return os.path.join(self.dir, "inbox_notes.jsonl")

    def push_note(self, author, text, sim_time=""):
        """Append an external feedback note to the persona's inbox. The
        Django viewer calls this; the BACKEND ingests notes into associative
        memory in-step (single-writer rule: the frontend never touches
        memory files — checkpoint saves would clobber a live injection)."""
        entry = {"author": author[:40], "text": text[:600],
                 "sim_time": sim_time,
                 "wall": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self.inbox_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def drain_notes(self):
        """Read and clear all pending inbox notes. Backend-side only."""
        if not os.path.exists(self.inbox_path):
            return []
        notes = []
        try:
            with open(self.inbox_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        notes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            os.remove(self.inbox_path)
        except OSError:
            return []
        return notes


# ---------- session prompt + parsing ----------

def session_prompt(name, project, tail, action_desc):
    """Prompt shown to the lane model for one work session. REAL shape — probe
    this on the lane model before deploying (probe protocol, shape #5-style)."""
    p = project or {}
    return f"""You are {name}, sitting down at your personal computer to work on your ongoing project.

Project: {p.get('title', 'untitled')} (type: {p.get('type', 'notes')})
Status: {p.get('status', 'drafting')} — you have worked on this {p.get('sessions_done', 0)} times before.
Your working file "{p.get('file', 'notes.txt')}" currently ends with:
---
{tail}
---
Right now you are: "{action_desc}".

Write the next piece of work. Respond in EXACTLY this format:
OP: <one of OPEN, APPEND, REVISE, NEW_FILE, FINISH>
CONTENT:
<your work this session — for OPEN just a one-line note about what you re-read; for FINISH a one-line summary of the completed piece>"""


_OP_LINE = re.compile(r"^\s*OP\s*:\s*(OPEN|APPEND|REVISE|NEW_FILE|FINISH)\s*$",
                      re.IGNORECASE)


def parse_session_output(raw):
    """Parse the LLM's session response into (op, content).
    Tolerant-but-strict: the OP line is required and must name a valid op;
    CONTENT is everything after the 'CONTENT:' marker. Returns None if invalid
    (caller retries / falls back — same contract as the artifact parsers)."""
    if not raw or not isinstance(raw, str):
        return None
    lines = raw.strip().splitlines()
    op = None
    content_start = None
    same_line_prefix = None
    for i, ln in enumerate(lines):
        m = _OP_LINE.match(ln)
        if m:
            op = m.group(1).upper()
            rest = lines[i + 1:]
            for j, r in enumerate(rest):
                stripped = r.strip()
                if stripped.upper().startswith("CONTENT"):
                    if len(stripped) > len("CONTENT") and stripped[len("CONTENT")] == ":":
                        after = stripped[len("CONTENT") + 1:].strip()
                    else:
                        after = stripped[len("CONTENT"):].strip()
                    # Real lane models put the work on the SAME line as
                    # CONTENT: — capture that remainder plus following lines.
                    same_line_prefix = after
                    content_start = i + 1 + j + 1
                    break
            break
    if op is None or op not in VALID_OPS:
        return None
    if content_start is None:
        # No CONTENT: marker — if the model wrote the op line plus work text,
        # accept the remainder as content (weak-lane tolerance), else it's a
        # bare op line: only valid for OPEN/FINISH which carry one-liners.
        content_start = lines.index(
            [l for l in lines if _OP_LINE.match(l)][0]) + 1
        same_line_prefix = None
    content = "\n".join(lines[content_start:]).strip()
    if same_line_prefix:
        content = (same_line_prefix + ("\n" + content if content else "")).strip()
    if not content and op in WRITING_OPS:
        return None
    return op, content[:MAX_CONTENT_CHARS]


def run_work_session(store, agent_name, project, action_desc,
                     model_call, sim_time=""):
    """One bounded work session against the workspace.

    model_call: callable(prompt) -> str (injected: real lane model in the sim,
    a stub in offline tests).
    Returns (op, file_name) or None on invalid output."""
    file_name = (project or {}).get("file") or "notes.txt"
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    tail = store.read(file_name)[-MAX_TAIL_CHARS:]
    prompt = session_prompt(agent_name, project, tail, action_desc)
    raw = model_call(prompt)
    parsed = parse_session_output(raw)
    if parsed is None:
        return None
    op, content = parsed
    if op == "APPEND":
        store.append(file_name, "\n" + content if store.read(file_name) else content)
        chars = len(content)
    elif op == "REVISE":
        store.write(file_name, content, rev_bump=True)
        chars = len(content)
    elif op == "NEW_FILE":
        base = re.sub(r"[^A-Za-z0-9_\-]", "_",
                      (project or {}).get("title", "untitled"))[:40]
        file_name = f"{base}_{int(time.time()) % 100000}.txt"
        store.write(file_name, content)
        if project is not None:
            project["file"] = file_name
        chars = len(content)
    elif op == "FINISH":
        store.set_status(file_name, "final")
        if project is not None:
            project["status"] = "done"
        chars = len(content)
    else:  # OPEN — re-read, no text change
        chars = 0
    store.log_session(agent_name, op, file_name, chars, sim_time,
                      note=content[:80])
    return op, file_name