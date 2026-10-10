# Ville Dots — Per-Agent Simulated PCs (Development Plan)

**Inspiration:** OpenAI dots — each agent is a persistent "dot" with its own computer,
long-term projects it works on over days, and an inspectable workspace.
**Our translation:** every GA persona gets a *simulated PC* — a persistent per-agent
workspace where projects live, evolve across sim-days, and can be inspected at any time.

Status: PLAN (v1 draft, Oct 2026). Frontier sim at time of writing: `live3d_n45`.

---

## 0. What dots has vs. what we already have

| dots concept | Our GA stack today | Gap |
|---|---|---|
| Own cloud computer, isolated | Per-sim storage trees, fully isolated already (it's a sim) | No per-agent *workspace* — nothing a persona "owns" and returns to |
| Works on long-term projects 24/7 | Daily plans, `artifact_progress` accumulation | Artifacts are one-shot: create at threshold → reset → next blob. No multi-day evolving project |
| Open the dot's computer anytime | `/artifacts/<sim>/` gallery (finished objects only) | No live workspace view: no file tree, no drafts, no version history, no activity log |
| Uses its browser/apps (tools) | One-shot artifact generation prompts | No file-operation vocabulary: agents can't open/read/append/revise/save |
| Learns from feedback | Associative memory + reflection (works well) | No external feedback channel into a persona's memory |
| Approval boundaries | n/a (nothing leaves the sim) | Mostly free — the sim store IS the sandbox |

**Core design decision:** build the PC as a *workspace layer* on top of the existing
artifacts machinery (stores, gallery routes, lanes) rather than a new subsystem. The
cognitive change is small and contained: replace "generate artifact blob" with
"run a bounded work session against the workspace."

---

## 1. Architecture (five layers)

### Layer 1 — Workspace store (`ville_os.py`, new module)
Per-persona virtual FS inside the sim tree (checkpointed, forked, and archived with it):

```
storage/<sim>/personas/<Name>/workspace/
├── index.json          # file registry: name, type, status(draft/final), rev count, sim-time stamps
└── files/
    ├── novel_ch4.r2.txt    # plain text; revisions kept, capped
    └── ...
```

- Files are text-first (books, letters, lyrics, code-ish sketches, study notes) —
  same artifact types we already generate, plus notes.
- Revisions: keep last **3** full-text revisions per file (cap or diff-store if disk
  becomes an issue at 25 agents; measure first).
- All writes go through one `WorkspaceStore` class with an API:
  `list_files, read(name), write(name, content, rev_bump), open_project, close_session`.
- **No host contact ever** — it lives and dies with the sim tree. That's the dots
  "separate cloud computer" guarantee for free.

### Layer 2 — Project state on scratch (`scratch.json` extension)
New checkpointed key: `active_projects`:

```json
"active_projects": [{
  "title": "The Lighthouse Murders",
  "type": "book",
  "target_revs": 8,
  "file": "lighthouse_ch1.txt",
  "status": "drafting|revising|done",
  "sessions_done": 12
}]
```

- Checkpointed like `artifact_progress` (schema change → offline `Persona()` load gate
  before any deploy; invent NO keys — clone real node/file schemas).
- Daily planning can read it: "continue the novel" becomes a plan item that survives
  sim-days. **This is what makes a persona a dot** — persistent goals, not one-shots.

### Layer 3 — Work-session loop (the cognitive core)
When a persona performs a PC-anchored creative action ("working on laptop", "writing"),
run a bounded session instead of only accumulating progress:

1. Prompt the lane model with: project title + status + current file tail (~last 800 chars)
   + a **file-op vocabulary**: `OPEN / APPEND / REVISE / NEW_FILE / FINISH`.
2. Constrained output format (cheap to validate, like the artifact `|||` 4-liner):
   one op line + payload. Retry/validation machinery reused as-is.
3. Execute the op against `WorkspaceStore`; log a session entry (op, chars written,
   sim-time) to `workspace/index.json`.
4. `FINISH` → promote file to `final`, feed the existing artifact pipeline
   (deep-creation store + gallery get their content from the workspace now).

**Budget guard:** sessions fire at most every N matching actions (e.g. 3rd action →
1 session) so step rate doesn't crater. New env knob: `GA_PC_SESSION_EVERY=3`.
All calls route the existing artifact lane (`GA_ARTIFACT_MODEL`) on the sim's ollama
daemon — **no new GPU-resident service** (VRAM rule: 8GB card is fully subscribed).

### Layer 4 — Frontend: "open the dot's computer"
- Django route `/pc/<sim>/<agent>/`: file tree, doc viewer, revision slider,
  session activity log. Reuse gallery plumbing (`views.py`, `content_full`
  largest-text fallback for JSON-wrapped prose).
- Pitfalls already on file: bootstrap.js AFTER jquery; verify a modal actually OPENS,
  not just that the page renders.
- Optional RichFX: an emote/desk-glow when a persona is in a PC session; later, click
  a sprite → PC panel.
- Phase-2 stretch: world tiles with desks (maze + spatial memory arena entries — a
  venue without an arena+rooms entry cannot be used, per staffing rules).

### Layer 5 — Feedback channel ("message your dot")
- On the PC page: a small "leave a note" form → injects an event node into that
  persona's associative memory (real schema, real embedding on the sim daemon).
- This is the dots "learn from feedback" loop; reflection already does the rest.
- Injection must happen **backend stopped** (memory files are rewritten at checkpoint
  saves from the in-memory roster — live injections get silently overwritten).

---

## 2. Phasing & acceptance gates

**Phase 0 — offline spike (½ day)**
- Build `WorkspaceStore` + session executor with stub persona; drive with REAL action
  strings harvested from n45 movement files. Fresh kernel per test run.
- Gate: session op executes, revision history correct, no schema drift.

**Phase 1 — backend integration (1–2 days)**
- scratch `active_projects` + matcher for PC actions (word-stem regex, negative
  guards, order-sensitive — rebuild from harvested n45 phrasings, not imagined ones).
- Seed 2–3 pilot personas with projects matching their identity (novelist, musician).
- Gate: offline `Persona()` load passes; then deploy to a fresh fork via the
  checkpoint-watcher zero-loss protocol; verify with `tr '\0' '\n' < /proc/<pid>/environ`.
- Acceptance: a pilot completes a multi-sim-day project with ≥5 revisions and a
  coherent evolution (chapter text actually progresses, not regenerate-from-scratch).

**Phase 2 — PC viewer (1 day)**
- `/pc/<sim>/<agent>/` route with tree + viewer + history.
- Acceptance: browse any pilot's workspace from the browser, revision slider works,
  modal opens (click-tested, not just rendered).

**Phase 3 — world wiring (½–1 day)**
- Desk/PC tiles in pilot homes; spatial memory entries; sprites visibly at desks.
- Acceptance: frontend shows pilot at desk during sessions; movement files carry it.

**Phase 4 — feedback + polish (½ day)**
- Note form → memory injection (backend-stopped protocol).
- Acceptance: note appears in the persona's next reflection/behavior; end-to-end demo.

---

## 3. Risks & known traps (from prior runs — all bit us before)

- **Silent-None matcher disease:** every keyword→behavior dispatch uses word-stem regex
  at word boundaries; validate against harvested real action phrasings offline before
  restart. Absence of error lines proves nothing — matchers fail silently.
- **Poison-wrap:** every site where LLM output becomes a stored string gets
  `_clean_action_text` (grep all entry points — a third was found after the first fix).
- **Validation retry storms:** probe the session output format on REAL prompt shapes
  with the lane model before deploying; small budgets eat reasoning tokens → probe ~300.
- **Compile-clean ≠ call-clean:** after every patch, CALL the changed function offline
  with a stub persona and assert output structure before restart.
- **Checkpoint save race:** scratch mutations gate on meta step == scratch curr_time − 10s;
  deploy only via the one-script background watcher (checkpoint wait → save-race gate →
  unit swap → boot verify).
- **Restart=on-failure crash-loops mask first error:** read the FIRST traceback in the
  backend log; `rm -rf` failed fork dir + `reset-failed` before re-fork.
- **Step-rate tax:** work sessions add LLM calls — A/B step rate on a fresh-fork window
  (phase-aligned!) before calling the `GA_PC_SESSION_EVERY` default permanent.
- **Disk:** 25 agents × multi-MB revisions × weeks of runs — cap revisions at 3,
  measure per-sim-day growth in Phase 1, prune like movement files if needed.
- **No sed edits, no bare `pkill -f`, numeric sort only** — house rules.

---

## 4. Explicitly out of scope (v1)

- Real network access / "connected apps" — the whole point of our version is a fully
  sandboxed simulated computer; no host reach, no passwords, no approvals needed.
- Out-of-band (out-of-step) background work — scratch is rewritten from the in-memory
  roster at checkpoints, so out-of-band mutation would be silently lost. Work happens
  in-step. A cron-style "night shift" runner is a v2 idea only.
- Multi-agent shared drives / collaboration — v2+ ("dots working together").

## 5. Open decisions (defaults chosen, flag if wrong)

- Pilot roster: 2–3 personas, chosen for identity fit (novelist, musician, student).
- Revision storage: full text ×3 (simple) vs diffs (compact) — start full-text, measure.
- Session model: reuse `GA_ARTIFACT_MODEL` lane (no new lane, no new VRAM).
- PC viewer styling: simple Django page first; RichFX integration later.