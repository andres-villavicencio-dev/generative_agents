# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stanford's "Generative Agents" simulation — AI agents with memory, planning, and social behavior living in a 2D town (Smallville). Forked and extended with needs/resources/economy layers (Phases 1-5) and local Ollama LLM integration.

## Running the Simulation

```bash
# Terminal 1: Django frontend (port 8000)
cd environment/frontend_server && python manage.py runserver

# Terminal 2: Backend simulation
cd reverie/backend_server && python reverie.py
# Prompts: fork name → new sim name → "run <steps>"

# One-command alternative:
python simstart.py --fork base_the_ville_isabella_maria_klaus --steps 100 --auto
```

**Prerequisites:** Ollama with a chat model and `embeddinggemma` (embeddings). The code default is `qwen3.5:2b` on `http://localhost:11434` (`GA_CHAT_MODEL` / `GA_OLLAMA_URL` override). Live runs currently use a dedicated instance — see "Live Run Operations" below.

## Live Run Operations (as of Oct 21 2026)

**Current run: `fresh25_1`** (unit `ga-fresh25`, started Oct 21 2026 15:00) — fresh 25-agent fork of `base_the_ville_n25`, target 25,920 steps (3 sim-days), minimax-m3.1-flash:cloud. All earlier sims (live3d_n17..n68, Apr/Mar tests, compressed_storage) were deleted Oct 21; their `/tmp/*_backend.log` are gzipped. `ga_traces/` (the training data) was kept.

- **Fresh run:** `systemd-run --user --unit <unit> --setenv=... venv/bin/python run_live3d.py <sim> <steps> <fork>` (cold start = day-one planning for all 25 agents, no movement dir until it finishes).
- **Resume from checkpoint:** `run_live3d_resume.py <old_sim> <new_sim> <start_step> <target_step>` (`run N` is relative). Checkpoints every 360 steps.
- Live env: `GA_OLLAMA_URL=http://127.0.0.1:11436 GA_CHAT_MODEL=GA_ARTIFACT_MODEL=minimax-m3.1-flash:cloud GA_THINK_MULT=4 GA_MOVE_WORKERS=12 GA_CALL_TRACE=1 GA_JEV_SCORING=1 GA_LAYA_POIGNANCY=1 GA_RIZZO_POIGNANCY=1 GA_RIZZO_REFLEX=1 GA_DRIVES=1 GA_ROUTINES=1 GA_CURRICULUM=1 GA_CURIOSITY=1`.
- Backend log `/tmp/<sim>_backend.log` is block-buffered (lags minutes) — judge liveness by the movement frontier or `ga_traces/calls/*.jsonl`, not log size. Progress: `logs/<sim>.log` / `journalctl --user -u <unit>`.
- **Viewer:** `/simulator_home` is read-only follow mode by default (`?drive=1` = legacy browser-driven mode; never use it while headless_driver runs). It auto-reloads on sim change, checkpoint rollback, or >60 steps of lag, and polls ~3x/s. The ops card's **Activity** line (from `ga_traces/calls`) shows in-step work — a minutes-long step (conversation + re-plan) reads as busy, red only after 3 min without a completed LLM call.

| Service | Port | Purpose |
|---------|------|---------|
| `ga-ollama` (user unit) | 11436 | Sim LLM endpoint. Proxies `minimax-m3.1-flash:cloud` (free until ~Oct 22) + local `embeddinggemma`. `OLLAMA_NUM_PARALLEL=5`. |
| system `ollama` | 11434 | Default instance; not used by live runs. |
| `ga-laya` (user unit) | 8018 | Laya v3 local fine-tuned decision server (poignancy / talk / react). POST only. |
| Django `runserver` | 8000 | Frontend; two PIDs is normal (autoreloader parent + worker). `/api/run_stats/<sim>/` feeds the simulator_home ops card (rate = steps written in a trailing 10-min window; unit resolved from the backend's cgroup). |
| `~/Scripts/m31_teacher_harvest.py` (transient unit `m31-teacher`) | — | Blind teacher-labels `ga_traces/{poignancy,talk,react}.jsonl` with minimax → `ga_traces/m31_teacher/`. Paced `M31_CALL_GAP_S=1.5`s between calls: unpaced batches shared the cloud's **per-minute quota** with the sim and caused 429 storms. `ga-ollama` stays at `OLLAMA_NUM_PARALLEL=5` — 8 produced more 429s. |

**Purpose of the minimax runs:** harvest cloud-teacher data to fine-tune the local sLM. Data products:
- `ga_traces/{poignancy,talk,react}.jsonl` — sim decision traces (`ga_trace.py`).
- `ga_traces/m31_teacher/*.jsonl` — blind teacher labels for those.
- `ga_traces/calls/<model>.jsonl` — **every** chat call (prompt, system, content, thinking, tokens, latency, caller, persona) (default on; `GA_CALL_TRACE=0` disables — `gpt_structure._trace_call`). Includes empty/overflow attempts — filter on `content`/`done_reason`. Rotates to `<model>.partNNN.jsonl` at `GA_CALL_TRACE_ROT_MB` (default 20): the data is committed, and GitHub rejects files > 100 MB (warns > 50). Rotating by hand while the sim runs is safe (the file is reopened per write) — just `mv` it to the next part number.

**Performance facts (measured Oct 21, n68):**
- minimax **always reasons, even with `think=false`** (~300-2000 hidden tokens per call). p50 latency ~7s, p90 ~35s, worst ~100s.
- Steps are a barrier over all personas (`ThreadPoolExecutor(GA_MOVE_WORKERS)` in `reverie.py`); the slowest persona sets step time.
- Conversations were the bottleneck: generation ran entirely under the global `CONVO_LOCK` (~24 serial calls) → whole town frozen ~5-10 min per chat, ~1.6 mean calls in flight. Fixed: lock now covers only the decision + claiming both speakers (`concurrency_utils.CHAT_BUSY`); relationship summary computed once per side per chat (`converse.agent_chat_v2`).
- `GA_THINK_MULT` scales `num_predict` (base 2048). At 2 (4096) minimax overflowed often, wasting a full attempt before the 2x retry; use 4.
- The endpoint handles ≥12 concurrent calls (queues at NUM_PARALLEL=5); `GA_MOVE_WORKERS=12` is safe.
- **Format-retry storms were the other big cost** (found via the call log): `_ollama_generate` ran the single-line YAML unwrap on `free_form` answers too, cutting every task decomposition to its first line → validation failed → 5 retries × ~40s → fail-safe anyway (45% of all call time). Fixed (free_form skips the unwrap) + `parse_loose_task_decomp` accepts minimax's markdown/numbered/table/time-range shapes. Focal points: minimax answers YAML block lists (`output:` + items) → now returned as a JSON list string; previously all 3 retries failed into `["Who am I"]*n`.
- **JEV (`GA_JEV_SCORING=1`) is served by laya** (`GA_JEV_BACKEND=laya`, default): `jev_decide` / `jev_score_choice` → laya `choice`, `jev_score_digit` → laya `score` (poignancy rubric). `GA_JEV_LAYA_MODE=shadow` (default) asks laya off-thread and traces to `ga_traces/{spatial,jev_decide,jev_digit}.jsonl` while the teacher still decides; `live` lets laya decide. Shadow is the default because laya-v3 is untrained on spatial choice (Oct 21 eval vs minimax: sector 6/11 with confident misses, objects 6-7/8, arenas 2/2) and poignancy/talk/react only reach JEV after laya already deferred. Pair shadow traces with `ga_traces/calls` (persona + timestamp) for spatial fine-tuning data. `GA_JEV_BACKEND=ollama` = legacy logprob path; ollama returns no logprobs for `:cloud` models, so a breaker disables it after 3 misses.
- `new_decomp_schedule` (re-plan after every chat) never validated for ANY model: the validator read the time window from the prompt's first line, which `generate_prompt` now fills with "IMPORTANT: Respond in English only" → 5 retries (~60s each with minimax) per participant per chat. Fixed (window via `_SCHED_WINDOW_RE`) + `parse_revised_schedule` tolerates minimax's restated/markdown schedules and makes them contiguous over the window.
- Task names could arrive exponentially nested (`X (X) (X (X)) ...`); `run_gpt_prompt_task_decomp` now cleans its input task with `_clean_action_text`.
- `wake_up_hour` parser did `int(resp.split("am")[0])` → failed on "9:00 AM" → every agent got the fail-safe 8am after 5 retries. Fixed (regex, am/pm aware).
- Drives had no time-of-day sense: on an empty event history curiosity won at 00:00 and sent agents "exploring" at midnight. Fixed: while the scheduled activity is sleep, only hunger/energy drives inject goals (`plan.py`, `_determine_action`).
- Emoji (`pronunciatio`) never uses the LLM by default: `EMOJI_LOOKUP` in `plan.py` (incl. object states) with a 🙂 fallback; `GA_PRONUNCIATIO_LLM=1` restores the LLM fallback. Purely visual — no effect on cognition.
- **Unified location** (default, `GA_UNIFIED_LOCATION=0` = legacy chain): `resolve_action_location` (plan.py) → `run_gpt_prompt_action_location` sends the persona's whole accessible spatial tree and gets `output: <area> | <room> | <object>` in ONE call (2-5s vs 3 serial calls); unvalidated levels fall back to the legacy per-level calls. Laya still shadows all three choices (`_laya_shadow_location`).
- **Object state templated** (default, `GA_OBJ_DESC_LLM=1` = LLM): act_obj_desc → "being used"/"being slept in"/"being cleaned"/"being fixed", triple `(object, is, state)`.
- **Async conversations** (default, `GA_ASYNC_CHAT=0` = in-step): a chat decided in step N is queued (`concurrency_utils.PENDING_CHATS`), started after the step's pool on its own thread, and both speakers are frozen (move skipped, 💬, `ASYNC_FROZEN`) until it's written back; the town keeps stepping. `save()` joins chat threads first.
- `GA_JEV_LAYA_LIVE_KINDS=sector,arena,object` lets laya decide those kinds on the legacy per-level path (default empty; object JEV is skipped anyway while the tool registry is loaded).
- **task_decomp few-shot leak**: `task_decomp_v3.txt`'s single worked example (Kelly's kindergarten lesson plan) leaked into ~27% of minimax decompositions of unrelated activities ("eat breakfast" → curriculum/worksheet steps). `task_decomp_v4.txt` (format-only labelled neutral example + the real activity restated with rules before the answer) leaked 0/12 on the same inputs (v3: 12/12). Exclude v3-era contaminated records from fine-tuning sets via `python pipelines/flag_decomp_leak.py` → `ga_traces/calls/flags/task_decomp_example_leak.jsonl` (join on ts + persona).
- Diagnose slowness from `ga_traces/calls/*.jsonl`: group by `caller` for latency share; identical `(caller, prompt)` pairs = validation retries.

**Restart gotchas:**
- Restarting a transient unit = `systemctl --user stop <unit>` then a new `systemd-run` (plain `systemctl start` fails: the transient unit file is gone after stop). `n68_stall_watch.py`'s self-heal uses `start` and therefore cannot actually restart.
- Before restarting, delete `movement/` and `environment/` files above `reverie/meta.json`'s `step` (checkpoint). Don't fork a shell per file — there are ~90k files; use a single Python/`find` pass.
- Resuming in place (`run_live3d_resume.py <sim> <sim> <ckpt> <target>` + `GA_FORK_SKIP_COPY=1`): the supervisor waits for the backend's "stale-file cleanup complete" marker before starting the driver. The backend log is appended across restarts, so a stale marker from the previous boot used to release the driver early → driver wrote `environment/<ckpt+1>`, backend cleanup deleted it → silent deadlock at the checkpoint step. Fixed (marker searched only past the log offset at launch). Symptom if it recurs: frontier frozen at the checkpoint, no LLM calls; killing the driver (supervisor restarts it) unblocks.
- Other agents (Hermes) run watchers that treat "unit inactive" as completion; any stop triggers a false COMPLETE report.

**Known behavior issues (n68, sim day Feb 23 — watch whether they recur in fresh25_1):** daytime sleeping/napping at 11am, "prepare dinner" before noon, clock times leaking into object/book names ("12:00 am Logic Stirrer"), activity/location mismatch (reading at the bathroom sink), references to non-existent people ("Jamie"), Isabella with "no café service scheduled".

## Running Tests

```bash
python test_phase5_economy.py          # Economy layer tests
python -m pytest tests/                # Resource contention tests
```

## Architecture

**Two-process system communicating via JSON files on disk:**

- **`reverie/backend_server/reverie.py`** — `ReverieServer`: step-based simulation loop. Manages personas, advances time, writes movement/environment files to `environment/frontend_server/storage/<sim>/`.
- **`environment/frontend_server/`** — Django app serving an HTML5 Canvas visualization. Reads movement files via AJAX, renders sprites.

**Forking model:** Every simulation forks from a base (e.g., `base_the_ville_isabella_maria_klaus`). Files are copied, then the new sim diverges.

### Agent (Persona) Architecture — `reverie/backend_server/persona/`

Three-layer cognitive loop per step: **perceive → retrieve → plan → execute → reflect**

| Layer | File | Purpose |
|-------|------|---------|
| Short-term memory | `memory_structures/scratch.py` | Current goals, beliefs, action queue |
| Long-term memory | `memory_structures/associative_memory.py` | Events/thoughts/chats with embeddings |
| Spatial memory | `memory_structures/spatial_memory.py` | World→sector→arena→object hierarchy |
| Cognition | `cognitive_modules/perceive.py, retrieve.py, plan.py, reflect.py, execute.py, converse.py` | Each step of the cognitive loop |
| LLM interface | `prompt_template/gpt_structure.py` | Ollama calls with JSON schema constraints |
| Prompt templates | `prompt_template/run_gpt_prompt.py` | ~1278 lines of prompt construction |

### World & Resources — `reverie/backend_server/`

- **`maze.py`** — 2D tile map with collision detection (Tiled editor format, collision block ID `32125`)
- **`path_finder.py`** — A* pathfinding
- **`resource_manager.py`** — Phase 1-5 extensions: need meters (hunger/energy), consumable resources, shared resource locks (bathrooms), economy (wallets, prices, café revenue, financial stress)

### Frontend Routes

| Route | Purpose |
|-------|---------|
| `/simulator_home` | Live simulation canvas |
| `/demo/<sim>/<step>/<speed>/` | Playback recorded simulation |
| `/replay/<sim>/<step>/` | Debug replay with agent inspection |
| `/api/needs/<sim>/<agent>/` | Agent needs API (Phase 1) |
| `/api/resources/<sim>/` | World resources API (Phase 2) |

## Key Configuration

- **`reverie/backend_server/utils.py`** — API keys, asset paths, storage paths, debug flag
- **`reverie/backend_server/persona/prompt_template/gpt_structure.py`** — Ollama URL, model names, retry logic
- **`environment/frontend_server/frontend_server/settings/`** — Django settings

## File-Based Communication Protocol

Backend writes per-step JSON files to `storage/<sim>/reverie/`:
- `environment/<step>.json` — agent positions and actions
- `movement/<step>.json` — sprite movement paths
- `curr_step.json` — current step number (frontend reads on refresh)

Frontend polls these files via AJAX. On fork, stale files from the parent sim beyond the starting step must be cleaned up to prevent desync.

## Simulation Storage Layout

```
environment/frontend_server/storage/<sim_name>/
├── reverie/meta.json                    # start date, step count, maze name
├── reverie/environment/<step>.json      # per-step environment state
├── reverie/personas/<Name>/bootstrap_memory/  # agent initial state
└── resources/world_state.json           # item quantities, capacities, prices
```

## Compression & Playback

```bash
python compress_sim.py          # Compress sim → compressed_storage/
python headless_driver.py <sim> <step>  # Headless animation render
```
