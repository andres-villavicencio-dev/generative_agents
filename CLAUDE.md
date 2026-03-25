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

**Prerequisites:** Ollama running locally (`http://localhost:11434`) with models `qwen3.5:9b` (chat) and `embeddinggemma` (embeddings).

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
