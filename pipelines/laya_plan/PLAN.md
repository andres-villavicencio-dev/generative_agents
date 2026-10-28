# Moving decisions from the LLM to laya — working plan and results

Principle: laya (typed `choice`/`score` encoder, ~80 ms server-side p50 on the GPU) takes every call whose
answer comes from an enumerable set; the LLM keeps free text.

## Phase 0 — census (done; `census.py`, `census.md`, `census.json`)
- 20.6k calls / 1,120 LLM-minutes over 7 trace files. Laya server compute: p50 83 ms, p90 152 ms
  (journal of ga-laya-v5 under load) vs 1.3–9 s per LLM call.
- Time by caller: action_location 24.4%, choose_activity 23.8% (now laya), utterances 12.3%, re-plan 9.3%,
  task_decomp 7.1%, insight 6.3%, artifacts/owner prompts ("?") 5.2%, everything else < 2.5% each.

## Phase 1 — tiers (evidence-based)
| tier | calls | why |
|---|---|---|
| 1 closed answer | action_location (area|room|object picked from the prompt's own list), choose_activity (done), wake_up_hour, action_arena/game_object/sector (legacy), poignancy (done), talk/react (done), commitments, owner yes/no | answer is one of the listed options or a small number |
| 2 structured, option-izable | task_decomp, new_decomp_schedule, insight_and_guidance | retrieve k cached past answers, laya ranks / picks |
| 3 free text (LLM) | utterances, summaries, memo/planning thoughts, focal points, daily plan, artifacts | |
- Label audit for action_location (31.6k distinct calls, 97.6% valid against their own option lists):
  mean 10 areas per prompt (max 18); 82.6% of answers stay in the current area; identical prompts re-asked
  agree on the full triple only 41% of the time (object-level equivalents), so object accuracy has a low ceiling
  and is judged "acceptable object", not exact match.
- Risks: 1,024-token context, label noise, compounding errors, new buildings (labels by option text, never ids).

## Phase 2 — pilots (shadow / offline)   Phase 3 — gating   Phase 4 — distillation
See the sections appended by each phase below.

## Phase 1 addenda — Tier 2 evidence
- task_decomp (7.1% of LLM time): 3,320 calls over 928 distinct (activity, duration); online hit rate of a
  (normalised activity, duration) cache = 72% (64.7% after dropping answers that name other people or mix
  genders). -> `decomp_cache.py` (GA_DECOMP_CACHE=1), seeded by `decomp_build.py`; LLM only on misses.
- insight_and_guidance (6.3%): numbered free-text reflections that cite memory numbers -> Tier 3, stays on the LLM.
- new_decomp_schedule (9.3%): revises the day around a specific chat -> not cacheable, stays on the LLM.
- Tier 1 leftovers (wake_up_hour 0.5%, arena/game_object/sector 0.5%, commitments 0.1%) are not worth a model each.

## Phase 2 — pilots (results)
- Laya v6 (v5 + `loc` area/room/object items from 6,000 action_location calls), held-out persona-disjoint:
  area 94.6%, room 93.9%, object 73.4% top-1 vs the LLM's own repeat agreement 81.1% / 86.3% / 43.9%.
- Live shadow on the 200x130 map (35 agents, 247 decisions): area 98%, area+room 93.5%, full triple 64%;
  12/13 decisions touching the new buildings agree on the area.

## Phase 3 — gating (results)
- Gate = per-level margin threshold (`loc_thresholds.json`, chosen so accepted accuracy >= the LLM's own repeat
  agreement). Laya already clears that at margin 0, so all thresholds are 0.0 (no fallback unless a laya call fails).
- Live run, 35 agents, 500 steps, `GA_LOC_LAYA=live GA_DECOMP_CACHE=1` vs shadow: wall 13.0 -> 9.0 min (-31%),
  LLM minutes 21.7 -> 6.8 (-69%), action_location LLM calls 248 -> 0, task_decomp 44 -> 13, no tracebacks,
  A-B-A shuffle 1 -> 2 steps.

## Phase 4 — distillation (round 1)
- `distill.py`: harvest -> train 1 epoch -> evaluate incumbent vs candidate on the same held-out set -> promote
  only if no level drops > 0.5 pt and the mean improves.
- Round 1 (~250 new shadow-run calls): candidate area 94.2 / room 92.1 / obj 73.9 vs incumbent 94.6 / 94.2 / 73.3
  -> not promoted (guard worked). Incumbent stays v6. Next rounds need more new LLM-labelled calls
  (e.g. a gated run with a threshold > 0 so declined decisions fall back to the LLM and become labels).
