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

## Phase 5 — v7 retrain on the 3070 (Nov 2 2026)
- Data: v6 used ~6k of 31.6k valid labelled action_location calls. v7 uses 21k calls (cap 8 per persona+activity) =
  49k train items, 14k persona-disjoint test items. Second teacher `glm-5.3-flash:cloud` (`glm_loc_label.py`, blind,
  12k prompts, 9.9k valid; it answers prose + `</think>output:` so parse the LAST `output:` match, num_predict 2048).
  `loc_build.py` consensus targets: GLM agrees -> 0.93, disagrees -> 0.55/0.30 split, no GLM label -> 0.85.
- v6 baseline on the v7 test set: area 94.5 / room 93.9 / obj 74.0. v7a (qwen labels + first 1.5k GLM): 94.3 / 94.2 / 73.7
  -> NOT promoted (mean not improved). Sliced by label quality (`loc_eval_consensus.py`): where qwen==GLM, room is 96.6 (v6)
  vs 97.0 (v7a); the errors live in the qwen!=GLM slice (room 73%), i.e. genuinely ambiguous items. Room accuracy is at the
  label-noise ceiling; more data of the same kind does not move it.
- Census of recent traces (43.6k calls): remaining closed-answer callers are wake_up_hour (2.5% of time), action_game_object /
  action_arena / commitments (<1% each); choose_activity (9.5%) is already laya. Everything else is free text.
- wake_up_hour needs no model: the answer is stated in the persona's lifestyle line. Persona-mode agreement 99.6%; a regex on
  "awakes up around Nam" equals the LLM answer on 853/856 calls. Implemented as a rule fast path in
  `run_gpt_prompt_wake_up_hour` (`GA_WAKE_RULE`, default on, =0 restores the LLM).
- v7b (v6 init, 2 epochs, 49k items, GLM-consensus targets from 9.9k valid GLM labels), same 14k test set: all-items
  94.5 / 93.7 / 75.2 vs v6 94.5 / 93.9 / 74.0; qwen==GLM slice room 96.9 = 96.9, obj 91.4 vs 88.6; vs-GLM room 87.9 vs 87.0,
  obj 72.5 vs 68.9. Guard passed (no level -0.5, mean up) -> `ga-laya-v6.service` now serves `laya-ga-finetuned-v7b`
  (rollback: set LAYA_CKPT back to `laya-ga-finetuned-v6`). Gain is in OBJECT choice; room stays at its label-noise ceiling.
- Error analysis of v7b room misses: 228/256 are in the artist's co-living space (own room vs common room vs kitchen vs bathroom);
  there qwen and GLM agree on the room only 77.7% (88.8% elsewhere) -> label noise, not a model gap.
- v7c (v7b init, 1 epoch, teacher-disagreement items DROPPED, co-living room/object items x3; `LOC_DROP_DISAGREE=1 LOC_COLIVING_DUP=3`):
  room on qwen==GLM items 97.8 (v6/v7b 96.9), co-living clean 96.1 (94.7), room vs GLM labels 90.0 (87.0 v6), obj all 75.1;
  but room vs raw qwen labels 93.0 (93.9 v6) -> fails the "no level -0.5 vs qwen" guard, so NOT deployed. Checkpoint:
  /home/andus/Projects/laya-test/laya-ga-finetuned-v7c. Deploy it if consensus labels are the preferred yardstick.

## Phase 6 — 3-teacher vote labels (v7e), 2026-11-03
- Teachers: glm-5.3, minimax-m3.1-flash, nemotron-3-ultra (qwen discarded as faulty; judge on cloud labels only). Pairwise room agreement 90-93% among the cloud models vs ~85% with qwen. 4198 prompts with all 3 valid votes.
- v7e = v6 init, 3 epochs, `LOC_VOTES=1 LOC_COLIVING_DUP=3 LOC_CAP=8 loc_build.py 40000` (v7e_data: 5707 loc train items, persona-disjoint 6-persona test).
- Vote-label test (area/obj/room): v6 90.9/79.6/84.0, v7b 91.1/80.9/86.3, v7d 90.5/79.3/87.8, **v7e 95.3/89.6/97.1**. Promoted to ga-laya-v6.service (rollback v7b/v6).
- Fill note: cloud labelling shares ga-ollama's queue with the sim (NUM_PARALLEL=5); it crawled while n55 ran.
