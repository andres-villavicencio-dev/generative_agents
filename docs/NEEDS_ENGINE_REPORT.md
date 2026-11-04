# Needs engine report (2026-11-03)

Question: why do only bladder, social and status ever go down, while hunger, energy, hydration, hygiene, comfort and stimulation sit near 100 on every agent?

Evidence: n55 at checkpoint 51120 (55 agents) plus the movement files for sim day Feb 18 (steps 43200-51840). Code read: `reverie/backend_server/reverie.py` (`tick_needs`, `satisfy_needs_for_action`), `persona/memory_structures/scratch.py`, `resource_manager.py`, `persona/cognitive_modules/plan.py`, `agentic_drives.py`.

## 1. Summary

Decay works as designed. The cause is relief: every time an agent's action description changes, `satisfy_needs_for_action` adds a lump of relief if the text contains a keyword. Because the planner splits every activity into 5-minute sub-tasks, relief fires dozens of times a day, and the amounts are large next to the decay. Six needs are therefore pinned at the 100 clamp. The three that move are the ones whose relief is rare or absent.

## 2. How the engine works

- `scratch.needs`: nine meters, 0-100 (100 = satisfied). Start random 40-80 for status, high for the rest.
- Decay per 10-second tick (`scratch.py:190`): hunger 1/180, energy 1/270, hydration 1/120, hygiene 1/360, bladder 1/150, social 1/270, comfort 1/540, stimulation 1/360, status 1/540. Scaled by a per-persona multiplier (`PERSONA_NEED_BASELINES`, `resource_manager.py:932`); only 3 personas have custom values, everyone else is 1.0.
- `tick_needs` (`reverie.py:822`): hunger, hydration, bladder decay at 30% while asleep; energy decays awake and regains +1/60 per tick asleep; social decays unless chatting; hygiene, comfort, stimulation, status pause while asleep.
- Relief: `satisfy_needs_for_action` (`reverie.py:1764`), called when `act_description` changes (`reverie.py:2035`). Keyword substring match on the description text. Plus meal artifacts (`ARTIFACT_TYPES`), coffee machine (`tool_registry.py:208`), staff meal (`economy.py:580`), luxury goods (status +70).
- Consumers: `plan.py:1605` emergency goal only when a need is below `needs_critical` = 20; `agentic_drives.py` uses deficit continuously; Laya/JEV prompts mention only bladder < 20 and critical needs.

## 3. Measurements

Needs across 55 agents at n55 checkpoint 51120:

| need | mean | min | agents < 20 | agents < 50 |
|---|---|---|---|---|
| hunger | 96.5 | 82 | 0 | 0 |
| energy | 98.2 | 94 | 0 | 0 |
| hydration | 94.8 | 73 | 0 | 0 |
| hygiene | 96.5 | 82 | 0 | 0 |
| bladder | 72.5 | 22 | 0 | 12 |
| social | 89.5 | 69 | 0 | 0 |
| comfort | 98.9 | 96 | 0 | 0 |
| stimulation | 99.0 | 95 | 0 | 0 |
| status | 7.8 | 0 | 49 | 52 |

Relief versus decay, per agent per sim day (Feb 18, sampled 1:3, so a lower bound; about 128 action changes per agent per day):

| need | relief hits/day | amount | relief/day | decay/day |
|---|---|---|---|---|
| hunger | 38.6 | +40 | ~1540 | ~37 |
| hydration | 14.1 | +30 | ~420 | ~55 |
| hygiene | 6.4 | +60 | ~380 | ~16 |
| bladder | 1.6 | +70 | ~110 | ~44 |
| comfort/energy | 20.2 | +30/+15 | ~600 | ~11 |
| social | 22.2 | +25 | ~550 | ~21 |
| stimulation | 24.7 | +20 | ~490 | ~16 |

Relief exceeds decay by 10x to 40x for everything except bladder, and values clamp at 100, so excess is simply discarded.

## 4. Root causes

1. **Lump relief per action change, not per time.** Task decomposition produces many sub-tasks per activity ("eating breakfast: set the table", "...: pour coffee"). Each one re-triggers +40 hunger.
2. **Relief amounts are huge against decay.** One meal restores 40 points; hunger loses 1 point per 30 sim-min, so one hit covers 20 hours.
3. **Relief is decoupled from reality.** It fires on the description alone: no food needs to exist, the agent need not have arrived, the fridge can be empty, and a broke agent still gets +40. The real food economy (fridge, grocery, cafe sales) never feels hunger. Hunger therefore cannot drive purchases, jobs or trips to venues. Planner-driven visits are the only demand.
4. **Energy cannot fall.** Sleeping restores +6 per hour; waking decays 1.33 per hour. 8 h of sleep gives +48 against ~21 lost in 16 h awake.
5. **Keyword substring matching has false positives**, but this is secondary. Share of unique (agent, action) pairs that matched only as a substring: comfort/energy 59% ("visit" contains "sit", "restock" contains "rest", "position"), stimulation 36% ("bread" contains "read", "display"), hygiene 16% ("bathroom" contains "bath"), hunger 16% ("cookout", "creative" contains "eat"), hydration 18% ("watercolor", "teaching"). Even the true-word matches alone (1,734 hunger hits) keep hunger saturated, so fixing matching on its own will not unpin the needs.
6. **Why bladder, social, status do move.** Bladder relief needs "toilet/bathroom/restroom", which appears about 1.6 times a day. Status relief exists only through a luxury purchase that agents cannot afford. Social relief is frequent (22 hits/day) but is the one the user sees falling most because its decay pauses only while chatting and its relief only gives +25; it still averages 89.5.
7. **The behavioural hooks need low values.** The emergency goal at < 20 has never fired for hunger, energy, hydration or hygiene in these runs, and the drive deficits stay near 0. The needs layer therefore has almost no influence on behaviour except bladder.

## 5. Options for adjustment

Each is independent; they compose. None changes the economy, jobs or luxury rules directly.

**A. Tie relief to real satisfaction (recommended first).**
Apply hunger/hydration relief only when `consume_resources_for_action` or a purchase/meal artifact actually succeeds at the location; hygiene only at a bathroom with hot water; bladder only at a bathroom. Effect: hunger becomes a real demand signal for food, cafe and grocery. Risk: poor agents can starve; keep the existing staff-meal safety net.

**B. Relief per episode and over time, not per sub-task.**
One relief per (need, activity episode) with a cooldown (for example hunger once per 2 sim-hours), or relief as a rate per tick while the activity lasts. Effect: breaks the 38-hits-a-day inflation.

**C. Rebalance rates so need curves have a daily rhythm.**
Targets (awake): hunger empties in ~16-20 h, hydration ~10 h, bladder ~6-8 h, hygiene ~36 h, energy ~24 h awake, comfort ~48 h, stimulation ~30 h, social ~36 h. Meal relief ~+50, drink +20, shower +70. Energy sleep recovery ~+4 per hour. Calibrate from logged time series (G).

**D. Replace keyword matching with a classifier.**
Cheapest: word-boundary regex with stems. Better fit for the Laya goal: a typed Laya decision ("which needs does this activity satisfy, and how much") trained on cloud-teacher labels over the 6,696 distinct actions already logged. This is a concrete new decision type for the "other aspects" goal.

**E. Persona variance.**
Derive `*_decay_mult` and starting values per persona from traits (only 3 personas have them today), so agents differ in appetite, sociability and stamina.

**F. Let more needs matter to behaviour.**
Add comfort, stimulation and social to the urgent map and to Laya/JEV prompts once they actually fall; keep status excluded by design.

**G. Instrument first.**
Log every agent's needs each sim-hour to `ga_traces/<run>/needs_timeseries.jsonl`. Needed for calibration, and it separates "saturated" from "balanced".

## 6. Suggested plan

1. G (logging only, no behaviour change).
2. A + B behind `GA_NEEDS_V2=1`, with C values in a config table.
3. Short A/B runs (2 sim days) from `base_the_ville_n55`: v1 against v2 engines, same Laya model. Measures: needs distributions per hour, share of agents under 20, number of emergency replans per day, LLM calls per sim hour, food purchases, cafe and grocery sales, starvation count.
4. D as a Laya decision type, trained on cloud-teacher labels, if the keyword path stays too coarse.
5. E and F after the curves look right.

## 7. Risks

- A rebalanced engine will trigger many more emergency replans (extra LLM load) and may starve broke agents; the staff-meal fallback and a wallet floor check are needed.
- n56 was started on the current engine. Changes need a new run, or an explicit decision to compare n56 as the v1 control.
- Rebalancing shifts every downstream result (economy, job market, culture), so compare against the n56 control rather than against n55.

## 8. Implementation status (2026-11-03)

Built, all behind flags that default OFF (the running n56 sim is unchanged, except hourly needs logging which defaults ON but only writes a file once a restart picks it up):

- `reverie/backend_server/needs_env.py` - environment-decided relief. Intent (Laya or word-boundary keywords) + outcome (real consumption ledger, bathroom/seat/work object under the agent, real conversation ticks). Per-need cooldowns, rebalanced decay (`DECAY_V2`), social gained per tick while chatting, energy regained asleep.
- `reverie.py` hooks: `needs_env.note_consumed` on the three successful consume/purchase paths; v2 skips `satisfy_needs_for_action`, and calls `needs_env.apply_relief` after consumption (arrival-gated, also on the deferred path); `tick_needs` uses `tick_value`; hourly `needs_timeseries.jsonl` snapshot.
- Flags: `GA_NEEDS_V2=1` (engine), `GA_NEEDS_LAYA=1` (Laya intent instead of keywords), `GA_NEEDS_LOG=0` (disable logging), `GA_NEEDS_SOCIAL_GAIN`.
- Trace outputs under `GA_TRACE_DIR`: `needs_timeseries.jsonl` (all nine meters per agent per sim hour), `needs_relief.jsonl` (intent, source, object, ledger and relief granted per decision).
- Tests: `tests/test_needs_env.py` (8 pass): no food means no hunger relief, sub-tasks don't re-trigger, toilet needs a bathroom, shower needs hot water or a fixture, cooldowns, decay reaches critical in 12-20 awake hours.
- Laya "need" decision: `pipelines/laya_plan/needs_extract.py` (3,698 stratified distinct actions from n55), `needs_label.py` (cloud-teacher labels: glm-5.3 + minimax-m3.1-flash, nemotron tie-break), `needs_build.py` (v7e replay data + need items, persona-disjoint test), trainer `OVERSAMPLE["need"]=3`. Training needs the GPU.

Not yet done: train/evaluate the Laya need model; run the v1 vs v2 A/B on `base_the_ville_n55`; persona variance (E); extend the urgent map and prompts to comfort/stimulation/social (F).

## 9. Need-zero crises (GA_NEEDS_CRISIS, on under v2)

A need that reaches 0 is an event, not a clamp. Each fires at most once per agent per step, writes a poignancy 7-9 unpleasant memory to the agent, appends a `crisis` chronicle milestone, restores the need so it cannot repeat at once, and is traced to `needs_crisis.jsonl`.

| need at 0 | event | consequence | restored to |
|---|---|---|---|
| energy (awake) | collapses from exhaustion | transported to the Willow Clinic bed for 150 min, $15 fee to the community fund; hunger >= 40, hydration >= 50 | 70 |
| hunger | faints from hunger | clinic 90 min, $10 fee; hydration >= 50 | 60 |
| hydration | collapses from dehydration | clinic 90 min, $8 fee | 70 |
| bladder | embarrassing accident | hygiene -40, social -15 | 65 |
| hygiene | shunned for poor hygiene | social -25 | 50 |
| social | breakdown from loneliness | financial_stress +0.25 | 35 |
| comfort | laid low by aches | energy -10 | 50 |
| stimulation | numb, empty mood | financial_stress +0.1 | 45 |

Fees are capped at the agent's wallet (a broke agent is waived, not indebted). The transport is a one-step `planned_path` to the clinic bed tile, followed by an ordinary timed action, after which the normal planner resumes. Agents mid-conversation, or asleep (for energy), are skipped until clear. status is excluded by design.

Tests: `tests/test_needs_env.py` (13 pass) covers transport, fee capped at wallet, no double billing during a stay, no hospital for personal crises, and that every need except status has a crisis.
