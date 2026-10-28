# LLM call census (20611 calls, 1120 LLM-minutes, 7 trace files)

| caller | calls | time % | mean s | out tok | short | numeric | self-consist | distinct |
|---|---|---|---|---|---|---|---|---|
| run_gpt_prompt_action_location | 6025 | 24.4 | 2.7 | 14 | 0.60 | 0.00 | 0.48 | 290 |
| run_gpt_prompt_choose_activity | 2642 | 23.8 | 6.0 | 566 | 0.00 | 0.00 | 0.00 | 2194 |
| generate_one_utterance | 3914 | 12.3 | 2.1 | 53 | 0.00 | 0.00 | - | 3000 |
| run_gpt_prompt_new_decomp_schedule | 929 | 9.3 | 6.7 | 642 | 0.01 | 0.00 | 0.00 | 878 |
| run_gpt_prompt_task_decomp | 644 | 7.1 | 7.4 | 554 | 0.00 | 0.00 | 0.01 | 445 |
| run_gpt_prompt_insight_and_guidance | 1481 | 6.3 | 2.9 | 303 | 0.00 | 0.00 | 0.00 | 1280 |
| ? | 635 | 5.2 | 5.6 | 305 | 0.05 | 0.00 | 0.19 | 592 |
| run_gpt_prompt_agent_chat_summarize_relationsh | 738 | 2.1 | 1.9 | 101 | 0.00 | 0.00 | 0.00 | 695 |
| run_gpt_prompt_planning_thought_on_convo | 760 | 1.7 | 1.5 | 49 | 0.00 | 0.00 | 0.00 | 758 |
| run_gpt_prompt_memo_on_convo | 760 | 1.6 | 1.4 | 43 | 0.00 | 0.00 | 0.00 | 758 |
| run_gpt_prompt_focal_pt | 490 | 1.2 | 1.7 | 93 | 0.00 | 0.00 | 0.00 | 484 |
| run_gpt_prompt_summarize_conversation | 385 | 1.0 | 1.7 | 56 | 0.00 | 0.00 | - | 385 |
| run_gpt_prompt_daily_plan | 61 | 0.8 | 9.3 | 187 | 0.00 | 0.00 | 0.00 | 61 |
| run_gpt_prompt_wake_up_hour | 60 | 0.5 | 5.2 | 9 | 0.97 | 0.00 | 0.36 | 34 |
| run_gpt_prompt_create_artifact | 232 | 0.4 | 1.3 | 76 | 0.00 | 0.00 | 0.00 | 232 |
| run_gpt_prompt_artifact_reaction | 250 | 0.4 | 1.2 | 55 | 0.00 | 0.00 | 0.00 | 250 |
| generate_meal_description | 113 | 0.4 | 2.3 | 218 | 0.00 | 0.00 | 0.00 | 113 |
| generate_written_content | 49 | 0.3 | 4.0 | 401 | 0.00 | 0.00 | 0.00 | 49 |
| run_gpt_prompt_action_game_object | 139 | 0.3 | 1.4 | 4 | 1.00 | 0.00 | 0.80 | 24 |
| run_gpt_prompt_action_arena | 69 | 0.2 | 2.2 | 6 | 1.00 | 0.00 | 1.00 | 19 |
| run_gpt_prompt_chat_poignancy | 77 | 0.2 | 1.6 | 6 | 1.00 | 0.00 | - | 8 |
| generate_painting_with_image | 67 | 0.1 | 1.5 | 132 | 0.00 | 0.00 | 0.00 | 67 |
| run_gpt_prompt_commitments | 60 | 0.1 | 1.3 | 12 | 0.80 | 0.00 | 0.00 | 13 |
| run_gpt_prompt_typewriter | 15 | 0.1 | 3.6 | 281 | 0.00 | 0.00 | 0.00 | 14 |
| run_gpt_prompt_event_poignancy | 11 | 0.0 | 1.6 | 6 | 1.00 | 0.00 | - | 3 |
| generate_invention_description | 2 | 0.0 | 1.9 | 232 | 0.00 | 0.00 | - | 2 |
| run_gpt_prompt_event_triple | 2 | 0.0 | 0.9 | 17 | 0.50 | 0.00 | - | 2 |
| run_gpt_prompt_action_sector | 1 | 0.0 | 0.5 | 7 | 1.00 | 0.00 | - | 1 |
