# Reproducing every number

Each figure in the Stage 1 report comes from one script in `scripts/`, run
against the raw-data folder described in the README. Nothing below needs a GPU
or a model: the scripts read generations that were already produced and score
them with `cap_grpo/checker.py`.

```bash
export JSSP_RAW=/path/to/JSSP-SLM-raw     # or pass --data to each script
```

| Report item | Command | Key figures it prints |
|---|---|---|
| Checker audit | `python scripts/05_audit_checker.py` | 455 of 455 round trips clean, 2,275 mutations caught, 0 schedules below a proven optimum |
| RL effect, validity and quality | `python scripts/14_rl_effect.py` | 43.3 to 51.6 % for V8, W/L 262/73, p 3.9e-26, median gap change 0.00 pp |
| pass@k by region and quality | `python scripts/06_passk.py` | known shapes 77.6 to 100 % for SFT, within 10 % 72.5 against 95.0 % at k=64 |
| Solved and lost sets | `python scripts/07_solved_lost.py` | grid: 48 to 73 gained against 9 to 25 lost; pass@k: RL-only 0 to 4, all on new shapes |
| Best of K at equal cost | `python scripts/08_best_of_k.py` | known k=64 gap 5.0 against 3.5 %; supervised needs k=3 to match one RL sample |
| Re-timing diagnostic | `python scripts/09_retiming.py` | repair recovers 0.4 to 3.4 points; 45 to 62 % of answers have the wrong operation set |
| Duration scaling | `python scripts/10_duration_scaling.py` | x1 41.2 % falls to 11.9 % at x10; long durations 0 to 45.0 % when divided by ten |
| Gap distribution, RL-only instances | `python scripts/11_gap_distribution.py --model V4` | 8.1 against 13.2 % mean gap; size-matched at 100 operations 14.2 against 15.6 % |
| Dispatching-rule reference | `python scripts/12_dispatching_rules.py` | MWKR 11.9 % over all 455 and never fails; models 9.6 to 12.4 % on 42 to 68 % of instances |
| Hosted-model failures | `python scripts/13_frontier_failures.py` | GPT-4o 0 of 18, all constraint violations; o3 one-shot 12 of 18, the rest truncation |

Three names exist for the same run: the number used in the tables, the key
written into the generation files, and the reward mode written into the
training log. They are kept as they were produced, so this table is the way
between them.

| Variant | Key in the generations | Training-log folder | `rewards/…` key in `trainer_state.json` |
|---|---|---|---|
| — | `Base-LLaMA` | — | no fine-tuning |
| — | `SFT-LoRA` | `sft_llama_lora` | supervised, no reward |
| V1 feasibility only | `Vbin` | `V1_feasibility_only` | `jssp_reward_uniform` |
| V2 feasibility plus makespan | `VbinMk` | `V2_feasibility_and_makespan` | `jssp_reward_binary_makespan` |
| V3 feasibility plus penalty | `StratNoMk` | `V3_feasibility_and_penalty` | `jssp_reward_stratified_nomakespan` |
| V4 stratified | `V1` | `V4_stratified` | `jssp_reward_stratified` |
| V5 stratified, matched settings | `V3` | `V5_stratified_matched_settings` | `jssp_reward_stratified` |
| V6 stratified plus length mask | `V6LoRA` | `V6_stratified_length_mask` | `jssp_reward_stratified` |
| V7 hybrid, CAP-GRPO | `V4` | `V7_hybrid_cap_grpo` | `jssp_reward_hybrid` |
| V8 hybrid plus over-emission | `V7` | `V8_hybrid_over_emission_penalty` | `jssp_reward_hybrid_v7` |
| 2-1 random reward | `RANDL42`, `RANDL43` | `stage2/test_2_1_random_reward/seed42`, `/seed43` | `jssp_reward_random` |
| 2-2 more supervised training | `SFTbudget5e6`, `SFTbudget2e4` | `stage2/test_2_2_more_supervised/lr5e-6`, `/lr2e-4` | supervised, no reward |
| 2-3 rejection sampling | `RFT` | `stage2/test_2_3_rejection_sampling/run` | supervised, no reward |

Three traps in that table, all of them from an earlier numbering that the
tables no longer use:

- The key `V1` in the generation files is variant **V4**, `V3` is **V5**, `V4`
  is **V7** and `V7` is **V8**. The keys were frozen before the variants were
  renumbered and were left alone so that stored files keep their meaning.
- V8's reward mode is called `hybrid_v7` in the code and in its log. The `v7`
  there is the old label of that reward, not variant V7.
- V4, V5 and V6 share one reward key, `jssp_reward_stratified`, and that is
  correct: V4 and V5 use the same reward with different training settings, and
  V6 adds a mask on the gradient rather than a change to the reward.

## What the raw data does and does not contain

The grid stores the full text of all 25,025 answers, so every verdict there can
be recomputed from scratch. The pass@k run as reported stores the verdict,
violation counts and token length of each of the 57,600 samples but not their
text; `scripts/06_passk.py` and `08_best_of_k.py` read those verdicts rather
than re-scoring text. A repeat of the same experiment with the text kept sits
beside it in `04_stage1_evaluation/passk/repeat_with_text/`; the README says
how far the two agree.

## Training logs

The training logs are in this repository, not in the raw-data folder, because
they are small. They are named by the number used in the tables:
`V1_feasibility_only` through `V8_hybrid_over_emission_penalty`, the two
supervised runs, and `stage2/` for the control tests.

`training_logs/<run>/trainer_state.json` holds reward, reward_std, KL, loss,
grad_norm and completion_length at every logging step, and
`training_logs/<run>/training.log` the full console output. One exception:
`V3_feasibility_and_penalty` has `trainer_state_from_log.json` instead, rebuilt
from its console log because that run's checkpoints were pruned when the
results were imported. It carries the same 100 logged steps. For the supervised
runs the same file records the evaluation loss at 49 checkpoints and the
selected checkpoint: step 9,800, evaluation loss 0.40401, the lowest of the
run. No benchmark instance took part in that choice.

## Stage 2, the control tests

`scripts/15_stage2_controls.py` recomputes every number in the Stage 2 section
of the README from the raw folder:

```bash
python scripts/15_stage2_controls.py --data "$JSSP_RAW"
```

It reads `05_stage2_tests/test_2_1_random_reward/generations.jsonl` and the two
beside it, together with the main grid, and compares each control with the
supervised checkpoint it started from, using the same exact sign test as step
15.

The three tests, and what each one removes:

| | Test | What is taken away | Result |
|---|---|---|---|
| **2-1** | random reward | the information in the reward | 29.7 % and 30.4 % valid, against 43.3 % at the start |
| **2-2** | more supervised training | the reward and the sampling altogether | 58.2 % at lr 5e-6 and 69.5 % at lr 2e-4 |
| **2-3** | rejection sampling | the policy gradient, leaving the checker as a filter | 43.1 %, where it started |

Each had the same supervised checkpoint, the same 2,000 records and the same
500 optimizer updates as one RL run, and 2-3 drew the same 8,000 completions.

`scripts/as_run/` holds the files that produced these runs on the GPU node,
copied without change. They are written against the private training
repository's module layout and will not run here; they are included because the
exact code behind a number is part of the record.
