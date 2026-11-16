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

Every file now calls a run by the number used in the tables. The keys the
files were produced under are kept beside them, so nothing was lost in the
renaming: each raw record carries `model_as_run`, and each training log carries
`reward_mode_as_run` and `variant` at the top level.

| Variant | Produced as, in the generations | Training-log folder | Produced as, in the log |
|---|---|---|---|
| `Base` | `Base-LLaMA`, `Base` | — | no fine-tuning |
| `SFT-LoRA` | `SFT-LoRA` | `sft_llama_lora` | supervised, no reward |
| `V1` feasibility only | `Vbin`, `Vbin-feasonly` | `V1_feasibility_only` | `uniform` |
| `V2` feasibility plus makespan | `VbinMk`, `VbinMk-feasmk` | `V2_feasibility_and_makespan` | `binary_makespan` |
| `V3` feasibility plus penalty | `StratNoMk`, `StratNoMk-feaspen` | `V3_feasibility_and_penalty` | `stratified_nomakespan` |
| `V4` stratified | `V1`, `V1-stratified` | `V4_stratified` | `stratified` |
| `V5` stratified, matched settings | `V3`, `V3-stratified` | `V5_stratified_matched_settings` | `stratified` |
| `V6` stratified plus length mask | `V6LoRA`, `V6LoRA-lenmask` | `V6_stratified_length_mask` | `stratified` |
| `V7` hybrid, CAP-GRPO | `V4`, `V4-hybrid` | `V7_hybrid_cap_grpo` | `hybrid` |
| `V8` hybrid plus over-emission | `V7`, `V7-hybridover` | `V8_hybrid_over_emission_penalty` | `hybrid_v7` |
| `T2-1-seed42`, `T2-1-seed43` | `RANDL42`, `RANDL43` | `stage2/test_2_1_random_reward/seed42`, `/seed43` | `random` |
| `T2-2-lr5e-6`, `T2-2-lr2e-4` | `SFTbudget5e6`, `SFTbudget2e4` | `stage2/test_2_2_more_supervised/lr5e-6`, `/lr2e-4` | supervised, no reward |
| `T2-3` | `RFT` | `stage2/test_2_3_rejection_sampling/run` | supervised, no reward |

Why the second column looks the way it does: the keys were frozen before the
variants were renumbered, so `V1` in a file produced then is variant V4, `V4`
is V7 and `V7` is V8. The reward of V8 is called `hybrid_v7` in the code, where
that `v7` is the reward's old label and not variant V7. V4, V5 and V6 share one
reward name, which is correct: V4 and V5 run the same reward with different
training settings, and V6 adds a mask on the gradient rather than a change to
the reward.

The two scripts that did the renaming are in the private training repository as
`analysis/canonical_names.py` and `analysis/canonical_reward_keys.py`; copies
are in `scripts/as_run/`. Both are idempotent and touch only the name fields.

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
`training_logs/<run>/training.log` the full console output. One note on
`V3_feasibility_and_penalty`: its checkpoints were deleted when the results were
imported, so its `trainer_state.json` was rebuilt from the console log beside
it. The file carries the same 100 logged steps as every other run and says so
in its own `reconstructed_from` field. For the supervised
runs the same file records the evaluation loss at 49 checkpoints and the
selected checkpoint: step 9,800, evaluation loss 0.40401, the lowest of the
run. No benchmark instance took part in that choice.

## The base model with one worked example

`04_stage1_evaluation/probes/base_one_shot/` holds 175 answers: the model with
no fine-tuning, on the 35 classic instances, 5 seeds, with one solved 2×2
instance placed before the question.

```bash
python evaluation/run_grid.py --models Base --seeds 0 1 2 3 4 \
    --sets classic --prompt-style alpaca_1shot --out grid_base_1shot
```

Zero-shot, none of those 175 answers contains a parseable operation. With the
example, 38.9 % of them do, and the constraint counters rise from zero to 47.1
machine conflicts and 28.1 precedence violations per answer. Still nothing
valid, and the closest answer breaks 46 constraints.

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
