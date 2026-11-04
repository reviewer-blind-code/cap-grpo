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
| Checker audit | `python scripts/06_audit_checker.py` | 455 of 455 round trips clean, 2,275 mutations caught, 0 schedules below a proven optimum |
| RL effect, validity and quality | `python scripts/15_rl_effect.py` | 43.3 to 51.6 % for V8, W/L 262/73, p 3.9e-26, median gap change 0.00 pp |
| pass@k by region and quality | `python scripts/07_passk.py` | known shapes 77.6 to 100 % for SFT, within 10 % 72.5 against 95.0 % at k=64 |
| Solved and lost sets | `python scripts/08_solved_lost.py` | grid: 48 to 73 gained against 9 to 25 lost; pass@k: RL-only 0 to 4, all on new shapes |
| Best of K at equal cost | `python scripts/09_best_of_k.py` | known k=64 gap 5.0 against 3.5 %; supervised needs k=3 to match one RL sample |
| Re-timing diagnostic | `python scripts/10_retiming.py` | repair recovers 0.4 to 3.4 points; 45 to 62 % of answers have the wrong operation set |
| Duration scaling | `python scripts/11_duration_scaling.py` | x1 41.2 % falls to 11.9 % at x10; long durations 0 to 45.0 % when divided by ten |
| Gap distribution, RL-only instances | `python scripts/12_gap_distribution.py --model V4` | 8.1 against 13.2 % mean gap; size-matched at 100 operations 14.2 against 15.6 % |
| Dispatching-rule reference | `python scripts/13_dispatching_rules.py` | MWKR 11.9 % over all 455 and never fails; models 9.6 to 12.4 % on 42 to 68 % of instances |
| Hosted-model failures | `python scripts/14_frontier_failures.py` | GPT-4o 0 of 18, all constraint violations; o3 one-shot 12 of 18, the rest truncation |

Model keys in the raw data map to the variant numbering as follows.

| Key | Variant | Reward |
|---|---|---|
| `Base-LLaMA` | — | no fine-tuning |
| `SFT-LoRA` | — | supervised, LoRA; the start of every RL run |
| `SFT-rsLoRA` | — | supervised, rsLoRA; a second baseline, never an RL start |
| `Vbin` | V1 | feasibility only |
| `VbinMk` | V2 | feasibility plus makespan |
| `StratNoMk` | V3 | feasibility plus constraint penalty |
| `V1` | V4 | stratified |
| `V3` | V5 | stratified, matched hyperparameters |
| `V6LoRA` | V6 | stratified plus length mask |
| `V4` | V7 | hybrid, CAP-GRPO |
| `V7` | V8 | hybrid plus over-emission penalty |

## What the raw data does and does not contain

The grid stores the full text of all 25,025 answers, so every verdict there can
be recomputed from scratch. The pass@k run stores the verdict, violation counts
and token length of each of the 57,600 samples but not their text, which would
have been about half a gigabyte; `scripts/07_passk.py` and `09_best_of_k.py`
therefore read those verdicts rather than re-scoring text.

## Training logs

`training_logs/<run>/trainer_state.json` holds reward, reward_std, KL, loss,
grad_norm and completion_length at every logging step, and
`training_logs/<run>/training.log` the full console output. For the supervised
runs the same file records the evaluation loss at 49 checkpoints and the
selected checkpoint: step 9,800, evaluation loss 0.40401, the lowest of the
run. No benchmark instance took part in that choice.
