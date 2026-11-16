# CAP-GRPO: Constraint-Aware Policy with GRPO for Job-Shop Scheduling

Fine-tune an LLM to solve the **Job-Shop Scheduling Problem (JSSP)** using
**GRPO** (Shao et al., 2024), guided by a constraint-aware reward function
(CAP) with a coverage gate that prevents reward hacking from vacuous outputs.

Everything here was re-measured on one evaluation grid and then controlled
against three alternatives. Two things came out of that, and they point in
different directions. The reward improves the **quality** of the schedules the
model gets right, by about one point of gap, and nothing else tried does that.
The rise in how **often** the model writes a valid schedule is not specific to
the reward: plain supervised training at the same budget raises it two to three
times further. Both are reported below, in [Stage 1](#stage-1-the-audit) and
[Stage 2](#stage-2-where-the-gain-comes-from).

## The supervised stage

Four base models were fine-tuned on the StarJob corpus, 108,000 schedules, one
pass, LoRA throughout. LLaMA 3.1 8B came out ahead and is the start of
everything after it.

| Base model | Valid schedules |
|---|---|
| **LLaMA 3.1 8B** | **95.5 %** |
| Ministral 8B | 93.0 % |
| Granite 3.2 8B | 87.0 % |
| Qwen2.5 7B | 1.0 % |

Those scores are measured on the corpus's own held-out split, and that split is
not clean: 198 of its 200 items also appear in the training data, because the
split was drawn with a different random generator than the one that built the
training set. The ranking of the four models still holds, but every number in
that column is too high. On the benchmark grid used everywhere else in this
README, the same LLaMA model reaches **43.3 %**.

The checkpoint carried forward is step 9,800, chosen as the lowest evaluation
loss of the run, 0.40401, measured on the corpus's own split. No benchmark
instance took part in that choice. `training_logs/sft_llama_lora/`
holds the evaluation loss at all 49 checkpoints.

## The RL stage

Eight reward designs were trained from that one checkpoint, each for 500 steps
with K = 4 samples per prompt over the same 2,000 records. The designs differ
in what the reward counts:

| | Design | Feasible flag | Per-constraint penalty | Makespan term |
|---|---|---|---|---|
| V1 | Feasibility only | yes | no | no |
| V2 | Feasibility plus makespan | yes | no | yes |
| V3 | Feasibility plus penalty | yes | yes | no |
| V4 | Stratified | yes | weighted | yes |
| V5 | Stratified, matched settings | yes | weighted | yes |
| V6 | Stratified plus length mask | yes | weighted | yes |
| V7 | Hybrid, CAP-GRPO | yes | equal weights | yes |
| V8 | Hybrid plus over-emission penalty | yes | equal weights | yes |

V4 and V5 use the same reward and differ only in the training settings. V6 adds
a mask that drops over-long samples from the gradient, V8 a penalty for writing
the same job too many times.

Training logs for all eight are in `training_logs/`, named by these numbers,
and every file inside calls the run by the same number. The names the files
were produced under are kept beside the new ones, `model_as_run` in the
generations and `reward_mode_as_run` in the logs, and `docs/REPRODUCE.md` maps
the two. That mapping is worth a look before reading a raw file by hand: the
old keys predate this numbering, so a file produced then calls variant V4
"V1".

## Stage 1: the audit

Every number below comes from one uniform evaluation grid: 11 models, 5
decoding seeds, 455 instances, 25,025 generations, temperature 0.1, vLLM,
scored with `cap_grpo/checker.py`. Earlier versions of this table reported
single-run numbers on 18 instances; those are superseded.

### Valid schedules over the whole grid

| Model | Valid | Gain over its supervised start | Wins / losses | p |
|---|---|---|---|---|
| Hybrid plus over-emission penalty (V8) | **51.6 %** | +8.3 pp | 262 / 73 | 3.9e-26 |
| Feasibility plus makespan (V2) | 51.4 % | +8.1 pp | 258 / 73 | 2.3e-25 |
| Stratified, matched hyperparameters (V5) | 51.3 % | +8.0 pp | 258 / 75 | 1.2e-24 |
| Hybrid, CAP-GRPO (V7) | 50.0 % | +6.7 pp | 230 / 78 | 1.5e-18 |
| Feasibility plus penalty (V3) | 49.2 % | +5.9 pp | 211 / 76 | 7.4e-16 |
| Feasibility only (V1) | 48.9 % | +5.6 pp | 205 / 77 | 1.4e-14 |
| Stratified (V4) | 47.4 % | +4.1 pp | 192 / 99 | 5.4e-08 |
| Stratified plus length mask (V6) | 45.5 % | +2.2 pp | 158 / 109 | 3.2e-03 |
| Supervised fine-tuning, LoRA | 43.3 % | — | — | — |
| No fine-tuning | 0.0 % | — | — | — |

Every RL variant starts from the same LoRA supervised checkpoint. The grid also
holds an rsLoRA supervised baseline, 38.1 % valid, which `14_rl_effect.py` will
print; it is a leftover from an earlier comparison and nothing in this work uses
it. The test is a two-sided exact sign test on the pairs where exactly one model
produced a valid schedule, equivalently the exact McNemar test.

### By benchmark family, valid % and mean gap

| Family | n | Supervised (LoRA) | CAP-GRPO (V7) | Over-emission (V8) |
|---|---|---|---|---|
| Classic literature | 35 | 25.7 % / 11.2 % | 39.4 % / 11.0 % | 38.3 % / 11.9 % |
| Generated | 200 | 36.1 % / 8.5 % | 39.6 % / 6.7 % | 39.8 % / 7.3 % |
| Learning to dispatch | 200 | 57.9 % / 9.6 % | 67.2 % / 9.5 % | 70.9 % / 9.9 % |
| Long durations | 10 | 0 % | 0 % | 0 % |
| Taillard | 10 | 0 % | 0 % | 0 % |

### What the gain is, and is not

- **Validity, not quality.** On the pairs where both models produce a valid
  schedule, the median change in gap is 0.00 pp for every variant, and the mean
  never moves more than 1.24 points either way.
- **Sampling efficiency, mostly.** With 64 samples the supervised model reaches
  100 % on known shapes too. On those shapes one RL sample buys what two to
  three supervised samples buy: the reward runs solve 83.7 to 84.0 % at one
  sample, and the supervised model is at 82.1 % with two and 87.1 % with four.
- **No new capability.** Long durations and Taillard stay at zero for every
  variant, every budget and every prompt.
- **A reference point.** The MWKR dispatching rule reaches 11.9 % mean gap over
  all 455 instances and never fails; the models reach 9.6 to 12.4 % on the 42
  to 68 per cent of instances they answer at all.

`docs/REPRODUCE.md` gives the command behind each of these numbers.

## Stage 2: where the gain comes from

The table above says the RL stage raises validity. It does not say why. Three
controls each take one possible cause away:

| | Test | What it takes away |
|---|---|---|
| **2-1** | random reward | the information in the reward: a draw from N(2.66, 1.93) that never reads the schedule, matched to the mean and spread of the real one |
| **2-2** | more supervised training | the reward and the sampling altogether, leaving plain next-token training on the gold schedules |
| **2-3** | rejection sampling | the policy gradient, leaving the checker as a filter on the model's own samples |

Each is given exactly what one RL run is given: the same supervised checkpoint,
the same 2,000 records, 500 optimizer updates, and where sampling is involved,
the same 8,000 completions.

| Model | Valid | Change | Wins / losses | p | Gap change on pairs both solve |
|---|---|---|---|---|---|
| **2-2** more supervised training, lr 2e-4 | **69.5 %** | +26.2 pp | 628 / 32 | 1.3e-144 | +0.06 pp |
| **2-2** more supervised training, lr 5e-6 | **58.2 %** | +14.9 pp | 397 / 57 | 8.9e-64 | −0.52 pp |
| Stratified, matched settings (V5) | 51.3 % | +8.0 pp | 258 / 75 | 1.2e-24 | **−1.12 pp** |
| Hybrid, CAP-GRPO (V7) | 50.0 % | +6.7 pp | 230 / 78 | 1.5e-18 | **−1.24 pp** |
| Supervised start | 43.3 % | — | — | — | — |
| **2-3** rejection sampling | 43.1 % | −0.2 pp | 95 / 100 | 0.78 | +0.16 pp |
| **2-1** random reward, seed 42 | 29.7 % | −13.6 pp | 19 / 328 | 6.8e-74 | +1.08 pp |
| **2-1** random reward, seed 43 | 30.4 % | −12.9 pp | 61 / 355 | 1.6e-51 | +3.53 pp |

Three things follow, one from each test.

- **2-1: the training loop alone does not do it.** With the reward replaced by
  a random draw of the same mean and spread, validity falls by about thirteen
  points on both seeds. The reward has to read the schedule.
- **2-2: the rise in validity is not what the reward designs achieve.** Plain
  supervised training at the same budget raises it two to three times further.
  It gets there by finishing the answer: the share cut off at the token limit
  falls from 10.6 % to 2.9 %, and the share that finishes but breaks a rule
  from 46.1 % to 27.6 %. The lr 2e-4 run matches the number of updates, not the
  size of each step; the matched run is lr 5e-6, and it still gains +14.9 pp.
- **2-3: using the checker as a filter is not enough.** It lands where it
  started, because the filter had almost nothing to remove: 7,312 of its 8,000
  samples were already valid.

What the reward does achieve is **quality**. On the problems a pair of models
both solve, only the reward runs move the gap, by about one point. None of the
three controls does.

A fourth test, a stronger supervised stage and a second base model, has not
been run.

`python scripts/15_stage2_controls.py` recomputes every number in this table,
and the raw folder keeps one directory per test under `05_stage2_tests/`.

## Raw data

The generations are too large for git. They live in a Google Drive folder,
507 MB, 171 files:

**https://drive.google.com/drive/folders/11bDSEA9ir1Pib3tmwzl87SV6ovgVbSrl**

```
JSSP-SLM-raw/
  00_README.md, 00_SHA256SUMS
  01_instances/            every problem set, one JSON object per line
  02_sft_stage/            how each base model was trained, and its console log
                           (`base_models/run_rslora.log` is the LLaMA rsLoRA run;
                           the LoRA run's own console log was not kept, only its
                           trainer_state.json, which is in this repository)
  03_rl_stage/             the eight reward designs, training logs
  04_stage1_evaluation/    one protocol for every model
      main_grid/           25,025 answers, full text, 11 models x 5 seeds x 455
      passk/               57,600 samples, twice: verdicts as reported, and a
                           repeat that kept the text
      sm_heldout/          9,000 answers on 225 held-out StarJob problems
      probes/              duration scaling, token budget, prompt sensitivity
      hosted_models/       stored answers of the hosted models
  05_stage2_tests/         one folder per control test, logs next to answers
  06_earlier_runs/         earlier evaluation output, kept for the record
```

The training logs are small, so they are in this repository under
`training_logs/`, named by the number used in the tables: `V1_feasibility_only`
through `V8_hybrid_over_emission_penalty`, plus the supervised runs and the
Stage 2 controls. Each holds `trainer_state.json` with reward, reward_std, KL,
loss, gradient norm, completion length and learning rate at every logging step.

Point the scripts at the raw folder once:

```bash
export JSSP_RAW=/path/to/JSSP-SLM-raw
python scripts/14_rl_effect.py
python scripts/15_stage2_controls.py
```

`docs/REPRODUCE.md` maps every reported number to the command that prints it.

### A note on pass@k

`04_stage1_evaluation/passk/` holds two files for the same experiment.
`as_reported_verdicts_only/` is the run the reported numbers come from; only
the verdicts were stored. `repeat_with_text/` is the same experiment run again
with the same problems, the same k, the same temperature and the same seed,
keeping the text, so the checker can be run again on the schedules themselves.

The two are not sample for sample identical: vLLM sampling with a LoRA adapter
is not reproducible bit for bit between runs. 95.8 % of the 57,600 verdicts
agree and the pass@k curves agree within 0.1 to 2.2 points, where one problem
out of 90 is worth 1.1 points. Both files are kept so that neither claim has to
be taken on trust.

## Feasibility checker

Every schedule is scored by `cap_grpo/checker.py`: six independent counters,
and a schedule is feasible when all six are zero. The checker is audited
against CP-SAT — it accepts every optimal schedule, reproduces its makespan,
never accepts a schedule below a proven optimum, and rejects five kinds of
injected defect.

```bash
python tests/test_checker.py            # 21 hand-built cases, one rule each
python scripts/05_audit_checker.py      # CP-SAT round trip and mutations
```

| Test | Coverage | Result |
|---|---|---|
| CP-SAT optimal schedule accepted | 455 instances | 0 failures |
| Makespan equals the CP-SAT objective | 455 instances | 0 mismatches |
| Feasible schedule below the proven optimum | 25,025 generations | 0 |
| Injected defects detected | 2,275 mutations | 0 missed |
| Hand-built cases | 21 cases | 21 / 21 |

`cap_grpo/checker_legacy.py` is the earlier implementation, kept for
comparison: it agrees on 99.964 % of 25,025 stored generations, and the three
rules that differ are described in [docs/CHECKER.md](docs/CHECKER.md).

## Pipeline Overview

Training, in four steps:

```
01_download_data.py       download StarJob from HuggingFace + OOD from OR-Library
02_run_sft.py             supervised fine-tuning (SFT) with Alpaca format
03_run_grpo.py            CAP-GRPO reinforcement learning from SFT adapter
04_eval_ood.py            a quick check on the 18 classic FT and LA instances
```

Analysis, reading the raw-data folder rather than running a model:

```
05_audit_checker.py       the checker against CP-SAT, and against mutations
06_passk.py               pass@k from the stored samples
07_solved_lost.py         which instances each model gains and loses
08_best_of_k.py           best of k at equal inference cost
09_retiming.py            how many invalid schedules a repair pass can save
10_duration_scaling.py    what happens when durations are divided by ten
11_gap_distribution.py    the gap distribution, not just its mean
12_dispatching_rules.py   FIFO, SPT, LPT, MWKR and LWKR as a reference point
13_frontier_failures.py   why the hosted models fail, counted by cause
14_rl_effect.py           the paired effect of the RL stage over its SFT start
15_stage2_controls.py     the three Stage 2 control tests
as_run/                   the files that produced the runs on the GPU node
```

---

## Setup

```bash
# 1. Create a dedicated virtual environment
python -m venv venv-grpo && source venv-grpo/bin/activate

# 2. Install PyTorch (CUDA 12.1 example — adjust to your CUDA version)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

**Hardware:** ≥ 16 GB VRAM required. RTX 4090 (24 GB) tested.
GRPO samples K=4 completions per prompt per step — more memory-intensive than SFT.

### Required environment variables

```bash
export TOKENIZERS_PARALLELISM=false
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2   # reduces CPU memory fragmentation
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
```

Install jemalloc if missing: `sudo apt install libjemalloc2`

---

## Step 1 — Download Data

```bash
python scripts/01_download_data.py
```

Downloads:
- **StarJob**: ~130k JSSP instances from HuggingFace (`mideavalwisard/Starjob`)
- **SM filter**: keeps instances with jobs ≤ 10 and machines ≤ 10 (~108k records)
- **OOD benchmarks**: OR-Library `jobshop1.txt` (FT + LA + other instances)

```bash
# Custom SM filter bounds
python scripts/01_download_data.py --max-jobs 8 --max-machines 8

# Skip StarJob if already downloaded
python scripts/01_download_data.py --skip-starjob
```

**Output:**
```
data/
├── starjob_raw.json           (full StarJob, ~130k records)
├── starjob_train_sm.jsonl     (SM subset, ~108k records)
└── benchmarks/
    └── jobshop1.txt
```

---

## Step 2 — Supervised Fine-Tuning (SFT)

SFT teaches the model the JSSP output format before RL begins, which
dramatically improves GRPO convergence speed and stability. It uses LoRA
(Hu et al., 2021). Rank-stabilised scaling (rsLoRA; Kalajdzievski, 2023) was
tried early on and is still available behind `--rslora`, but every result in
this repository comes from plain LoRA.

```bash
# Train LLaMA-3.1-8B (recommended)
python scripts/02_run_sft.py --model llama

# Other supported models
python scripts/02_run_sft.py --model qwen2
python scripts/02_run_sft.py --model granite
python scripts/02_run_sft.py --model ministral

# Smoke test (100 records only)
python scripts/02_run_sft.py --model llama --max-records 100

# From a locally downloaded base model
python scripts/02_run_sft.py --model-path /path/to/local/llama
```

**Output:** `outputs/sft_llama/final_adapter/`

### SFT hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| LoRA rank | 32 | `--lora-r` |
| LoRA alpha | 32 | matches rank |
| rsLoRA | off | `--rslora` turns on rank-stabilised scaling |
| Learning rate | 2e-4 | `--lr` |
| Epochs | 1 | `--epochs` |
| Batch size | 1 | `--batch-size` |
| Gradient accum | 8 | effective batch = 8 |

---

## Step 3 — CAP-GRPO Training

Applies GRPO with the constraint-aware reward (CAP) starting from the SFT adapter.

```bash
# Standard run (500 steps, hybrid reward)
python scripts/03_run_grpo.py --model-path outputs/sft_llama/final_adapter

# With length control (recommended for long runs)
python scripts/03_run_grpo.py \
    --model-path outputs/sft_llama/final_adapter \
    --length-control

# Smoke test
python scripts/03_run_grpo.py \
    --model-path outputs/sft_llama/final_adapter \
    --max-steps 10 --max-records 50

# Resume from checkpoint
python scripts/03_run_grpo.py \
    --model-path outputs/cap_grpo_v1/checkpoint-200 \
    --resume-from outputs/cap_grpo_v1/checkpoint-200 \
    --max-steps 500
```

**Output:** `outputs/cap_grpo_<label>_<mode>/final_adapter/`

### Reward modes

| Mode | Description | Range |
|------|-------------|-------|
| `hybrid` | 7-component CAP + coverage gate (default) | (-∞, 7]; unparseable floor = −1.0 |
| `hybrid_v7` | hybrid + over-emit penalty R_O | (-∞, 8]; same floor |
| `stratified` | per-category weighted penalty | [−1, 1] |
| `uniform` | binary: {0, 1, 7} | {0, 1, 7} |

### CAP reward decomposition

```
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality    (hybrid)
R = R_format + R_M + R_R + R_C + R_T + R_P + R_quality + R_O  (hybrid_v7)

R_format = +1 if parseable, else -1.0 (hard floor)
R_M      = +1 if no missing ops; else -(missing / N_ops)
R_R      = +cov if no routing violations; else -(violations / N_ops)
R_C      = +cov if no machine-capacity violations; else -(violations / N_ops)
R_T      = +cov if no timing inconsistencies (s+d≠e); else -(violations / N_ops)
R_P      = +cov if no precedence violations; else -(violations / N_ops)
R_quality= BKS/Cmax if fully feasible, else 0
R_O      = +1 if no over-emitted ops; else -(over_count / N_ops)  [hybrid_v7 only]

cov = min(ops_emitted / ops_expected, 1.0)   ← coverage gate
```

### Length control

Completions longer than `2.0 × (12.5 × N_ops + 50)` tokens get their GRPO
advantage zeroed — they contribute no gradient. This prevents the
**length-escape collapse** where the model learns to pad output to avoid
constraint checks (observed in early V2/V3 runs).

### GRPO hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| K (completions/prompt) | 4 | `K_SAMPLES` in config.py |
| Max steps | 500 | `--max-steps` |
| Learning rate | 5e-6 | `--lr` |
| KL coefficient | 0.05 | `--kl-coef` |
| Gradient accum | 4 | `--grad-accum` |
| Temperature | 0.7 | `--temperature` |
| Max new tokens | 4096 | `MAX_NEW_TOKENS` in config.py |

---

## Step 4 — Evaluation

`04_eval_ood.py` is a quick check on 18 classic instances, useful while a run
is still training. It is **not** the evaluation the results in this README come
from: those use the uniform grid of 455 instances and 5 seeds described in
[Stage 1](#stage-1-the-audit), produced by `scripts/as_run/run_grid.py` and
analysed by `scripts/14_rl_effect.py` and `scripts/15_stage2_controls.py`.

```bash
# OOD benchmarks (all 18 FT+LA instances)
python scripts/04_eval_ood.py --model-path outputs/cap_grpo_v1/final_adapter

# Save results to JSON
python scripts/04_eval_ood.py \
    --model-path outputs/cap_grpo_v1/final_adapter \
    --out results/cap_grpo_v1_ood.json

# Only Fisher-Thompson instances (ft06, ft10, ft20)
python scripts/04_eval_ood.py --model-path ... --dataset ft

# Only Lawrence instances (la01-la10, la16-la20)
python scripts/04_eval_ood.py --model-path ... --dataset la

# In-distribution sanity check (SM held-out 2% test split)
python scripts/04_eval_ood.py --model-path ... --dataset sm --n-sm-samples 20

# Single-instance debug
python scripts/04_eval_ood.py --model-path ... --infer ft06
```

**OOD instances and best-known makespans:**

| Instance | Size | BKS | | Instance | Size | BKS |
|----------|------|-----|-|----------|------|-----|
| ft06 | 6×6 | 55 | | la06 | 15×5 | 926 |
| ft10 | 10×10 | 930 | | la07 | 15×5 | 890 |
| ft20 | 20×5 | 1165 | | la08 | 15×5 | 863 |
| la01 | 10×5 | 666 | | la09 | 15×5 | 951 |
| la02 | 10×5 | 655 | | la10 | 15×5 | 958 |
| la03 | 10×5 | 597 | | la16 | 10×10 | 945 |
| la04 | 10×5 | 590 | | la17 | 10×10 | 784 |
| la05 | 10×5 | 593 | | la18 | 10×10 | 848 |
| | | | | la19 | 10×10 | 842 |
| | | | | la20 | 10×10 | 902 |

**ORB instances (extended OOD — not used during training or validation):**

| Instance | Size | BKS | | Instance | Size | BKS |
|----------|------|-----|-|----------|------|-----|
| orb01 | 10×10 | 1059 | | orb06 | 10×10 | 1010 |
| orb02 | 10×10 | 888  | | orb07 | 10×10 | 397  |
| orb03 | 10×10 | 1005 | | orb08 | 10×10 | 899  |
| orb04 | 10×10 | 1005 | | orb09 | 10×10 | 934  |
| orb05 | 10×10 | 887  | | orb10 | 10×10 | 944  |

---

## Repository Structure

```
cap-grpo/
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/                        ← numbered pipeline steps
│   ├── 01_download_data.py         step 1: download StarJob + OOD data
│   ├── 02_run_sft.py               step 2: supervised fine-tuning
│   ├── 03_run_grpo.py              step 3: CAP-GRPO reinforcement learning
│   ├── 04_eval_ood.py              step 4: quick check on 18 instances
│   ├── 05_audit_checker.py … 15_stage2_controls.py
│   │                               analysis, reading the raw-data folder
│   └── as_run/                     the files as they ran on the GPU node
│
├── cap_grpo/                       ← Python package
│   ├── config.py                   all hyperparameters and paths
│   ├── checker.py                  JSSP feasibility checker (no ML deps)
│   ├── checker_legacy.py           the earlier checker, kept for comparison
│   ├── reward.py                   CAP reward computation
│   ├── dataset.py                  download, sample, and load data
│   ├── rawdata.py                  find and read the raw-data folder
│   ├── model.py                    Unsloth model loading
│   ├── sft.py                      supervised fine-tuning
│   ├── grpo.py                     CAP-GRPO training
│   └── evaluator.py                generate + check + aggregate metrics
│
├── tests/
│   └── test_checker.py             21 hand-built cases, one rule each
│
├── docs/
│   ├── CHECKER.md                  the three rules the two checkers differ on
│   └── REPRODUCE.md                every reported number and its command
│
├── training_logs/                  ← small enough for git
│   ├── V1_feasibility_only/ … V8_hybrid_over_emission_penalty/
│   ├── sft_llama_lora/             evaluation loss at 49 checkpoints
│   ├── sft_llama_rslora/
│   └── stage2/
│       ├── test_2_1_random_reward/        seed42/, seed43/
│       ├── test_2_2_more_supervised/      lr5e-6/, lr2e-4/
│       └── test_2_3_rejection_sampling/   run/
│
├── data/
│   ├── instances/                  the problem sets, small enough for git
│   │   ├── ood_main_445.jsonl      the benchmark grid, 445 problems
│   │   ├── ood_taillard_10.jsonl   the 10 Taillard problems in the grid
│   │   ├── passk_90.jsonl          the 90 problems used for pass@k
│   │   ├── scale_90.jsonl          the duration-scaling probe
│   │   └── ta_probe_20.jsonl       the Taillard token-budget probe
│   └── starjob_train_sm.jsonl      written by 01_download_data.py (~108k)
│
└── outputs/                        ← created during training
    ├── sft_llama/final_adapter/
    └── cap_grpo_llama_hybrid/final_adapter/
```

The generations themselves are in the Drive folder; see [Raw data](#raw-data).

---

## Module Reference

### `cap_grpo.checker`

Standalone JSSP feasibility checker — **no ML dependencies**.
Can be used independently of the training code.

```python
from cap_grpo.checker import check_violations

jobs_spec = [[(1, 3), (0, 2)], [(0, 2), (1, 4)]]   # 2 jobs, 2 machines
schedule  = "J0-M1: 0+3->3, J0-M0: 3+2->5\nJ1-M0: 0+2->2, J1-M1: 3+4->7"

result = check_violations(schedule, jobs_spec)
# {
#   "feasible": True,
#   "makespan": 7,
#   "ops_emitted": 4,
#   "ops_expected": 4,
#   "missing_op_count": 0,
#   "over_op_count": 0,
#   "routing_order_violations": 0,
#   "machine_capacity_violations": 0,
#   "timing_consistency_violations": 0,
#   "precedence_violations": 0,
#   "total_violations": 0,
# }
```

### `cap_grpo.dataset`

```python
from cap_grpo.dataset import (
    download_starjob,       # download full StarJob from HuggingFace
    sample_sm,              # filter to SM instances
    load_starjob_sm,        # load train/test split
    download_ood,           # download OR-Library jobshop1.txt
    load_ood_benchmarks,    # parse jobshop1.txt → eval records
)

records = load_starjob_sm(split="train")   # 98% train split
test    = load_starjob_sm(split="test")    # 2% held-out test split
ood     = load_ood_benchmarks()            # 18 FT+LA instances
```

### `cap_grpo.reward`

```python
from cap_grpo.reward import compute_reward
from cap_grpo.checker import check_violations

v = check_violations(schedule_text, jobs_spec)
r = compute_reward(v, n_ops=30, bks=55, mode="hybrid")
```

---

## Datasets

| Tag | Source | Count | Role |
|-----|--------|-------|------|
| SM | StarJob (Abgaryan et al., 2025), filtered jobs≤10 × machines≤10 | ~108,000 | Training |
| FT | Fisher & Thompson (1963) — ft06/ft10/ft20 | 3 | OOD eval |
| LA | Lawrence (1984) — la01–la10, la16–la20 | 15 | OOD eval |
| ORB | Applegate & Cook (1991) — orb01–orb10, all 10×10 | 10 | Extended OOD eval |

The 2% test split (seed=42) is held out from **both** SFT and GRPO training.
The 18 OOD FT+LA instances are never used for training.

## References

- Abgaryan, H., Cazenave, T., & Harutyunyan, A. (2025). StarJob: Dataset for LLM-Driven Job Shop Scheduling. *arXiv preprint arXiv:2503.01877*.
- Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv preprint arXiv:2402.03300*.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.
- Kalajdzievski, D. (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. *arXiv preprint arXiv:2312.03732*.
- Grattafiori, A., et al. (2024). The Llama 3 Herd of Models. *arXiv preprint arXiv:2407.21783*.
- Fisher, H., & Thompson, G. L. (1963). Probabilistic Learning Combinations of Local Job-Shop Scheduling Rules. *Industrial Scheduling*, 225–251.
- Lawrence, S. (1984). *Resource Constrained Project Scheduling: An Experimental Investigation of Heuristic Scheduling Techniques*. Carnegie-Mellon University.
- Applegate, D., & Cook, W. (1991). A Computational Study of the Job-Shop Scheduling Problem. *ORSA Journal on Computing*, 3(2), 149–156.

---

## License

Code released for research purposes. Dependencies retain their own licenses:
Unsloth (Apache-2.0), TRL (Apache-2.0), LLaMA-3.1 (Meta Community License),
Qwen2 (Apache-2.0), Granite (Apache-2.0).
