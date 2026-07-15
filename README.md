# CAP-GRPO: Constraint-Aware Policy with GRPO for Job-Shop Scheduling

Fine-tune an LLM to solve the **Job-Shop Scheduling Problem (JSSP)** using
**GRPO** (Shao et al., 2024), guided by a constraint-aware reward function
(CAP) with a coverage gate that prevents reward hacking from vacuous outputs.

## Results (LLaMA-3.1-8B, V6 CAP-GRPO)

| Metric | CAP-GRPO V6 | o3-mini (zero-shot) |
|--------|-------------|---------------------|
| Feasible (18 OOD) | **8/18 (44%)** | 7/18 (39%) |
| Mean gap to BKS | **+5.5%** | +182.7% |
| Cost | offline | $1.11 / run |

---

## Pipeline Overview

```
01_download_data.py       download StarJob from HuggingFace + OOD from OR-Library
02_run_sft.py             supervised fine-tuning (SFT) with Alpaca format
03_run_grpo.py            CAP-GRPO reinforcement learning from SFT adapter
04_eval_ood.py            evaluate on 18 OOD FT+LA benchmark instances
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
dramatically improves GRPO convergence speed and stability. We use LoRA
(Hu et al., 2021) with rank-stabilised scaling (rsLoRA; Kalajdzievski, 2023).

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
| rsLoRA | enabled | rank-stabilised scaling |
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
│   └── 04_eval_ood.py              step 4: OOD evaluation + debug inference
│
├── cap_grpo/                       ← Python package
│   ├── __init__.py
│   ├── config.py                   all hyperparameters and paths
│   ├── checker.py                  JSSP feasibility checker (no ML deps)
│   ├── reward.py                   CAP reward computation
│   ├── dataset.py                  download, sample, and load data
│   ├── model.py                    Unsloth model loading
│   ├── sft.py                      supervised fine-tuning
│   ├── grpo.py                     CAP-GRPO training
│   └── evaluator.py                generate + check + aggregate metrics
│
├── data/                           ← populated by 01_download_data.py
│   ├── starjob_train_sm.jsonl      SM training set (~108k records)
│   └── benchmarks/
│       └── jobshop1.txt            OR-Library: FT + LA + other instances
│
└── outputs/                        ← created during training
    ├── sft_llama/
    │   └── final_adapter/
    └── cap_grpo_llama_hybrid/
        └── final_adapter/
```

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

---

## License

Code released for research purposes. Dependencies retain their own licenses:
Unsloth (Apache-2.0), TRL (Apache-2.0), LLaMA-3.1 (Meta Community License),
Qwen2 (Apache-2.0), Granite (Apache-2.0).
