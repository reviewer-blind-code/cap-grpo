"""Central configuration: paths, model registry, training hyperparameters, reward settings."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

SM_TRAIN_FILE = DATA_DIR / "starjob_train_sm.jsonl"
JOBSHOP1      = DATA_DIR / "benchmarks" / "jobshop1.txt"

# ---------------------------------------------------------------------------
# Prompt template (Alpaca format)
# ---------------------------------------------------------------------------
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "    ### Instruction:\n    {instruction}\n\n"
    "    ### Input:\n    {input}\n\n"
    "    ### Response:\n    "
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "llama": {
        "hf_name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "eager": False,
        "label": "LLaMA-3.1-8B-Instruct (4-bit)",
    },
    "qwen2": {
        "hf_name": "unsloth/Qwen2-7B-Instruct-bnb-4bit",
        "eager": False,
        "label": "Qwen2-7B-Instruct (4-bit)",
    },
    "granite": {
        "hf_name": "unsloth/granite-3.2-8b-instruct-bnb-4bit",
        "eager": True,
        "label": "Granite-3.2-8B-Instruct (4-bit)",
    },
    "ministral": {
        "hf_name": "mistralai/Ministral-8B-Instruct-2410",
        "eager": True,
        "label": "Ministral-8B-Instruct-2410 (4-bit)",
    },
}

# ---------------------------------------------------------------------------
# Dataset / split
# ---------------------------------------------------------------------------
SPLIT_SEED = 42
TEST_FRAC  = 0.02   # 2% held-out; must match SFT to keep test set identical

OOD_INSTANCES = [
    "ft06", "ft10", "ft20",
    "la01", "la02", "la03", "la04", "la05",
    "la06", "la07", "la08", "la09", "la10",
    "la16", "la17", "la18", "la19", "la20",
]

BEST_KNOWN = {
    "ft06":  55, "ft10":  930, "ft20": 1165,
    "la01": 666, "la02":  655, "la03":  597, "la04": 590, "la05": 593,
    "la06": 926, "la07":  890, "la08":  863, "la09": 951, "la10": 958,
    "la16": 945, "la17":  784, "la18":  848, "la19": 842, "la20": 902,
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.7
TOP_P          = 0.95

# ---------------------------------------------------------------------------
# SFT hyperparameters
# ---------------------------------------------------------------------------
SFT_LORA_R             = 32
SFT_LORA_ALPHA         = 32
SFT_LORA_DROPOUT       = 0.0
SFT_USE_RSLORA         = True
SFT_LEARNING_RATE      = 2e-4
SFT_NUM_EPOCHS         = 1
SFT_BATCH_SIZE         = 1
SFT_GRAD_ACCUM_STEPS   = 8
SFT_WARMUP_STEPS       = 5
SFT_SAVE_STEPS         = 200
SFT_WEIGHT_DECAY       = 0.01
SFT_LR_SCHEDULER       = "linear"

SFT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------------------
# GRPO hyperparameters
# ---------------------------------------------------------------------------
K_SAMPLES        = 4       # completions sampled per prompt per step
LEARNING_RATE    = 5e-6
NUM_TRAIN_STEPS  = 500
SAVE_EVERY       = 50
LOGGING_STEPS    = 5
KL_COEF          = 0.05
MAX_GRAD_NORM    = 1.0
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS     = 20
SEED             = 42

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
DEFAULT_REWARD_MODE = "hybrid"  # hybrid | hybrid_v7 | stratified | uniform

# V1 stratified weights — frozen; derived from rsLoRA-SFT violation statistics
V1_WEIGHTS = {
    "missing_op_count":               4 / 233,
    "routing_order_violations":      33 / 233,
    "machine_capacity_violations":  145 / 233,
    "timing_consistency_violations": 33 / 233,
    "precedence_violations":         18 / 233,
}

# Length-control: advantage masking for over-long completions (V5/V6 technique)
GOLD_EST_SLOPE = 12.5   # gold_est = slope * N_ops + base
GOLD_EST_BASE  = 50
OVERLEN_FACTOR = 2.0    # completions > factor * gold_est get advantage zeroed
