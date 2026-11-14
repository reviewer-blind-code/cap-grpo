#!/bin/bash
# Stage 2a, experiments 2-2 and 2-3, then the evaluation of all three adapters.
# Sequential: one GPU, one phase at a time. No set -e; a phase that fails is
# reported and the pipeline goes on, so a later phase still gets its chance.
# A phase whose final adapter already exists is skipped, so this can be rerun.

cd /home/user/Documents/Starjob
source venv-grpo/bin/activate
source /home/user/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0

ROOT=/home/user/Documents/Starjob
RUNS=$ROOT/grpo_jssp/runs
LOGS=$ROOT/logs/stage2a
SFT_CKPT=$ROOT/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800
SUMMARY=$LOGS/summary.log
mkdir -p "$LOGS"

say() { echo "=== $(date '+%F %T') $*" | tee -a "$SUMMARY"; }

phase() {                      # phase <label> <logfile> <command...>
  local label="$1"; shift
  local log="$1"; shift
  say "[$label] start"
  "$@" >> "$log" 2>&1
  local ec=$?
  say "[$label] end exit=$ec"
  return $ec
}

if [ ! -d "$SFT_CKPT" ]; then say "ERROR: no LoRA checkpoint at $SFT_CKPT"; exit 1; fi
say "stage 2a start"

# --- 2-2: supervised training at the budget of one RL run -------------------
# Two learning rates: the one the RL runs use, and the one the original
# supervised stage used. Either could be called the fair control, so both run.
for pair in "lr5e6 5e-6" "lr2e4 2e-4"; do
  set -- $pair
  tag=$1; lr=$2
  if [ -d "$RUNS/sft_budget_$tag/final_adapter" ]; then
    say "[2-2 $tag] already done, skipping"
    continue
  fi
  phase "2-2 $tag" "$LOGS/sft_budget_$tag.log" \
    python -u -m grpo_jssp.sft_budget \
      --sft-checkpoint "$SFT_CKPT" --data starjob --max-records 2000 \
      --max-steps 500 --grad-accum 4 --lr "$lr" --seed 42 \
      --run-name "sft_budget_$tag"
done

# --- 2-3 step 1: sample from the supervised model, keep what the checker takes
if [ -f "$ROOT/results/rft_samples/kept.jsonl" ]; then
  say "[2-3 sample] already done, skipping"
else
  deactivate 2>/dev/null
  source venv-vllm/bin/activate
  phase "2-3 sample" "$LOGS/rft_sample.log" \
    python -u -m grpo_jssp.sample_rft \
      --adapter output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800 \
      --max-records 2000 --k 4 --temperature 0.7 --seed 42 \
      --out results/rft_samples
  deactivate 2>/dev/null
  source venv-grpo/bin/activate
fi

# --- 2-3 step 2: fine-tune on those schedules, same budget ------------------
if [ -d "$RUNS/rft_budget/final_adapter" ]; then
  say "[2-3 train] already done, skipping"
elif [ -f "$ROOT/results/rft_samples/kept.jsonl" ]; then
  phase "2-3 train" "$LOGS/rft_train.log" \
    python -u -m grpo_jssp.sft_budget \
      --sft-checkpoint "$SFT_CKPT" --data results/rft_samples/kept.jsonl \
      --max-steps 500 --grad-accum 4 --lr 5e-6 --seed 42 \
      --run-name rft_budget
else
  say "[2-3 train] no kept.jsonl, skipping"
fi

# --- evaluation on the same grid as every other model -----------------------
deactivate 2>/dev/null
source venv-vllm/bin/activate
MODELS=""
[ -d "$RUNS/sft_budget_lr5e6/final_adapter" ] && MODELS="$MODELS SFTbudget5e6"
[ -d "$RUNS/sft_budget_lr2e4/final_adapter" ] && MODELS="$MODELS SFTbudget2e4"
[ -d "$RUNS/rft_budget/final_adapter" ]       && MODELS="$MODELS RFT"
if [ -n "$MODELS" ]; then
  phase "eval grid" "$LOGS/grid.log" \
    python -u evaluation/run_grid.py --models $MODELS --seeds 0 1 2 3 4 \
      --out grid_stage2a
else
  say "[eval grid] no adapters to evaluate"
fi

say "stage 2a done"
