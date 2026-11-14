#!/bin/bash
# Stage 2-1, rerun from the LoRA checkpoint.
# The October pair started from the rsLoRA checkpoint because the launcher left
# out --sft-checkpoint. Everything else is unchanged, with one correction: the
# random reward stream is now seeded from the training seed, so the two runs are
# independent draws of the noise instead of sharing one stream.
# Waits for any Stage 2a work to finish first; one GPU, one job at a time.

cd /home/user/Documents/Starjob
source venv-grpo/bin/activate
source /home/user/Documents/Starjob/grpo_jssp/_cuda_env.sh
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0

ROOT=/home/user/Documents/Starjob
RUNS=$ROOT/grpo_jssp/runs
LOGS=$ROOT/logs/rerun21
SFT_CKPT=$ROOT/output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800
SUMMARY=$LOGS/summary.log
mkdir -p "$LOGS"
say() { echo "=== $(date '+%F %T') $*" | tee -a "$SUMMARY"; }

while pgrep -f _stage2a.sh >/dev/null; do sleep 300; done
say "stage 2a finished, starting the 2-1 rerun"

for seed in 42 43; do
  name=full_lora_random_lora_n2000_s$seed
  if [ -d "$RUNS/$name/final_adapter" ]; then say "[$name] already done, skipping"; continue; fi
  say "[$name] start"
  RANDOM_REWARD_SEED=$seed python -u -m grpo_jssp.run train \
    --reward-mode random --seed "$seed" --sft-checkpoint "$SFT_CKPT" \
    --max-records 2000 --max-steps 500 --run-name "$name" \
    >> "$LOGS/$name.log" 2>&1
  say "[$name] end exit=$?"
done

deactivate 2>/dev/null
source venv-vllm/bin/activate
MODELS=""
[ -d "$RUNS/full_lora_random_lora_n2000_s42/final_adapter" ] && MODELS="$MODELS RANDL42"
[ -d "$RUNS/full_lora_random_lora_n2000_s43/final_adapter" ] && MODELS="$MODELS RANDL43"
if [ -n "$MODELS" ]; then
  say "[eval grid] start$MODELS"
  python -u evaluation/run_grid.py --models $MODELS --seeds 0 1 2 3 4 \
    --out grid_random_lora >> "$LOGS/grid.log" 2>&1
  say "[eval grid] end exit=$?"
fi
say "rerun done"
