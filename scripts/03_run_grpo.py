"""Step 3 — CAP-GRPO Training.

Fine-tunes the SFT adapter with Group Relative Policy Optimization using a
constraint-aware reward (CAP). Starting from an SFT adapter is strongly
recommended — base-model GRPO rarely converges within 500 steps.

Run:
    # From an SFT adapter (recommended):
    python scripts/03_run_grpo.py --model-path outputs/sft_llama/final_adapter

    # From a registered base model (not recommended, but possible):
    python scripts/03_run_grpo.py --model llama

    # With length control (zero advantage for over-long completions):
    python scripts/03_run_grpo.py --model-path outputs/sft_llama/final_adapter --length-control

    # Smoke test (10 steps, 50 records):
    python scripts/03_run_grpo.py --model-path outputs/sft_llama/final_adapter \\
        --max-steps 10 --max-records 50

    # Resume from checkpoint:
    python scripts/03_run_grpo.py --model-path outputs/cap_grpo_v1/checkpoint-200 \\
        --resume-from outputs/cap_grpo_v1/checkpoint-200 --max-steps 500

Options:
    --model / --model-path   base model key or local adapter path
    --reward-mode            hybrid | hybrid_v7 | stratified | uniform
    --max-steps              total GRPO steps (default: 500)
    --max-records            limit training records (default: full 98%% SM train)
    --length-control         zero advantages for completions > 2× gold estimate
    --kl-coef                KL penalty coefficient (default: 0.05)
    --lr                     learning rate (default: 5e-6)
    --temperature            sampling temperature (default: 0.7)
    --run-name               override the output directory name
    --resume-from            checkpoint directory to resume from
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from cap_grpo.config import (
    DEFAULT_REWARD_MODE, GRAD_ACCUM_STEPS, KL_COEF,
    LEARNING_RATE, MODEL_REGISTRY, NUM_TRAIN_STEPS,
    SAVE_EVERY, TEMPERATURE,
)
from cap_grpo.dataset import load_starjob_sm
from cap_grpo.grpo import run_grpo
from cap_grpo.model import load_model


def main():
    parser = argparse.ArgumentParser(
        description="CAP-GRPO: constraint-aware RL fine-tuning for JSSP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model",      choices=list(MODEL_REGISTRY),
                     help="Registered base model key")
    src.add_argument("--model-path", metavar="PATH",
                     help="Local path to SFT adapter or base model")

    parser.add_argument("--reward-mode",   choices=["hybrid", "hybrid_v7", "stratified", "uniform"],
                        default=DEFAULT_REWARD_MODE)
    parser.add_argument("--max-steps",     type=int,   default=NUM_TRAIN_STEPS)
    parser.add_argument("--max-records",   type=int,   default=None,
                        help="Limit training records (default: full 98%% SM train)")
    parser.add_argument("--length-control", action="store_true")
    parser.add_argument("--kl-coef",       type=float, default=KL_COEF)
    parser.add_argument("--lr",            type=float, default=LEARNING_RATE)
    parser.add_argument("--temperature",   type=float, default=TEMPERATURE)
    parser.add_argument("--grad-accum",    type=int,   default=GRAD_ACCUM_STEPS)
    parser.add_argument("--save-every",    type=int,   default=SAVE_EVERY)
    parser.add_argument("--run-name",      default=None)
    parser.add_argument("--resume-from",   default=None, metavar="CKPT_DIR")
    parser.add_argument("--dtype",         choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    label    = args.model or Path(args.model_path).name
    run_name = args.run_name or f"cap_grpo_{label}_{args.reward_mode}"

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(
        model_key=args.model,
        model_path=args.model_path,
        dtype=dtype,
        for_training=True,
    )

    records = load_starjob_sm(split="train", limit=args.max_records)
    print(f"[grpo] Training records: {len(records):,}")

    final_dir = run_grpo(
        model, tokenizer, records,
        run_name=run_name,
        reward_mode=args.reward_mode,
        max_steps=args.max_steps,
        length_control=args.length_control,
        resume_from=args.resume_from,
        kl_coef=args.kl_coef,
        grad_accum=args.grad_accum,
        temperature=args.temperature,
        learning_rate=args.lr,
        save_every=args.save_every,
    )

    print(f"\nGRPO complete. Final adapter: {final_dir}")
    print(f"Next step: python scripts/04_eval_ood.py --model-path {final_dir}")


if __name__ == "__main__":
    main()
