"""Step 2 — Supervised Fine-Tuning (SFT).

Trains a LoRA adapter on the StarJob SM dataset using Unsloth + TRL.
Starting GRPO from an SFT adapter significantly improves sample quality
because the model already knows the output format before RL begins.

Run:
    # From a registered base model (auto-downloads from HuggingFace):
    python scripts/02_run_sft.py --model llama

    # Resume from checkpoint:
    python scripts/02_run_sft.py --model llama --resume-from outputs/sft_llama/checkpoint-400

Options:
    --model         base model key: llama | qwen2 | granite | ministral
    --model-path    local path to a base model (alternative to --model)
    --output-dir    output directory (default: outputs/sft_<model>)
    --epochs        number of training epochs (default: 1)
    --lr            learning rate (default: 2e-4)
    --batch-size    per-device batch size (default: 1)
    --grad-accum    gradient accumulation steps (default: 8)
    --max-records   limit number of training records (default: all)
    --lora-r        LoRA rank (default: 32)
    --no-rslora     disable rsLoRA (rsLoRA is on by default)
    --resume-from   checkpoint directory to resume from
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from cap_grpo.config import MODEL_REGISTRY, OUTPUTS_DIR, SFT_LORA_R
from cap_grpo.dataset import load_starjob_sm
from cap_grpo.model import load_model
from cap_grpo.sft import attach_lora, run_sft


def main():
    parser = argparse.ArgumentParser(
        description="SFT: train LoRA adapter on StarJob SM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model",      choices=list(MODEL_REGISTRY),
                     help="Registered base model key")
    src.add_argument("--model-path", metavar="PATH",
                     help="Local path to base model")

    parser.add_argument("--output-dir",  default=None,
                        help="Output directory (default: outputs/sft_<model>)")
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--max-records", type=int,   default=None,
                        help="Limit training records (default: full 98%% SM train)")
    parser.add_argument("--lora-r",      type=int,   default=SFT_LORA_R)
    parser.add_argument("--no-rslora",   action="store_true")
    parser.add_argument("--resume-from", default=None, metavar="CKPT_DIR")
    parser.add_argument("--dtype",       choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    label      = args.model or Path(args.model_path).name
    output_dir = args.output_dir or str(OUTPUTS_DIR / f"sft_{label}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(
        model_key=args.model,
        model_path=args.model_path,
        dtype=dtype,
        for_training=True,
    )

    model = attach_lora(
        model,
        r=args.lora_r,
        use_rslora=not args.no_rslora,
    )

    records = load_starjob_sm(split="train", limit=args.max_records)
    print(f"[sft] Training records: {len(records):,}")

    final_dir = run_sft(
        model, tokenizer, records,
        output_dir=output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    print(f"\nSFT complete. Final adapter: {final_dir}")
    print(f"Next step: python scripts/03_run_grpo.py --model-path {final_dir}")


if __name__ == "__main__":
    main()
