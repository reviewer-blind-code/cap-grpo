"""Stage 2-2 and 2-3: supervised training at the same budget as one RL run.

The RL runs take 500 optimizer steps with gradient accumulation 4 and one prompt
per micro-batch, so each run sees 2,000 prompts and makes 500 weight updates.
This script does the same number of updates over the same number of sequences,
from the same supervised checkpoint, with a plain next-token loss.

  2-2  --data starjob   the gold schedules of the same 2,000 training records
  2-3  --data PATH      schedules the model produced itself and the checker
                        accepted (written by grpo_jssp/sample_rft.py)

Run on the GPU node, from the repo root:
    venv-grpo/bin/python -u -m grpo_jssp.sft_budget \
        --sft-checkpoint output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800 \
        --run-name sft_budget_lr5e6 --lr 5e-6
"""
import os
os.environ["UNSLOTH_NUM_PROC"] = "1"
os.environ.setdefault("WANDB_MODE", "offline")

import argparse
import json
from pathlib import Path

import unsloth  # noqa: F401  (must precede transformers/trl)
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from grpo_jssp.config import (
    SFT_CHECKPOINT, STARJOB_SM_PATH, MAX_SEQ_LENGTH, LEARNING_RATE,
    MAX_GRAD_NORM, WARMUP_STEPS, OUTPUT_DIR, SEED,
)
from grpo_jssp.data_utils import load_starjob_sm, format_prompt


def build_rows(data: str, max_records: int, eos: str) -> list:
    """One text per record: the alpaca prompt the models were fine-tuned on,
    the schedule, and the end-of-sequence token."""
    if data == "starjob":
        recs = load_starjob_sm(STARJOB_SM_PATH, limit=max_records, split="train")
        return [{"text": r["prompt"] + r["output"] + eos} for r in recs]
    rows = []
    for line in Path(data).read_text().splitlines():
        d = json.loads(line)
        prompt = d.get("prompt") or format_prompt(d["instruction"], d["input"])
        rows.append({"text": prompt + d["output"] + eos})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--data", default="starjob",
                    help="'starjob' for the gold schedules, or a jsonl of kept samples")
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args()

    print(f"[sft-budget] ckpt={args.sft_checkpoint} data={args.data} "
          f"steps={args.max_steps} grad_accum={args.grad_accum} lr={args.lr} "
          f"seed={args.seed}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    model.train()

    rows = build_rows(args.data, args.max_records, tokenizer.eos_token)
    print(f"[sft-budget] sequences: {len(rows)}", flush=True)
    train_ds = Dataset.from_list(rows)

    run_dir = OUTPUT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = SFTConfig(
        output_dir=str(run_dir),
        run_name=args.run_name,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,                       # one record per sequence, as in the RL run
        dataset_num_proc=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="linear",
        logging_steps=5,
        save_steps=args.save_every,
        save_strategy="steps",
        report_to=["wandb"],
        seed=args.seed,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         processing_class=tokenizer)
    trainer.train()
    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    print(f"[sft-budget] saved final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
