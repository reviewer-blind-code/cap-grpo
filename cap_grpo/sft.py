"""Supervised Fine-Tuning (SFT) with Unsloth + TRL SFTTrainer.

Pipeline
--------
1. Load base model via Unsloth (4-bit QLoRA)
2. Attach LoRA adapter (rsLoRA by default)
3. Format StarJob SM records with Alpaca prompt template
4. Train with SFTTrainer
5. Save the final adapter

Typical usage:
    from cap_grpo.model import load_model
    from cap_grpo.dataset import load_starjob_sm
    from cap_grpo.sft import attach_lora, run_sft

    model, tokenizer = load_model(model_key="llama")
    model = attach_lora(model)
    records = load_starjob_sm(split="train")
    run_sft(model, tokenizer, records, output_dir="outputs/sft_llama")
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

from .config import (
    ALPACA_PROMPT,
    OUTPUTS_DIR,
    SFT_BATCH_SIZE,
    SFT_GRAD_ACCUM_STEPS,
    SFT_LEARNING_RATE,
    SFT_LORA_ALPHA,
    SFT_LORA_DROPOUT,
    SFT_LORA_R,
    SFT_LORA_TARGET_MODULES,
    SFT_LR_SCHEDULER,
    SFT_NUM_EPOCHS,
    SFT_SAVE_STEPS,
    SFT_USE_RSLORA,
    SFT_WARMUP_STEPS,
    SFT_WEIGHT_DECAY,
    MAX_SEQ_LENGTH,
    SEED,
)


def attach_lora(
    model,
    r: int = SFT_LORA_R,
    lora_alpha: int = SFT_LORA_ALPHA,
    lora_dropout: float = SFT_LORA_DROPOUT,
    use_rslora: bool = SFT_USE_RSLORA,
    target_modules: list = SFT_LORA_TARGET_MODULES,
):
    """Attach a LoRA adapter to the loaded base model.

    Args:
        model        : model returned by load_model(for_training=True)
        r            : LoRA rank
        lora_alpha   : LoRA scaling factor
        lora_dropout : dropout applied to LoRA weights
        use_rslora   : enable rank-stabilised LoRA (rsLoRA)
        target_modules: list of module names to apply LoRA to

    Returns:
        model with LoRA adapter attached
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=use_rslora,
        loftq_config=None,
    )


def _format_records(records: list, tokenizer) -> Dataset:
    """Format StarJob records into Alpaca prompt+response pairs for SFT."""
    eos = tokenizer.eos_token or ""

    def _fmt(r):
        text = (
            ALPACA_PROMPT.format(instruction=r["instruction"], input=r["input"])
            + r["output"]
            + eos
        )
        return {"text": text}

    return Dataset.from_list([_fmt(r) for r in records])


def run_sft(
    model,
    tokenizer,
    records: list,
    output_dir: str | Path | None = None,
    num_epochs: int = SFT_NUM_EPOCHS,
    learning_rate: float = SFT_LEARNING_RATE,
    batch_size: int = SFT_BATCH_SIZE,
    grad_accum: int = SFT_GRAD_ACCUM_STEPS,
    warmup_steps: int = SFT_WARMUP_STEPS,
    save_steps: int = SFT_SAVE_STEPS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> str:
    """Run supervised fine-tuning and save the final adapter.

    Args:
        model         : model with LoRA adapter (from attach_lora)
        tokenizer     : matching tokenizer
        records       : StarJob SM training records (from load_starjob_sm)
        output_dir    : directory to save checkpoints and final adapter
        num_epochs    : number of training epochs
        learning_rate : optimizer learning rate
        batch_size    : per-device batch size
        grad_accum    : gradient accumulation steps
        warmup_steps  : linear warmup steps
        save_steps    : save a checkpoint every N steps
        max_seq_length: maximum sequence length (must match model load setting)

    Returns:
        str path to the final adapter directory
    """
    os.environ.setdefault("WANDB_MODE", "offline")

    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR / "sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _format_records(records, tokenizer)
    print(f"[sft] Training on {len(dataset):,} records | epochs={num_epochs} lr={learning_rate}")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        optim="adamw_8bit",
        weight_decay=SFT_WEIGHT_DECAY,
        lr_scheduler_type=SFT_LR_SCHEDULER,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=10,
        seed=SEED,
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer.train()

    final_dir = out_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] Saved final adapter → {final_dir}")
    return str(final_dir)
