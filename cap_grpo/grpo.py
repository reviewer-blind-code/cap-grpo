"""CAP-GRPO training: Group Relative Policy Optimization with constraint-aware reward.

Reward modes
------------
hybrid (default V4): 7-component constraint-aware reward
hybrid_v7           : hybrid + over-emit penalty R_O
stratified          : V1 per-category weighted penalty (ablation)
uniform             : {0 unparseable, 1 infeasible, 7 feasible}

Length control (V5/V6 technique)
---------------------------------
Completions whose token count exceeds OVERLEN_FACTOR × gold_est get their GRPO
advantage zeroed — no gradient contribution. This removes the length-escape
collapse trigger without imposing a reward cliff.

Usage:
    from cap_grpo.model import load_model
    from cap_grpo.dataset import load_starjob_sm
    from cap_grpo.grpo import run_grpo

    model, tokenizer = load_model(model_path="outputs/sft/final_adapter")
    records = load_starjob_sm(split="train")
    run_grpo(model, tokenizer, records, run_name="cap_grpo_v1")
"""
from __future__ import annotations

import json
import os

import unsloth  # noqa: F401 — must be imported before trl/transformers

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from .checker import check_violations
from .config import (
    DEFAULT_REWARD_MODE,
    GOLD_EST_BASE,
    GOLD_EST_SLOPE,
    GRAD_ACCUM_STEPS,
    K_SAMPLES,
    KL_COEF,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_GRAD_NORM,
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_STEPS,
    OUTPUTS_DIR,
    OVERLEN_FACTOR,
    SAVE_EVERY,
    SEED,
    TEMPERATURE,
    WARMUP_STEPS,
)
from .reward import compute_reward

_TOKENIZER_REF: dict = {"tok": None}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _serialize_jobs(jobs_spec: list) -> str:
    return json.dumps(jobs_spec)


def _deserialize_jobs(s: str) -> list:
    return [[tuple(op) for op in job] for job in json.loads(s)]


def build_grpo_dataset(records: list) -> Dataset:
    """Convert StarJob records to a HuggingFace Dataset for GRPOTrainer."""
    rows = [{
        "prompt":    r["prompt"],
        "jobs_spec": _serialize_jobs(r["jobs_spec"]),
        "bks":       r["bks"] or 0,
        "n_ops":     r["n_ops"],
    } for r in records]
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _token_length(text: str):
    tok = _TOKENIZER_REF.get("tok")
    if tok is None:
        return max(1, len(text) // 4), False
    ids    = tok.encode(text, add_special_tokens=False)
    eos_id = tok.eos_token_id
    ended  = (eos_id is not None) and (len(ids) > 0) and (ids[-1] == eos_id)
    return len(ids), ended


def make_reward_fn(mode: str, lp_alpha: float = 0.10, eos_beta: float = 0.05):
    """Build a GRPO-compatible reward function.

    The returned function accepts a batch of completions and their associated
    problem metadata, and returns a list of scalar rewards.

    Args:
        mode     : reward mode (hybrid | hybrid_v7 | stratified | uniform)
        lp_alpha : length-penalty coefficient (stratified_v2 only)
        eos_beta : EOS bonus coefficient (stratified_v2 only)

    Returns:
        reward_fn(completions, jobs_spec, bks, n_ops, **kwargs) → list[float]
    """
    needs_len = (mode == "stratified_v2")

    def reward_fn(completions, jobs_spec, bks, n_ops, **kwargs):
        rewards, n_parseable, n_feasible = [], 0, 0
        for comp, js_json, b, n in zip(completions, jobs_spec, bks, n_ops):
            text    = comp if isinstance(comp, str) else comp[0]["content"]
            js      = _deserialize_jobs(js_json)
            v       = check_violations(text, js)
            bks_val = None if b in (0, None) else int(b)

            if needs_len:
                gen_len, ended = _token_length(text)
                r = compute_reward(v, int(n), bks_val, mode=mode,
                                   gen_len=gen_len, ended_with_eos=ended,
                                   lp_alpha=lp_alpha, eos_beta=eos_beta)
            else:
                r = compute_reward(v, int(n), bks_val, mode=mode)

            rewards.append(float(r))
            if (v["ops_emitted"] + v["timing_consistency_violations"]) > 0:
                n_parseable += 1
            if v["feasible"]:
                n_feasible += 1

        k      = len(rewards)
        mean_r = sum(rewards) / k if k else 0.0
        std_r  = (sum((x - mean_r) ** 2 for x in rewards) / k) ** 0.5 if k else 0.0
        print(
            f"[reward:{mode}] n={k} parseable={n_parseable}/{k} "
            f"feasible={n_feasible}/{k} r_mean={mean_r:.3f} r_std={std_r:.3f} "
            f"r={[round(x, 2) for x in rewards]}",
            flush=True,
        )
        return rewards

    reward_fn.__name__ = f"cap_grpo_reward_{mode}"
    return reward_fn


# ---------------------------------------------------------------------------
# Length-controlled trainer
# ---------------------------------------------------------------------------

class LengthControlledGRPOTrainer(GRPOTrainer):
    """Zero advantages for completions that exceed OVERLEN_FACTOR × gold_est tokens.

    Over-length completions produce no gradient (advantage = 0) rather than a
    penalty reward, which avoids reward cliffs while still preventing the
    length-escape collapse mode seen in V2/V3 runs.
    """

    def _prepare_inputs(self, inputs):
        out  = super()._prepare_inputs(inputs)
        adv  = out["advantages"]
        clen = out["completion_mask"].sum(dim=1).float()

        if len(adv) != len(inputs):
            return out

        n_ops    = torch.tensor(
            [float(x["n_ops"]) for x in inputs],
            device=adv.device, dtype=torch.float,
        )
        gold_est = GOLD_EST_SLOPE * n_ops + GOLD_EST_BASE
        over     = clen > (OVERLEN_FACTOR * gold_est)
        n_over   = int(over.sum().item())

        if n_over:
            out["advantages"] = torch.where(over, torch.zeros_like(adv), adv)

        self._metrics["overlen_frac"].append(over.float().mean().item())
        print(
            f"[lenctrl] masked {n_over}/{len(over)} over-length "
            f"| clen_max={int(clen.max().item())}",
            flush=True,
        )
        return out


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_grpo(
    model,
    tokenizer,
    records: list,
    run_name: str = "cap_grpo",
    reward_mode: str = DEFAULT_REWARD_MODE,
    max_steps: int = NUM_TRAIN_STEPS,
    length_control: bool = False,
    resume_from: str | None = None,
    kl_coef: float = KL_COEF,
    grad_accum: int = GRAD_ACCUM_STEPS,
    temperature: float = TEMPERATURE,
    learning_rate: float = LEARNING_RATE,
    save_every: int = SAVE_EVERY,
    lp_alpha: float = 0.10,
    eos_beta: float = 0.05,
) -> str:
    """Run CAP-GRPO fine-tuning and return the path to the final saved adapter.

    Args:
        model          : model loaded via load_model(for_training=True)
        tokenizer      : matching tokenizer
        records        : StarJob SM training records (from load_starjob_sm)
        run_name       : name of this run (used as output directory name)
        reward_mode    : hybrid | hybrid_v7 | stratified | uniform
        max_steps      : total GRPO training steps
        length_control : zero advantages for over-length completions (V5/V6)
        resume_from    : path to a checkpoint directory to resume from
        kl_coef        : KL divergence penalty coefficient (beta)
        grad_accum     : gradient accumulation steps
        temperature    : sampling temperature during training
        learning_rate  : optimizer learning rate
        save_every     : save a checkpoint every N steps
        lp_alpha       : length-penalty coefficient (stratified_v2 only)
        eos_beta       : EOS bonus coefficient (stratified_v2 only)

    Returns:
        str path to the final adapter directory
    """
    _TOKENIZER_REF["tok"] = tokenizer
    os.environ.setdefault("WANDB_MODE", "offline")

    print(
        f"[grpo] mode={reward_mode} K={K_SAMPLES} steps={max_steps} "
        f"length_control={length_control} resume={resume_from}"
    )

    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_grpo_dataset(records)

    config = GRPOConfig(
        output_dir=str(run_dir),
        run_name=run_name,
        learning_rate=learning_rate,
        warmup_steps=WARMUP_STEPS,
        max_steps=max_steps,
        per_device_train_batch_size=K_SAMPLES,
        gradient_accumulation_steps=grad_accum,
        num_generations=K_SAMPLES,
        max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=temperature,
        beta=kl_coef,
        max_grad_norm=MAX_GRAD_NORM,
        save_steps=save_every,
        save_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to=["none"],
        seed=SEED,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        use_vllm=False,
    )

    reward_fn   = make_reward_fn(reward_mode, lp_alpha=lp_alpha, eos_beta=eos_beta)
    trainer_cls = LengthControlledGRPOTrainer if length_control else GRPOTrainer
    trainer     = trainer_cls(
        model=model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    print(f"[grpo] Saved final adapter → {final_dir}")
    return str(final_dir)
