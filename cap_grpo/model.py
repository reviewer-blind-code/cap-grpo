"""Model loading via Unsloth FastLanguageModel.

Supports two loading modes:
    model_key   : registered name → auto-downloads from HuggingFace Hub
    model_path  : local path to a base model or saved LoRA adapter

Unsloth's FastLanguageModel is required; vanilla HuggingFace PEFT is not supported.
"""
from __future__ import annotations

import torch
from unsloth import FastLanguageModel

from .config import MAX_SEQ_LENGTH, MODEL_REGISTRY


def _ensure_cached(repo_id: str) -> None:
    """Download model from HF Hub if not already in the local cache (idempotent)."""
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    cached = try_to_load_from_cache(repo_id, "config.json")
    if isinstance(cached, str):
        return
    print(f"[model] Downloading {repo_id!r} (~4-5 GB, first run only) ...")
    snapshot_download(repo_id, ignore_patterns=["*.pt", "original/"])
    print(f"[model] Download complete: {repo_id}")


def load_model(
    model_key: str | None = None,
    model_path: str | None = None,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    for_training: bool = True,
):
    """Load model and tokenizer.

    Provide exactly one of model_key or model_path.

    Args:
        model_key      : key from MODEL_REGISTRY (llama / qwen2 / granite / ministral)
        model_path     : local path to a base model or SFT adapter directory
        max_seq_length : context window (must match SFT training value)
        load_in_4bit   : enable 4-bit quantization (QLoRA)
        dtype          : torch dtype (bfloat16 recommended for Ampere+ GPUs)
        for_training   : True → model.train(); False → FastLanguageModel.for_inference()

    Returns:
        (model, tokenizer) tuple
    """
    if (model_key is None) == (model_path is None):
        raise ValueError("Provide exactly one of model_key or model_path.")

    if model_key is not None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_key {model_key!r}. Choices: {list(MODEL_REGISTRY)}")
        spec         = MODEL_REGISTRY[model_key]
        name_or_path = spec["hf_name"]
        eager        = spec.get("eager", False)
        label        = spec["label"]
        _ensure_cached(name_or_path)
    else:
        name_or_path = model_path
        eager        = False
        label        = model_path

    print(f"[model] Loading: {label}  ({name_or_path})")
    kwargs = dict(
        model_name=name_or_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if eager:
        kwargs["attn_implementation"] = "eager"

    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    if for_training:
        model.train()
    else:
        FastLanguageModel.for_inference(model)

    return model, tokenizer
