"""Evaluation utilities: generate schedules and compute feasibility + gap metrics.

Two evaluation sets:
    SM  : in-distribution, held-out 2% test split from StarJob SM
    OOD : out-of-distribution, 18 Fisher-Thompson and Lawrence instances
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel

from .checker import check_violations
from .config import MAX_NEW_TOKENS, MAX_SEQ_LENGTH
from .dataset import load_ood_benchmarks, load_starjob_sm


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, float]:
    """Generate one schedule response and return (text, elapsed_seconds)."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - max_new_tokens,
    ).to(model.device)

    t0  = time.time()
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    elapsed  = time.time() - t0
    gen_ids  = out[0][enc["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response, elapsed


def _aggregate(results: list) -> dict:
    """Compute summary statistics over a list of per-instance result dicts."""
    feas = [r for r in results if r["feasible"]]
    gaps = [
        (r["makespan"] - r["bks"]) / r["bks"]
        for r in feas
        if r.get("bks") and r.get("makespan")
    ]
    return {
        "n_total":                           len(results),
        "n_feasible":                        len(feas),
        "feasibility_rate":                  len(feas) / max(len(results), 1),
        "mean_gap_to_bks":                   statistics.mean(gaps) if gaps else None,
        "median_gap_to_bks":                 statistics.median(gaps) if gaps else None,
        "missing_op_count_sum":              sum(r["missing_op_count"] for r in results),
        "routing_order_violations_sum":      sum(r["routing_order_violations"] for r in results),
        "machine_capacity_violations_sum":   sum(r["machine_capacity_violations"] for r in results),
        "timing_consistency_violations_sum": sum(r["timing_consistency_violations"] for r in results),
        "precedence_violations_sum":         sum(r["precedence_violations"] for r in results),
    }


def eval_sm(
    model,
    tokenizer,
    num_samples: int = 20,
    max_new_tokens: int = MAX_NEW_TOKENS,
    output_json: Path | None = None,
) -> dict:
    """Evaluate on the held-out 2% SM test split (in-distribution).

    Args:
        model          : loaded model (inference mode)
        tokenizer      : matching tokenizer
        num_samples    : number of test instances to evaluate
        max_new_tokens : maximum generation length
        output_json    : optional path to save results as JSON

    Returns:
        dict with "summary" and "results" keys
    """
    FastLanguageModel.for_inference(model)
    records = load_starjob_sm(split="test", limit=num_samples)
    print(f"[eval] SM test split: {len(records)} instances")

    results = []
    for i, r in enumerate(records, 1):
        gen, dt = generate_one(model, tokenizer, r["prompt"], max_new_tokens)
        v = check_violations(gen, r["jobs_spec"])
        v.update({"idx": i, "bks": r["bks"], "n_ops": r["n_ops"], "gen_time_s": round(dt, 2)})
        results.append(v)

        flag = "FEAS" if v["feasible"] else f"viol={v['total_violations']}"
        print(f"  [{i}/{len(records)}] {flag} cmax={v['makespan']} ({dt:.1f}s)")

    out = {"summary": _aggregate(results), "results": results}
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(out, indent=2))
        print(f"[eval] Saved → {output_json}")
    return out


def eval_ood(
    model,
    tokenizer,
    dataset: str = "all",
    max_new_tokens: int = MAX_NEW_TOKENS,
    output_json: Path | None = None,
) -> dict:
    """Evaluate on OOD benchmarks: Fisher-Thompson and/or Lawrence instances.

    Args:
        model          : loaded model (inference mode)
        tokenizer      : matching tokenizer
        dataset        : "ft" (3 instances) | "la" (15) | "all" (18)
        max_new_tokens : maximum generation length
        output_json    : optional path to save results as JSON

    Returns:
        dict with "summary" and "results" keys
    """
    FastLanguageModel.for_inference(model)

    if dataset == "ft":
        names = ["ft06", "ft10", "ft20"]
    elif dataset == "la":
        names = [f"la{i:02d}" for i in list(range(1, 11)) + list(range(16, 21))]
    else:
        names = None  # all 18 OOD_INSTANCES

    records = load_ood_benchmarks(names=names)
    print(f"[eval] OOD ({dataset}): {len(records)} instances")

    results = []
    for r in records:
        gen, dt = generate_one(model, tokenizer, r["prompt"], max_new_tokens)
        v = check_violations(gen, r["jobs_spec"])
        v.update({
            "name": r["name"], "bks": r["bks"],
            "n_ops": r["n_ops"], "gen_time_s": round(dt, 2),
        })
        results.append(v)

        flag  = "FEAS" if v["feasible"] else f"viol={v['total_violations']}"
        gap_s = ""
        if v["feasible"] and v["makespan"] and r["bks"]:
            gap_s = f" gap={((v['makespan'] - r['bks']) / r['bks']) * 100:.1f}%"
        print(f"  [{r['name']}] {flag} cmax={v['makespan']} bks={r['bks']}{gap_s} ({dt:.1f}s)")

    out = {"summary": _aggregate(results), "results": results}
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(out, indent=2))
        print(f"[eval] Saved → {output_json}")
    return out


def print_summary(result: dict) -> None:
    """Pretty-print the evaluation summary dict."""
    print("\n=== Summary ===")
    for k, v in result["summary"].items():
        if isinstance(v, float):
            print(f"  {k:>40}: {v:.4f}")
        else:
            print(f"  {k:>40}: {v}")
