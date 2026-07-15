"""Step 4 — OOD Benchmark Evaluation.

Evaluates a trained adapter on Fisher-Thompson (FT) and Lawrence (LA) instances
from OR-Library — 18 instances total, never used during training.

Run:
    python scripts/04_eval_ood.py --model-path outputs/cap_grpo_v1/final_adapter

    # Evaluate only FT instances:
    python scripts/04_eval_ood.py --model-path outputs/cap_grpo_v1/final_adapter --dataset ft

    # Save results to JSON:
    python scripts/04_eval_ood.py --model-path outputs/cap_grpo_v1/final_adapter \\
        --out results/my_run_ood.json

    # In-distribution sanity check (SM test split):
    python scripts/04_eval_ood.py --model-path outputs/cap_grpo_v1/final_adapter \\
        --dataset sm --n-sm-samples 20

    # Single-instance debug:
    python scripts/04_eval_ood.py --model-path outputs/cap_grpo_v1/final_adapter \\
        --infer ft06
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from cap_grpo.config import MAX_NEW_TOKENS, MODEL_REGISTRY
from cap_grpo.dataset import load_ood_benchmarks
from cap_grpo.checker import check_violations
from cap_grpo.evaluator import eval_ood, eval_sm, generate_one, print_summary
from cap_grpo.model import load_model


def cmd_infer(model, tokenizer, instance: str, max_new_tokens: int) -> None:
    """Run inference on a single named OOD instance and print the verdict."""
    all_records = {r["name"]: r for r in load_ood_benchmarks()}
    if instance not in all_records:
        available = list(all_records)
        print(f"Instance '{instance}' not found. Available: {available}", file=sys.stderr)
        sys.exit(2)

    r            = all_records[instance]
    response, dt = generate_one(model, tokenizer, r["prompt"], max_new_tokens)
    v            = check_violations(response, r["jobs_spec"])
    bks          = r.get("bks")
    gap          = (v["makespan"] - bks) / bks if (v["feasible"] and v["makespan"] and bks) else None

    print("\n----- RESPONSE (first 2000 chars) -----")
    print(response[:2000])
    print("\n----- VERDICT -----")
    print(json.dumps({
        "instance":   instance,
        "bks":        bks,
        "feasible":   v["feasible"],
        "makespan":   v["makespan"],
        "gap_to_bks": round(gap, 4) if gap is not None else None,
        "gen_time_s": round(dt, 2),
        "violations": {k: v[k] for k in (
            "missing_op_count", "over_op_count", "routing_order_violations",
            "machine_capacity_violations", "timing_consistency_violations",
            "precedence_violations",
        )},
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CAP-GRPO adapter on OOD benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model",      choices=list(MODEL_REGISTRY),
                     help="Registered base model key")
    src.add_argument("--model-path", metavar="PATH",
                     help="Local path to adapter or base model")

    parser.add_argument("--dataset",       choices=["ft", "la", "all", "sm"], default="all",
                        help="Benchmark subset to evaluate")
    parser.add_argument("--n-sm-samples",  type=int, default=20,
                        help="Number of SM test instances (only used when --dataset sm)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--out",           default=None, metavar="PATH",
                        help="Save results as JSON to this path")
    parser.add_argument("--infer",         default=None, metavar="INSTANCE",
                        help="Run single-instance inference + feasibility check")
    parser.add_argument("--dtype",         choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model(
        model_key=args.model,
        model_path=args.model_path,
        dtype=dtype,
        for_training=False,
    )

    if args.infer:
        cmd_infer(model, tokenizer, args.infer, args.max_new_tokens)
        return

    out_path = Path(args.out) if args.out else None

    if args.dataset == "sm":
        result = eval_sm(model, tokenizer,
                         num_samples=args.n_sm_samples,
                         max_new_tokens=args.max_new_tokens,
                         output_json=out_path)
    else:
        result = eval_ood(model, tokenizer,
                          dataset=args.dataset,
                          max_new_tokens=args.max_new_tokens,
                          output_json=out_path)

    print_summary(result)


if __name__ == "__main__":
    main()
