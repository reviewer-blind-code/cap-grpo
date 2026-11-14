"""Stage 2-3, step 1: sample from the supervised model and keep what the checker accepts.

The RL run draws K = 4 completions for each of 2,000 training prompts, 8,000 in
all, and turns the checker verdict into a gradient. This script draws the same
8,000 completions from the same checkpoint at the same temperature, keeps the
schedules the checker accepts, and writes them as training data. Fine-tuning on
that file (grpo_jssp/sft_budget.py --data) is rejection-sampling fine-tuning: the
same information, used without the policy-gradient step.

The filter is checker v2, the one used for every number in the audit, not the v1
checker the reward called during training.

Run on the GPU node, from the repo root:
    venv-vllm/bin/python -u -m grpo_jssp.sample_rft \
        --adapter output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800 \
        --out results/rft_samples
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]
from checker_v2 import check_all, parse_schedule  # noqa: E402
from grpo_jssp.config import STARJOB_SM_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P  # noqa: E402
from grpo_jssp.data_utils import load_starjob_sm  # noqa: E402

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_MODEL_LEN = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800")
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/rft_samples")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    recs = load_starjob_sm(STARJOB_SM_PATH, limit=args.max_records, split="train")
    print(f"[rft] prompts: {len(recs)}  k={args.k}  T={args.temperature}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=BASE_MODEL, quantization="bitsandbytes", load_format="bitsandbytes",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=args.gpu_mem,
              enable_lora=True, max_lora_rank=32, max_loras=1, seed=args.seed)
    lora = LoRARequest("sft", 1, str(ROOT / args.adapter))

    sps = [SamplingParams(n=args.k, temperature=args.temperature, top_p=TOP_P,
                          max_tokens=MAX_NEW_TOKENS, seed=args.seed * 100000 + i)
           for i in range(len(recs))]
    t0 = time.time()
    outs = llm.generate([r["prompt"] for r in recs], sps, lora_request=lora)
    print(f"[rft] generated in {(time.time() - t0) / 60:.1f} min", flush=True)

    n_feas = n_prompt_hit = 0
    with open(out_dir / "samples.jsonl", "w") as fs, open(out_dir / "kept.jsonl", "w") as fk:
        for rec, o in zip(recs, outs):
            jobs = [[(int(m), int(d)) for m, d in job] for job in rec["jobs_spec"]]
            best = None
            for cand in o.outputs:
                c = check_all(parse_schedule(cand.text), jobs)
                n_feas += c["feasible"]
                fs.write(json.dumps({
                    "instruction": rec["instruction"][:80], "n_ops": rec["n_ops"],
                    "feasible": c["feasible"], "makespan": c["makespan"],
                    "gold_makespan": rec["bks"], "gen_tokens": len(cand.token_ids),
                    **c["violations"]}) + "\n")
                if c["feasible"] and c["makespan"] and (best is None or c["makespan"] < best[0]):
                    best = (c["makespan"], cand.text)
            if best:
                n_prompt_hit += 1
                fk.write(json.dumps({
                    "instruction": rec["instruction"], "input": rec["input"],
                    "output": best[1], "makespan": best[0],
                    "gold_makespan": rec["bks"]}) + "\n")

    total = len(recs) * args.k
    summary = {"prompts": len(recs), "k": args.k, "completions": total,
               "feasible_completions": n_feas,
               "feasible_pct": round(n_feas / total * 100, 2),
               "prompts_with_a_valid_schedule": n_prompt_hit,
               "prompt_coverage_pct": round(n_prompt_hit / len(recs) * 100, 2),
               "adapter": args.adapter, "temperature": args.temperature,
               "checker": "v2", "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
