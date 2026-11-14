"""pass@k sampling with vLLM: k samples per instance per model.

Samples k completions per prompt at a fixed temperature, scores each with the
independent checker, and writes one JSON line per sample. pass@k for any k' <= k
is then computed offline from the same samples.

Base model and adapters match the evaluation protocol: the 4-bit base plus the
LoRA adapter of each fine-tuned model, so numbers stay comparable with the rest
of the project.

Run on a GPU node, from the repo root:
    venv-vllm/bin/python -u evaluation/run_passk.py --models SFT-LoRA V4 \
        --k 64 --temperature 0.7 --out passk/run1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "evaluation")]
from checker_v2 import check_all, parse_schedule  # noqa: E402

ALPACA = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
ADAPTERS = {
    "Base": None,
    "SFT-LoRA": "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800",
    "V3-stratified": "grpo_jssp/runs/full_lora_stratified_n2000_v3/final_adapter",
    "V4-hybrid": "grpo_jssp/runs/full_lora_hybrid_n2000_v4/final_adapter",
    "V7-hybridover": "grpo_jssp/runs/full_lora_hybrid_lc_over_n2000_v7/final_adapter",
    "VbinMk-feasmk": "grpo_jssp/runs/full_lora_binmk_n2000_vA/final_adapter",
    "Vbin-feasonly": "grpo_jssp/runs/full_lora_uniform_n2000_vbin/final_adapter",
    "StratNoMk-feaspen": "grpo_jssp/runs/full_lora_stratnomk_n2000_vB/final_adapter",
    "V1-stratified": "grpo_jssp/runs/full_lora_stratified_2000_v1/final_adapter",
    "V6LoRA-lenmask": "grpo_jssp/runs/full_lora_stratified_lc_n2000_v6/final_adapter",
}
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 8192


def prompt_for(rec):
    n, m = rec["n"], rec["m"]
    instruction = (
        f"Optimize schedule for {n} Jobs (denoted as J) across {m} Machines "
        "(denoted as M) to minimize makespan. The makespan is the completion "
        "time of the last operation in the schedule. Each M can process only "
        "one J at a time, and once started, J cannot be interrupted.\n\n"
    )
    parts = []
    for j, ops in enumerate(rec["jobs"]):
        parts.append(f"J{j}:")
        parts.append(" ".join(f"M{int(mc)}:{int(d)}" for mc, d in ops) + " ")
    return ALPACA.format(instruction, "\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ADAPTERS), choices=list(ADAPTERS))
    ap.add_argument("--instances", default=str(ROOT / "data/benchmarks/passk/instances.jsonl"))
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N instances (canary)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--store-text", action="store_true",
                    help="also write the generated text of every sample; the first run stored "
                         "only the verdicts, which is the one thing that cannot be re-checked")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    recs = [json.loads(l) for l in Path(args.instances).read_text().splitlines()]
    if args.limit:
        recs = recs[: args.limit]
    prompts = [prompt_for(r) for r in recs]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "samples.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            d = json.loads(line)
            done.add((d["model"], d["instance"], d["sample"]))

    (out_dir / "config.json").write_text(json.dumps({
        "models": {m: ADAPTERS[m] for m in args.models}, "k": args.k,
        "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
        "instances": len(recs), "base_model": BASE_MODEL,
        "store_text": args.store_text,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len}, indent=2))

    needs_lora = any(ADAPTERS[m] for m in args.models)
    llm = LLM(model=BASE_MODEL, quantization="bitsandbytes", load_format="bitsandbytes",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enable_lora=needs_lora, max_lora_rank=32, max_loras=1,
              enforce_eager=False, seed=args.seed)

    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, seed=args.seed)

    t_start = time.time()
    total = len(args.models) * len(recs)
    step = 0
    for li, model in enumerate(args.models, 1):
        adapter = ADAPTERS[model]
        lora = LoRARequest(model, li, str(ROOT / adapter)) if adapter else None
        todo = [i for i, r in enumerate(recs)
                if (model, r["name"], 0) not in done]
        print(f"\n=== {model} ({adapter or 'no adapter'}): {len(todo)} instances x {args.k} samples ===",
              flush=True)
        for i in todo:
            rec, prompt = recs[i], prompts[i]
            jobs = [[(int(a), int(b)) for a, b in ops] for ops in rec["jobs"]]
            t0 = time.time()
            outs = llm.generate([prompt], sp, lora_request=lora)[0].outputs
            dt = time.time() - t0
            feas = 0
            with open(res_path, "a") as f:
                for s, o in enumerate(outs):
                    c = check_all(parse_schedule(o.text), jobs)
                    gap = (round((c["makespan"] - rec["bks"]) / rec["bks"] * 100, 4)
                           if c["feasible"] and c["makespan"] and rec["bks"] else None)
                    feas += c["feasible"]
                    f.write(json.dumps({
                        "model": model, "instance": rec["name"], "group": rec["group"],
                        "set": rec["set"], "shape": f"{rec['n']}x{rec['m']}", "ops": rec["ops"],
                        "sample": s, "reference": rec["bks"], "reference_source": rec["bks_status"],
                        "feasible": c["feasible"], "makespan": c["makespan"], "gap_pct": gap,
                        "gen_tokens": len(o.token_ids), "violations": c["violations"],
                        **({"generation": o.text} if args.store_text else {}),
                    }) + "\n")
            step += 1
            with open("metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "total_steps": total,
                                    "elapsed_s": round(time.time() - t_start, 1),
                                    "model": model, "instance": rec["name"],
                                    "feasible_of_k": feas}) + "\n")
            print(f"  [{step}/{total}] {model} {rec['name']:12s} {rec['n']}x{rec['m']:<3} "
                  f"feasible {feas}/{args.k}  {dt:.1f}s", flush=True)

    print(f"\nselesai dalam {(time.time() - t_start) / 60:.1f} menit -> {res_path}")


if __name__ == "__main__":
    main()
