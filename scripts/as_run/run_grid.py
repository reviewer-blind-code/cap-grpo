"""Uniform evaluation grid: every model, every instance, the same seeds, one engine.

The earlier results were collected in stages, so coverage was uneven (445 or 455
instances, three or five seeds) and produced with two different inference stacks.
This script regenerates the whole grid with vLLM alone, at the evaluation
protocol used throughout the project (temperature 0.1), so every cell of the
comparison tables rests on the same procedure.

One generation per (model, instance, seed). Results are appended to
<out>/results.jsonl and already-present cells are skipped, so the run resumes.

Run on the GPU node, from the repo root:
    venv-vllm/bin/python -u evaluation/run_grid.py --seeds 0 1 2 3 4 --out grid/run1
"""
import argparse
import json
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
    "Base-LLaMA": None,
    "SFT-LoRA":   "output_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800",
    "SFT-rsLoRA": "output_llama8b_rslora_alpha32_r32_seq8192_b1_ga8_ep1/checkpoint-9800",
    "V1":         "grpo_jssp/runs/full_lora_stratified_2000_v1/final_adapter",
    "V3":         "grpo_jssp/runs/full_lora_stratified_n2000_v3/final_adapter",
    "V4":         "grpo_jssp/runs/full_lora_hybrid_n2000_v4/final_adapter",
    "V7":         "grpo_jssp/runs/full_lora_hybrid_lc_over_n2000_v7/final_adapter",
    "V6LoRA":     "grpo_jssp/runs/full_lora_stratified_lc_n2000_v6/final_adapter",
    "Vbin":       "grpo_jssp/runs/full_lora_uniform_n2000_vbin/final_adapter",
    "VbinMk":     "grpo_jssp/runs/full_lora_binmk_n2000_vA/final_adapter",
    "StratNoMk":  "grpo_jssp/runs/full_lora_stratnomk_n2000_vB/final_adapter",
    "RANDs42":    "grpo_jssp/runs/full_lora_random_n2000_s42/final_adapter",
    "RANDs43":    "grpo_jssp/runs/full_lora_random_n2000_s43/final_adapter",
    # stage 2-1, rerun from the LoRA checkpoint the eight designs start from
    "RANDL42":      "grpo_jssp/runs/full_lora_random_lora_n2000_s42/final_adapter",
    "RANDL43":      "grpo_jssp/runs/full_lora_random_lora_n2000_s43/final_adapter",
    # stage 2-2: supervised training at the budget of one RL run
    "SFTbudget5e6": "grpo_jssp/runs/sft_budget_lr5e6/final_adapter",
    "SFTbudget2e4": "grpo_jssp/runs/sft_budget_lr2e4/final_adapter",
    # stage 2-3: fine-tuned on its own schedules that the checker accepted
    "RFT":          "grpo_jssp/runs/rft_budget/final_adapter",
}
INSTANCE_FILES = ["data/benchmarks/extended/instances.jsonl",
                  "data/benchmarks/extended/instances_ta.jsonl"]
MAX_NEW_TOKENS = 4096
MAX_MODEL_LEN = 8192
TEMPERATURE = 0.1          # the evaluation protocol used throughout the project


CHAT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}{}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def prompt_for(rec, style="alpaca"):
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
    body = "\n".join(parts) + "\n"
    return (ALPACA if style == "alpaca" else CHAT).format(instruction, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ADAPTERS), choices=list(ADAPTERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None, help="first N instances (canary)")
    ap.add_argument("--sets", nargs="+", default=None, help="only these benchmark families")
    ap.add_argument("--prompt-style", choices=["alpaca", "chat"], default="alpaca",
                    help="alpaca: the template the models were fine-tuned on; "
                         "chat: the Llama 3.1 instruct template, to test prompt sensitivity")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    recs = []
    for f in INSTANCE_FILES:
        p = ROOT / f
        if p.exists():
            recs += [json.loads(l) for l in p.read_text().splitlines()]
    if args.sets:
        recs = [r for r in recs if r["set"] in args.sets]
    if args.limit:
        recs = recs[: args.limit]
    prompts = [prompt_for(r, args.prompt_style) for r in recs]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            d = json.loads(line)
            done.add((d["model"], d["seed"], d["name"]))

    (out_dir / "config.json").write_text(json.dumps({
        "models": {m: ADAPTERS[m] for m in args.models}, "seeds": args.seeds,
        "instances": len(recs), "temperature": args.temperature,
        "max_new_tokens": MAX_NEW_TOKENS, "base_model": BASE_MODEL,
        "prompt_style": args.prompt_style,
        "engine": "vllm", "note": "one generation per model x instance x seed"}, indent=2))

    needs_lora = any(ADAPTERS[m] for m in args.models)
    llm = LLM(model=BASE_MODEL, quantization="bitsandbytes", load_format="bitsandbytes",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=args.gpu_mem,
              enable_lora=needs_lora, max_lora_rank=32, max_loras=1, seed=0)

    total = len(args.models) * len(args.seeds) * len(recs)
    step = sum(1 for m in args.models for s in args.seeds for r in recs
               if (m, s, r["name"]) in done)
    t0 = time.time()
    print(f"[grid] models={len(args.models)} seeds={args.seeds} instances={len(recs)} "
          f"total={total} already_done={step}", flush=True)

    for li, model in enumerate(args.models, 1):
        adapter = ADAPTERS[model]
        lora = LoRARequest(model, li, str(ROOT / adapter)) if adapter else None
        for seed in args.seeds:
            todo = [(i, r) for i, r in enumerate(recs) if (model, seed, r["name"]) not in done]
            if not todo:
                continue
            sps = [SamplingParams(n=1, temperature=args.temperature, top_p=1.0,
                                  max_tokens=MAX_NEW_TOKENS, seed=1000 * seed + i)
                   for i, _ in todo]
            outs = llm.generate([prompts[i] for i, _ in todo], sps, lora_request=lora)
            feas = 0
            with open(res_path, "a") as f:
                for (i, rec), o in zip(todo, outs):
                    text = o.outputs[0].text
                    jobs = [[(int(a), int(b)) for a, b in ops] for ops in rec["jobs"]]
                    c = check_all(parse_schedule(text), jobs)
                    gap = (round((c["makespan"] - rec["bks"]) / rec["bks"] * 100, 4)
                           if c["feasible"] and c["makespan"] and rec["bks"] else None)
                    feas += c["feasible"]
                    f.write(json.dumps({
                        "model": model, "seed": seed, "set": rec["set"], "name": rec["name"],
                        "size": f"{rec['n']}x{rec['m']}", "ops": rec["ops"], "bks": rec["bks"],
                        "bks_status": rec["bks_status"], "feasible": c["feasible"],
                        "makespan": c["makespan"], "gap_pct": gap,
                        "gen_tokens": len(o.outputs[0].token_ids),
                        **c["violations"], "generation": text}) + "\n")
            step += len(todo)
            with open("metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "total_steps": total,
                                    "elapsed_s": round(time.time() - t0, 1),
                                    "model": model, "seed": seed,
                                    "feasible": feas, "of": len(todo)}) + "\n")
            print(f"  [{step}/{total}] {model} seed {seed}: {feas}/{len(todo)} feasible "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {res_path}")


if __name__ == "__main__":
    main()
