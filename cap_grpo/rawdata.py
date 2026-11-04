"""Loading the raw evaluation data that the analysis scripts read.

The generations are too large for git, so they live in a separate folder
(see the Raw data section of the README). Every script takes --data pointing
at that folder, or reads the JSSP_RAW environment variable.

Layout expected:

    <data>/instances/ood_455_main.jsonl      445 instances
    <data>/instances/ood_455_taillard.jsonl   10 instances
    <data>/instances/passk_90.jsonl           90 instances
    <data>/grid_uniform/results.jsonl         25,025 generations
    <data>/passk_k64/samples.jsonl            57,600 samples
    <data>/sm_heldout/…                       225 instances, 9,000 generations
    <data>/multiseed_1008/generations.jsonl   the October 8 evaluation
    <data>/probes/…                           scale, token budget, prompt
    <data>/training_logs/<run>/…              reward, KL and loss per step
"""
import json
import os
from pathlib import Path

NAMES = {
    "Base-LLaMA": "No fine-tuning", "Base": "No fine-tuning",
    "SFT-LoRA": "Supervised fine-tuning (LoRA)",
    "SFT-rsLoRA": "Supervised fine-tuning (rsLoRA)",
    "Vbin": "Feasibility-only reward (V1)", "VbinMk": "Feasibility plus makespan (V2)",
    "StratNoMk": "Feasibility plus penalty (V3)", "V1": "Stratified reward (V4)",
    "V3": "Stratified reward, matched hyperparameters (V5)",
    "V6LoRA": "Stratified reward with length mask (V6)",
    "V4": "Hybrid reward, CAP-GRPO (V7)", "V7": "Hybrid with over-emission penalty (V8)",
    "RANDs42": "Uninformative reward, seed 42", "RANDs43": "Uninformative reward, seed 43",
    # the pass@k run uses longer keys for the same adapters
    "Vbin-feasonly": "Feasibility-only reward (V1)",
    "VbinMk-feasmk": "Feasibility plus makespan (V2)",
    "StratNoMk-feaspen": "Feasibility plus penalty (V3)",
    "V1-stratified": "Stratified reward (V4)",
    "V3-stratified": "Stratified reward, matched hyperparameters (V5)",
    "V6LoRA-lenmask": "Stratified reward with length mask (V6)",
    "V4-hybrid": "Hybrid reward, CAP-GRPO (V7)",
    "V7-hybridover": "Hybrid with over-emission penalty (V8)",
}
BASELINE = "SFT-LoRA"


def data_root(arg=None) -> Path:
    root = arg or os.environ.get("JSSP_RAW")
    if not root:
        raise SystemExit("point --data at the raw-data folder, or set JSSP_RAW "
                         "(see the Raw data section of the README)")
    p = Path(root).expanduser()
    if not p.exists():
        raise SystemExit(f"{p} does not exist")
    return p


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def instances(root, which="ood"):
    """name -> instance record, with jobs as a list of (machine, duration)."""
    files = {"ood": ["instances/ood_455_main.jsonl", "instances/ood_455_taillard.jsonl"],
             "passk": ["instances/passk_90.jsonl"],
             "scale": ["instances/scale_90.jsonl"],
             "ta_probe": ["instances/ta_probe_20.jsonl"]}[which]
    out = {}
    for f in files:
        for r in read_jsonl(root / f):
            r["jobs"] = [[(int(a), int(b)) for a, b in ops] for ops in r["jobs"]]
            out[r["name"]] = r
    return out


def generations(root, which="grid"):
    f = {"grid": "grid_uniform/results.jsonl",
         "passk": "passk_k64/samples.jsonl",
         "sm": "sm_heldout/generations.jsonl",
         "oct8": "multiseed_1008/generations.jsonl"}[which]
    return read_jsonl(root / f)


def label(model):
    return NAMES.get(model, model)
