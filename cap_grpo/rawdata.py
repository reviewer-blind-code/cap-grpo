"""Loading the raw evaluation data that the analysis scripts read.

The generations are too large for git, so they live in a separate folder
(see the Raw data section of the README). Every script takes --data pointing
at that folder, or reads the JSSP_RAW environment variable.

Layout expected:

    <data>/01_instances/ood_main_445.jsonl       445 instances
    <data>/01_instances/ood_taillard_10.jsonl    10 instances
    <data>/01_instances/passk_90.jsonl            90 instances
    <data>/04_stage1_evaluation/main_grid/results.jsonl          25,025 generations
    <data>/04_stage1_evaluation/passk/as_reported_verdicts_only/samples.jsonl
    <data>/04_stage1_evaluation/passk/repeat_with_text/samples.jsonl
    <data>/04_stage1_evaluation/sm_heldout/generations.jsonl      9,000 generations
    <data>/04_stage1_evaluation/probes/…          scale, token budget, prompt
    <data>/05_stage2_tests/test_2_1_random_reward/generations.jsonl
    <data>/05_stage2_tests/test_2_2_more_supervised/generations.jsonl
    <data>/05_stage2_tests/test_2_3_rejection_sampling/generations.jsonl
    <data>/06_earlier_runs/multiseed_oct08/generations.jsonl      the October 8 evaluation

The training logs are small enough for git and live in `training_logs/` in this
repository, not in the raw-data folder.
"""
import json
import os
from pathlib import Path

NAMES = {
    "Base": "No fine-tuning",
    "SFT-LoRA": "Supervised fine-tuning (LoRA)",
    "SFT-rsLoRA": "Supervised fine-tuning (rsLoRA)",
    "V1": "Feasibility-only reward (V1)",
    "V2": "Feasibility plus makespan (V2)",
    "V3": "Feasibility plus penalty (V3)",
    "V4": "Stratified reward (V4)",
    "V5": "Stratified reward, matched settings (V5)",
    "V6": "Stratified reward with length mask (V6)",
    "V7": "Hybrid reward, CAP-GRPO (V7)",
    "V8": "Hybrid with over-emission penalty (V8)",
    "T2-1-seed42": "Random reward, seed 42 (test 2-1)",
    "T2-1-seed43": "Random reward, seed 43 (test 2-1)",
    "T2-2-lr5e-6": "More supervised training, lr 5e-6 (test 2-2)",
    "T2-2-lr2e-4": "More supervised training, lr 2e-4 (test 2-2)",
    "T2-3": "Rejection-sampling fine-tuning (test 2-3)",
}
# Every raw record also carries `model_as_run`, the key the file was produced
# under; docs/REPRODUCE.md maps the two.
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
    files = {"ood": ["01_instances/ood_main_445.jsonl",
                     "01_instances/ood_taillard_10.jsonl"],
             "passk": ["01_instances/passk_90.jsonl"],
             "scale": ["01_instances/scale_90.jsonl"],
             "ta_probe": ["01_instances/ta_probe_20.jsonl"]}[which]
    out = {}
    for f in files:
        for r in read_jsonl(root / f):
            r["jobs"] = [[(int(a), int(b)) for a, b in ops] for ops in r["jobs"]]
            out[r["name"]] = r
    return out


def generations(root, which="grid"):
    f = {"grid": "04_stage1_evaluation/main_grid/results.jsonl",
         "passk": "04_stage1_evaluation/passk/as_reported_verdicts_only/samples.jsonl",
         "passk_text": "04_stage1_evaluation/passk/repeat_with_text/samples.jsonl",
         "sm": "04_stage1_evaluation/sm_heldout/generations.jsonl",
         "oct8": "06_earlier_runs/multiseed_oct08/generations.jsonl",
         "test_2_1": "05_stage2_tests/test_2_1_random_reward/generations.jsonl",
         "test_2_2": "05_stage2_tests/test_2_2_more_supervised/generations.jsonl",
         "test_2_3": "05_stage2_tests/test_2_3_rejection_sampling/generations.jsonl"}[which]
    return read_jsonl(root / f)


def label(model):
    return NAMES.get(model, model)
