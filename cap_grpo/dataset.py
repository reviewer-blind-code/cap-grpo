"""Dataset utilities: download, sample, and load StarJob SM and OOD benchmarks.

Three-step workflow
-------------------
1. download_starjob()   -- fetch full StarJob dataset from HuggingFace
2. sample_sm()          -- filter to small-medium instances (jobs<=10, machines<=10)
3. load_starjob_sm()    -- read the sampled JSONL, return train/test split records

OOD benchmarks (FT + LA instances) are read from OR-Library's jobshop1.txt:
    download_ood()       -- fetch jobshop1.txt from OR-Library
    load_ood_benchmarks() -- parse and return eval records
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .checker import extract_makespan, parse_ops
from .config import (
    ALPACA_PROMPT, BEST_KNOWN, JOBSHOP1,
    OOD_INSTANCES, SM_TRAIN_FILE, SPLIT_SEED, TEST_FRAC,
)

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_PROMPT.format(instruction=instruction, input=input_text)


def instruction_for_instance(n_jobs: int, n_machines: int) -> str:
    return (
        f"Optimize schedule for {n_jobs} Jobs (denoted as J) across {n_machines} Machines "
        "(denoted as M) to minimize makespan. The makespan is the completion "
        "time of the last operation in the schedule. Each M can process only "
        "one J at a time, and once started, J cannot be interrupted.\n\n"
    )


def jobs_to_input_text(jobs: list) -> str:
    """Convert jobs_spec list → StarJob input text format."""
    lines = []
    for j, ops in enumerate(jobs):
        lines.append(f"J{j}:")
        lines.append(" ".join(f"M{mi}:{du}" for mi, du in ops) + " ")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# StarJob: download + sample
# ---------------------------------------------------------------------------

def download_starjob(
    output_path: Path | str | None = None,
    hf_dataset: str = "mideavalwisard/Starjob",
    hf_file: str = "starjob_130k_filled.json",
) -> Path:
    """Download the full StarJob dataset from HuggingFace Hub.

    The raw file is saved as-is to ``output_path``.
    Default output: ``data/starjob_raw.json``

    Args:
        output_path : where to save the raw download (JSON or JSONL)
        hf_dataset  : HuggingFace dataset repo id
        hf_file     : filename inside the dataset repo

    Returns:
        Path to the saved file
    """
    from datasets import load_dataset

    out = Path(output_path) if output_path else SM_TRAIN_FILE.parent / "starjob_raw.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] Loading {hf_dataset}/{hf_file} from HuggingFace ...")
    ds = load_dataset(hf_dataset, data_files=hf_file, split="train")
    print(f"[download] {len(ds):,} total records")

    records = [
        {"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
        for r in ds
    ]

    with open(out, "w") as f:
        json.dump(records, f)
    print(f"[download] Saved raw → {out}")
    return out


def sample_sm(
    raw_path: Path | str | None = None,
    output_path: Path | str | None = None,
    max_jobs: int = 10,
    max_machines: int = 10,
) -> Path:
    """Filter raw StarJob to small-medium (SM) instances and save as JSONL.

    SM instances are those with num_jobs <= max_jobs AND num_machines <= max_machines.
    The filter is applied by parsing the instruction field, e.g.:
        "Optimize schedule for 6 Jobs ... across 5 Machines ..."

    Args:
        raw_path     : path to the raw starjob JSON file (default: data/starjob_raw.json)
        output_path  : where to save the JSONL (default: data/starjob_train_sm.jsonl)
        max_jobs     : upper bound on number of jobs
        max_machines : upper bound on number of machines

    Returns:
        Path to the saved JSONL file
    """
    raw  = Path(raw_path)  if raw_path    else SM_TRAIN_FILE.parent / "starjob_raw.json"
    out  = Path(output_path) if output_path else SM_TRAIN_FILE

    print(f"[sample_sm] Reading {raw} ...")
    with open(raw) as f:
        records = json.load(f)

    _pat = re.compile(r'(\d+)\s+Jobs.*?(\d+)\s+Machines', re.IGNORECASE)
    kept = []
    for r in records:
        m = _pat.search(r.get("instruction", ""))
        if m and int(m.group(1)) <= max_jobs and int(m.group(2)) <= max_machines:
            kept.append(r)

    print(f"[sample_sm] {len(kept):,} / {len(records):,} records pass filter "
          f"(jobs≤{max_jobs}, machines≤{max_machines})")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")
    print(f"[sample_sm] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# StarJob: load + split
# ---------------------------------------------------------------------------

def _parse_input_jobs(input_text: str) -> list:
    """Parse ``J0: M2:8 M0:5 ...`` block → [[(machine, dur), ...], ...]."""
    jobs, current = [], []
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if re.match(r"^J\d+:", line):
            if current:
                jobs.append(current)
            rest    = re.sub(r"^J\d+:\s*", "", line)
            current = [(int(m), int(d)) for m, d in re.findall(r"M(\d+):(\d+)", rest)]
        else:
            current.extend((int(m), int(d)) for m, d in re.findall(r"M(\d+):(\d+)", line))
    if current:
        jobs.append(current)
    return jobs


def _bks_from_output(output_text: str):
    """Extract best-known makespan from a gold-standard output string."""
    m = extract_makespan(output_text)
    if m is not None:
        return m
    ops, _ = parse_ops(output_text)
    return max((e for *_, e in ops), default=None)


def _make_record(raw: dict) -> dict:
    jobs_spec = _parse_input_jobs(raw["input"])
    return {
        "instruction": raw["instruction"],
        "input":       raw["input"],
        "output":      raw["output"],
        "prompt":      format_prompt(raw["instruction"], raw["input"]),
        "jobs_spec":   jobs_spec,
        "n_ops":       sum(len(j) for j in jobs_spec),
        "bks":         _bks_from_output(raw["output"]),
    }


def load_starjob_sm(
    path: Path | str = SM_TRAIN_FILE,
    split: str = "train",
    limit: int | None = None,
) -> list:
    """Load StarJob SM JSONL with a deterministic 2% held-out test split.

    The same seed (42) and fraction (2%) must be used in both SFT and GRPO
    so the test set is truly unseen during all training stages.

    Args:
        path  : path to starjob_train_sm.jsonl
        split : "train" (98%) | "test" (2%) | "all"
        limit : optional cap on number of records returned

    Returns:
        list of record dicts with keys:
            instruction, input, output, prompt, jobs_spec, n_ops, bks
    """
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line))

    if split == "all":
        chosen = raw
    else:
        rng     = random.Random(SPLIT_SEED)
        indices = list(range(len(raw)))
        rng.shuffle(indices)
        test_size = int(len(raw) * TEST_FRAC)
        test_set  = set(indices[:test_size])

        if split == "test":
            chosen = [raw[i] for i in indices[:test_size]]
        elif split == "train":
            chosen = [raw[i] for i in range(len(raw)) if i not in test_set]
        else:
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")

    if limit is not None:
        chosen = chosen[:limit]

    return [_make_record(r) for r in chosen]


# ---------------------------------------------------------------------------
# OOD benchmarks: download + load
# ---------------------------------------------------------------------------

def download_ood(output_path: Path | str | None = None) -> Path:
    """Download OR-Library jobshop1.txt containing FT and LA benchmark instances.

    The file is fetched from the official OR-Library mirror at Brunel University.

    Args:
        output_path : save path (default: data/benchmarks/jobshop1.txt)

    Returns:
        Path to the saved file
    """
    import urllib.request

    url = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt"
    out = Path(output_path) if output_path else JOBSHOP1
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] Fetching OR-Library jobshop1.txt ...")
    try:
        urllib.request.urlretrieve(url, out)
        print(f"[download] Saved → {out}")
    except Exception as e:
        print(f"[download] Primary mirror failed: {e}")
        print("[download] Try downloading manually and place at:", out)
        raise

    return out


def _parse_orlib_block(lines: list):
    n, m = map(int, lines[0].split())
    jobs = []
    for j in range(1, n + 1):
        toks = list(map(int, lines[j].split()))
        jobs.append([(toks[2 * k], toks[2 * k + 1]) for k in range(len(toks) // 2)])
    return n, m, jobs


def load_ood_benchmarks(
    path: Path | str = JOBSHOP1,
    names: list | None = None,
) -> list:
    """Parse jobshop1.txt and return eval records for the specified OOD instances.

    Args:
        path  : path to jobshop1.txt (download with download_ood() if missing)
        names : list of instance names to load (default: all 18 FT+LA instances)

    Returns:
        list of record dicts with keys:
            name, n, m, instruction, input, prompt, jobs_spec, n_ops, bks
    """
    names = names or OOD_INSTANCES
    with open(path) as f:
        text = f.read()

    blocks = re.split(r"\n\s*instance\s+(\w+)\s*\n", text)
    parsed = {}
    for i in range(1, len(blocks), 2):
        name       = blocks[i].strip()
        body_lines = [
            ln for ln in blocks[i + 1].splitlines()
            if ln.strip()
            and not ln.lstrip().startswith("+")
            and not re.match(r"^\s*[A-Za-z]", ln)
        ]
        try:
            n, m, jobs = _parse_orlib_block(body_lines)
            if len(jobs) == n and all(len(j) == m for j in jobs):
                parsed[name] = (n, m, jobs)
        except Exception:
            continue

    records = []
    for name in names:
        if name not in parsed:
            continue
        n, m, jobs = parsed[name]
        instr      = instruction_for_instance(n, m)
        inp        = jobs_to_input_text(jobs)
        records.append({
            "name":        name,
            "n":           n,
            "m":           m,
            "instruction": instr,
            "input":       inp,
            "prompt":      format_prompt(instr, inp),
            "jobs_spec":   jobs,
            "n_ops":       sum(len(j) for j in jobs),
            "bks":         BEST_KNOWN.get(name),
        })
    return records
