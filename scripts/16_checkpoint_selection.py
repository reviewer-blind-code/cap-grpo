"""Does the checkpoint the project started from matter?

Every model in this repository starts from one supervised checkpoint, step
9,800, selected because its evaluation loss was the lowest available. The epoch
it belongs to ran to completion at step 14,406, and over those remaining 4,606
steps the loss changes by 0.0022 -- so by the loss alone, step 9,800 looks like
a settled choice.

This script compares the two checkpoints on the uniform grid, cell by cell:
validity with a two-sided exact sign test, and the mean paired gap difference
over the cells both of them solve. It is the same comparison, computed the same
way, as the reward designs in `scripts/14_rl_effect.py`.

Usage:
    export JSSP_RAW=/path/to/JSSP-SLM-raw
    python scripts/16_checkpoint_selection.py
"""
import argparse
import json
import os
import statistics as st
from math import comb
from pathlib import Path

GRID = "04_stage1_evaluation/main_grid/results.jsonl"
FULL = "07_checkpoint_selection/results.jsonl"


def load(path, model):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if f'"{model}"' not in line:
            continue
        r = json.loads(line)
        if r["model"] == model:
            rows[(r["name"], r["seed"])] = r
    return rows


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.environ.get("JSSP_RAW"),
                    help="the raw data folder; defaults to $JSSP_RAW")
    args = ap.parse_args()
    if not args.raw:
        raise SystemExit("set JSSP_RAW or pass --raw")
    raw = Path(args.raw)

    a = load(raw / GRID, "SFT-LoRA")
    b = load(raw / FULL, "SFT-LoRA-full")
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no paired cells; check --raw")

    va = 100 * sum(a[k]["feasible"] for k in keys) / len(keys)
    vb = 100 * sum(b[k]["feasible"] for k in keys) / len(keys)
    wins = sum(1 for k in keys if b[k]["feasible"] and not a[k]["feasible"])
    losses = sum(1 for k in keys if a[k]["feasible"] and not b[k]["feasible"])

    pairs = [(a[k]["gap_pct"], b[k]["gap_pct"]) for k in keys
             if a[k].get("gap_pct") is not None and b[k].get("gap_pct") is not None]
    diffs = [y - x for x, y in pairs]
    better = sum(1 for d in diffs if d < 0)
    worse = sum(1 for d in diffs if d > 0)

    print(f"paired cells                 {len(keys)}")
    print(f"valid, step 9,800            {va:.2f} %")
    print(f"valid, step 14,406           {vb:.2f} %  ({vb - va:+.2f} pp)")
    print(f"  wins / losses              {wins} / {losses}   p = {sign_test(wins, losses):.3g}")
    print(f"gap, cells both solve        {len(pairs)}")
    print(f"  mean paired difference     {st.mean(diffs):+.2f} pp")
    print(f"  better / worse             {better} / {worse}   p = {sign_test(better, worse):.3g}")
    for name, rows in (("9,800", a), ("14,406", b)):
        cut = sum(1 for k in keys if rows[k].get("gen_tokens", 0) >= 4096)
        print(f"cut off at the token limit, step {name:6s} {100 * cut / len(keys):.1f} %")


if __name__ == "__main__":
    main()
