"""Step 8 — which instances each model solves, and which it loses.

An instance counts as solved by a model when at least one of its samples is a
valid schedule. Every RL variant is then compared with SFT-LoRA instance by
instance: both, RL only, supervised only, neither.

Two budgets are reported: five decoding seeds on the 455-instance grid, and 64
samples on the 90 pass@k instances, where the three groups are listed
separately because known shapes are solved by everyone and long durations by
no one.

Run:
    python scripts/08_solved_lost.py --data /path/to/raw
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations                       # noqa: E402
from cap_grpo.rawdata import BASELINE, data_root, generations, instances, label  # noqa: E402

PASSK_BASE = "SFT-LoRA"


def classify(sft, rl):
    return "both" if (sft and rl) else "rl_only" if rl else "sft_only" if sft else "neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    # --- grid, five seeds
    inst = instances(root, "ood")
    solved = defaultdict(set)
    for g in generations(root, "grid"):
        if g["name"] not in inst:
            continue
        if check_violations(g["generation"], inst[g["name"]]["jobs"])["feasible"]:
            solved[g["model"]].add(g["name"])
    grid = []
    for m in sorted(solved):
        if m == BASELINE:
            continue
        both = len(solved[m] & solved[BASELINE])
        grid.append({"model": m, "name": label(m), "both": both,
                     "rl_only": len(solved[m] - solved[BASELINE]),
                     "sft_only": len(solved[BASELINE] - solved[m]),
                     "neither": len(inst) - len(solved[m] | solved[BASELINE])})
    grid.sort(key=lambda d: -d["rl_only"])

    # --- pass@k, 64 samples, by group
    group_of, solved_k = {}, defaultdict(set)
    for g in generations(root, "passk"):
        group_of[g["instance"]] = g["group"]
        if g["feasible"]:
            solved_k[g["model"]].add(g["instance"])
    groups = sorted(set(group_of.values()))
    base_key = next(k for k in solved_k if k.startswith(PASSK_BASE))
    passk = []
    for m in sorted(solved_k):
        if m == base_key:
            continue
        row = {"model": m, "name": label(m)}
        for grp in groups:
            names = {i for i, g in group_of.items() if g == grp}
            a, b = solved_k[base_key] & names, solved_k[m] & names
            row[grp] = {"n": len(names), "both": len(a & b), "rl_only": len(b - a),
                        "sft_only": len(a - b), "neither": len(names - (a | b))}
        passk.append(row)
    passk.sort(key=lambda d: -sum(d[g]["rl_only"] for g in groups))

    print(f"grid, 5 seeds, {len(inst)} instances, against {label(BASELINE)}")
    print(f"{'variant':46s}{'both':>7s}{'RL only':>9s}{'SFT only':>10s}{'neither':>9s}")
    for d in grid:
        print(f"{d['name'][:44]:46s}{d['both']:7d}{d['rl_only']:9d}{d['sft_only']:10d}{d['neither']:9d}")

    for grp in groups:
        print(f"\npass@k, 64 samples, {grp}")
        print(f"{'variant':46s}{'n':>5s}{'both':>7s}{'RL only':>9s}{'SFT only':>10s}{'neither':>9s}")
        for d in passk:
            c = d[grp]
            print(f"{d['name'][:44]:46s}{c['n']:5d}{c['both']:7d}{c['rl_only']:9d}"
                  f"{c['sft_only']:10d}{c['neither']:9d}")

    if args.json:
        Path(args.json).write_text(json.dumps({"grid": grid, "passk": passk}, indent=2))
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
