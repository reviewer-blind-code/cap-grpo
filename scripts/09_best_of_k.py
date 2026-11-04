"""Step 9 — best of K at equal inference cost.

Draw k of the 64 stored samples, keep the valid schedule with the smallest
makespan, and report two things: how often an instance is solved at all, and
how far the kept schedule is from the reference. Models are compared within one
budget, which is what makes the inference cost equal.

The supervised budget that matches one RL sample is searched over every k from
1 to 64, not only the budgets printed.

Run:
    python scripts/09_best_of_k.py --data /path/to/raw --trials 2000
"""
import argparse
import csv
import random
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.rawdata import data_root, generations, label         # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
BASE = "SFT-LoRA"


def solved_at(per, model, group, k):
    """Unbiased probability that k draws contain at least one valid schedule."""
    cells = [v for (m, g, _), v in per.items() if m == model and g == group]
    if not cells:
        return None
    tot = 0.0
    for v in cells:
        n, c = len(v), sum(1 for x in v if x is not None)
        tot += 0.0 if c == 0 else 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
    return tot / len(cells) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    per = defaultdict(list)                    # (model, group, instance) -> [gap or None]
    for g in generations(root, "passk"):
        per[(g["model"], g["group"], g["instance"])].append(
            g["gap_pct"] if g["feasible"] else None)

    models, groups = [], []
    for (m, grp, _) in per:
        if m not in models:
            models.append(m)
        if grp not in groups:
            groups.append(grp)

    rng = random.Random(42)
    rows = []
    for m in models:
        for grp in groups:
            cells = [v for (mm, gg, _), v in per.items() if mm == m and gg == grp]
            for k in KS:
                kept = []
                for v in cells:
                    for _ in range(args.trials // 40 if k < 64 else 1):
                        draw = rng.sample(v, k) if k < len(v) else v
                        good = [x for x in draw if x is not None]
                        if good:
                            kept.append(min(good))
                rows.append({"model": m, "model_name": label(m), "group": grp, "k": k,
                             "solved_pct": round(solved_at(per, m, grp, k), 1),
                             "gap_of_kept": round(st.mean(kept), 2) if kept else None})

    for grp in groups:
        print(f"\n{grp}: solved % / gap of the kept schedule")
        print(f"{'model':46s}" + "".join(f"{'k=' + str(k):>15s}" for k in KS))
        for m in models:
            sel = {r["k"]: r for r in rows if r["model"] == m and r["group"] == grp}
            print(f"{label(m)[:44]:46s}" + "".join(
                f"{sel[k]['solved_pct']:8.1f}/{str(sel[k]['gap_of_kept']):>6s}" for k in KS))

    print("\nsupervised budget that matches one RL sample")
    for grp in groups:
        for m in models:
            if m.startswith(BASE) or solved_at(per, m, grp, 1) == 0:
                continue
            target = solved_at(per, m, grp, 1)
            need = next((k for k in range(1, 65)
                         if (solved_at(per, BASE, grp, k) or 0) >= target), None)
            if need:
                print(f"  {grp:16s} {label(m)[:40]:42s} k=1 gives {target:5.1f}% -> "
                      f"supervised needs k={need} ({solved_at(per, BASE, grp, need):.1f}%)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
