"""Step 7 — pass@k by region and by quality.

For each model, instance group and budget k, the unbiased estimator

    pass@k = 1 - C(n - c, k) / C(n, k)

over the n = 64 stored samples, averaged across the instances of the group.
Three outcomes are reported: a valid schedule, a valid schedule within ten per
cent of the reference, and neither.

Run:
    python scripts/07_passk.py --data /path/to/raw
    python scripts/07_passk.py --data /path/to/raw --csv passk_curves.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.rawdata import data_root, generations, label  # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
GROUPS = [("known_shape", "Known shapes"), ("new_shape", "New shapes"),
          ("long_duration", "Long durations")]


def pass_at_k(n, c, k):
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    per = defaultdict(lambda: {"n": 0, "valid": 0, "within10": 0})
    for g in generations(root, "passk"):
        # the pass@k run stored the verdict of each sample, not its text:
        # 57,600 schedules of raw text would be about half a gigabyte. The
        # verdicts were produced by the same checker as everything else.
        gap = g.get("gap_pct")
        d = per[(g["model"], g["group"], g["instance"])]
        d["n"] += 1
        d["valid"] += bool(g["feasible"])
        d["within10"] += bool(g["feasible"] and gap is not None and gap <= 10)

    models, rows = [], []
    for (m, _, _) in per:
        if m not in models:
            models.append(m)
    for m in models:
        for key, gname in GROUPS:
            cells = [v for (mm, gg, _), v in per.items() if mm == m and gg == key]
            if not cells:
                continue
            for k in KS:
                feas = sum(pass_at_k(c["n"], c["valid"], k) for c in cells) / len(cells) * 100
                good = sum(pass_at_k(c["n"], c["within10"], k) for c in cells) / len(cells) * 100
                rows.append({"model": m, "model_name": label(m), "group": gname,
                             "instances": len(cells), "k": k,
                             "feasible_pct": round(feas, 1),
                             "feasible_le10_pct": round(good, 1),
                             "infeasible_pct": round(100 - feas, 1)})

    for gname in [g for _, g in GROUPS]:
        print(f"\n{gname}: valid % / within 10 % of the reference")
        print(f"{'model':46s}" + "".join(f"{'k=' + str(k):>14s}" for k in KS))
        for m in models:
            sel = {r["k"]: r for r in rows if r["model"] == m and r["group"] == gname}
            if not sel:
                continue
            print(f"{label(m)[:44]:46s}" + "".join(
                f"{sel[k]['feasible_pct']:7.1f}/{sel[k]['feasible_le10_pct']:<6.1f}" for k in KS))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
