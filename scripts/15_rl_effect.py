"""Step 15 — the paired effect of the RL stage over its supervised start.

Every RL run is compared with SFT-LoRA on the same instance and the same seed.
Pairs where exactly one model produced a valid schedule are the decisive ones;
they are tested with a two-sided exact sign test, equivalently the exact
McNemar test on the two discordant cells:

    n = wins + losses,  k = min(wins, losses)
    p = 2 * sum_{i=0..k} C(n, i) * (1/2)^n      (capped at 1)

Pairs where both models are valid carry the quality comparison: the change in
gap, reported as mean, median and deciles, because the mean alone is carried by
a minority of large changes.

Run:
    python scripts/15_rl_effect.py --data /path/to/raw
    python scripts/15_rl_effect.py --data /path/to/raw --json out.json
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations                      # noqa: E402
from cap_grpo.rawdata import BASELINE, data_root, generations, instances, label  # noqa: E402


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)


def score(gen, inst):
    r = check_violations(gen["generation"], inst["jobs"])
    gap = None
    if r["feasible"] and r["makespan"] and inst["bks"]:
        gap = (r["makespan"] - inst["bks"]) / inst["bks"] * 100
    return r["feasible"], gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    inst = instances(root, "ood")
    scored = {}
    for g in generations(root, "grid"):
        if g["name"] not in inst:
            continue
        scored[(g["model"], g["seed"], g["name"])] = score(g, inst[g["name"]])

    models = sorted({m for m, _, _ in scored} - {BASELINE})
    out = []
    for m in models:
        pairs = [(scored[(BASELINE, s, i)], v) for (mm, s, i), v in scored.items()
                 if mm == m and (BASELINE, s, i) in scored]
        if not pairs:
            continue
        wins = sum(1 for a, b in pairs if b[0] and not a[0])
        losses = sum(1 for a, b in pairs if a[0] and not b[0])
        gaps = sorted(b[1] - a[1] for a, b in pairs
                      if a[0] and b[0] and a[1] is not None and b[1] is not None)
        q = lambda f: round(gaps[min(int(f * (len(gaps) - 1)), len(gaps) - 1)], 2) if gaps else None
        better = sum(1 for g in gaps if g < 0)
        worse = sum(1 for g in gaps if g > 0)
        out.append({
            "model": m, "name": label(m), "pairs": len(pairs),
            "baseline_pct": round(sum(a[0] for a, _ in pairs) / len(pairs) * 100, 1),
            "rl_pct": round(sum(b[0] for _, b in pairs) / len(pairs) * 100, 1),
            "wins": wins, "losses": losses, "p_value": sign_test(wins, losses),
            "gap_pairs": len(gaps), "gap_better": better, "gap_worse": worse,
            "gap_sign_p": sign_test(better, worse),
            "gap_mean": round(st.mean(gaps), 2) if gaps else None,
            "gap_median": round(st.median(gaps), 2) if gaps else None,
            "gap_p10": q(0.10), "gap_p90": q(0.90),
        })
    out.sort(key=lambda d: -(d["rl_pct"] - d["baseline_pct"]))

    print(f"{'variant':46s}{'SFT':>7s}{'RL':>7s}{'diff':>7s}{'W':>6s}{'L':>5s}"
          f"{'p':>11s}{'gap mean':>10s}{'median':>8s}")
    for d in out:
        print(f"{d['name'][:44]:46s}{d['baseline_pct']:6.1f}%{d['rl_pct']:6.1f}%"
              f"{d['rl_pct'] - d['baseline_pct']:+7.1f}{d['wins']:6d}{d['losses']:5d}"
              f"{d['p_value']:11.2e}{str(d['gap_mean']):>10s}{str(d['gap_median']):>8s}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
