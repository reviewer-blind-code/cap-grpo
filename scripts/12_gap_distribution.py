"""Step 12 — gap distribution on the instances only RL solves.

Was feasibility bought with quality? The gap of every valid schedule is split
by whether the supervised model also solves that instance, and then split again
by instance size, because the instances only RL solves are the larger ones.

Run:
    python scripts/12_gap_distribution.py --data /path/to/raw --model V4
"""
import argparse
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations                      # noqa: E402
from cap_grpo.rawdata import BASELINE, data_root, generations, instances, label  # noqa: E402


def bucket(ops):
    ops = int(ops)
    return "<= 49" if ops <= 49 else "50-74" if ops <= 74 else "75-99" if ops <= 99 else "100"


def describe(gaps):
    if not gaps:
        return None
    s = sorted(gaps)
    q = lambda f: s[min(int(f * (len(s) - 1)), len(s) - 1)]
    return {"n": len(s), "mean": round(st.mean(s), 2), "median": round(st.median(s), 2),
            "p25": round(q(0.25), 2), "p75": round(q(0.75), 2), "p90": round(q(0.90), 2),
            "le10": round(sum(1 for g in s if g <= 10) / len(s) * 100, 1),
            "le5": round(sum(1 for g in s if g <= 5) / len(s) * 100, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default="V4", help="the RL variant to split by size")
    args = ap.parse_args()
    root = data_root(args.data)

    inst = instances(root, "ood")
    solved, gaps = defaultdict(set), defaultdict(list)
    for g in generations(root, "grid"):
        if g["name"] not in inst:
            continue
        r = check_violations(g["generation"], inst[g["name"]]["jobs"])
        if r["feasible"]:
            solved[g["model"]].add(g["name"])
            bks = inst[g["name"]]["bks"]
            if r["makespan"] and bks:
                gaps[(g["model"], g["name"])].append((r["makespan"] - bks) / bks * 100)

    print(f"{'variant':42s}{'group':18s}{'gens':>6s}{'mean':>7s}{'median':>8s}"
          f"{'p25':>7s}{'p75':>7s}{'p90':>7s}{'<=10%':>7s}{'<=5%':>7s}")
    for m in sorted(solved):
        if m == BASELINE:
            continue
        both = solved[m] & solved[BASELINE]
        only = solved[m] - solved[BASELINE]
        for gname, names in (("solved by both", both), ("solved only by RL", only)):
            d = describe([x for i in names for x in gaps[(m, i)]])
            if not d:
                continue
            print(f"{label(m)[:40]:42s}{gname:18s}{d['n']:6d}{d['mean']:7.2f}{d['median']:8.2f}"
                  f"{d['p25']:7.2f}{d['p75']:7.2f}{d['p90']:7.2f}{d['le10']:6.1f}%{d['le5']:6.1f}%")

    m = args.model
    both, only = solved[m] & solved[BASELINE], solved[m] - solved[BASELINE]
    print(f"\nby instance size, {label(m)}")
    print(f"{'operations':12s}{'both: n':>9s}{'mean gap':>10s}{'RL only: n':>12s}{'mean gap':>10s}")
    for b in ["<= 49", "50-74", "75-99", "100"]:
        bb = [i for i in both if bucket(inst[i]["ops"]) == b]
        oo = [i for i in only if bucket(inst[i]["ops"]) == b]
        gb = [x for i in bb for x in gaps[(m, i)]]
        go = [x for i in oo for x in gaps[(m, i)]]
        print(f"{b:12s}{len(bb):9d}{(st.mean(gb) if gb else float('nan')):10.2f}"
              f"{len(oo):12d}{(st.mean(go) if go else float('nan')):10.2f}")


if __name__ == "__main__":
    main()
