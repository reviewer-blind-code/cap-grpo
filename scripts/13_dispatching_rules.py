"""Step 13 — classical dispatching rules as a reference point.

Five standard priority rules on the same instances, non-delay simulation. Every
schedule a rule produces is valid by construction, which is the comparison that
matters: the models answer some of the instances, the rules answer all of them.

    FIFO  the job that became ready first      SPT   shortest processing time
    LPT   longest processing time              MWKR  most work remaining
    LWKR  least work remaining

Run:
    python scripts/13_dispatching_rules.py --data /path/to/raw
"""
import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations                      # noqa: E402
from cap_grpo.rawdata import data_root, generations, instances, label  # noqa: E402

RULES = ["FIFO", "SPT", "LPT", "MWKR", "LWKR"]


def schedule(jobs, rule):
    n = len(jobs)
    pos, job_free, ready = [0] * n, [0] * n, [0] * n
    machine_free = defaultdict(int)
    remaining = [sum(d for _, d in ops) for ops in jobs]
    done, total, makespan = 0, sum(len(o) for o in jobs), 0
    while done < total:
        cands = [(j, *jobs[j][pos[j]]) for j in range(n) if pos[j] < len(jobs[j])]
        earliest = min(max(job_free[j], machine_free[m]) for j, m, _ in cands)
        pool = [c for c in cands if max(job_free[c[0]], machine_free[c[1]]) <= earliest]

        def key(c):
            j, _, d = c
            return {"FIFO": (ready[j], j), "SPT": (d, j), "LPT": (-d, j),
                    "MWKR": (-remaining[j], j), "LWKR": (remaining[j], j)}[rule]

        j, m, d = min(pool, key=key)
        start = max(job_free[j], machine_free[m])
        job_free[j] = machine_free[m] = ready[j] = start + d
        remaining[j] -= d
        pos[j] += 1
        done += 1
        makespan = max(makespan, start + d)
    return makespan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    root = data_root(args.data)
    inst = instances(root, "ood")

    rule_gap, best_gap = {}, {}
    per_family = defaultdict(lambda: defaultdict(list))
    for name, r in inst.items():
        gaps = {}
        for rule in RULES:
            ms = schedule(r["jobs"], rule)
            gaps[rule] = (ms - r["bks"]) / r["bks"] * 100 if r["bks"] else None
            per_family[r["set"]][rule].append(gaps[rule])
            per_family["ALL"][rule].append(gaps[rule])
        rule_gap[name] = gaps["MWKR"]
        best_gap[name] = min(gaps.values())
        per_family[r["set"]]["best"].append(best_gap[name])
        per_family["ALL"]["best"].append(best_gap[name])

    print(f"{'family':12s}{'n':>6s}" + "".join(f"{r:>9s}" for r in RULES) + f"{'best':>9s}")
    for fam in sorted(per_family):
        d = per_family[fam]
        n = len(d["MWKR"])
        print(f"{fam:12s}{n:6d}" + "".join(f"{st.mean(d[r]):8.1f}%" for r in RULES)
              + f"{st.mean(d['best']):8.1f}%")

    # --- the models on the instances they answer
    per = defaultdict(lambda: defaultdict(list))
    for g in generations(root, "grid"):
        if g["name"] not in inst:
            continue
        r = check_violations(g["generation"], inst[g["name"]]["jobs"])
        if r["feasible"] and r["makespan"] and inst[g["name"]]["bks"]:
            per[g["model"]][g["name"]].append(
                (r["makespan"] - inst[g["name"]]["bks"]) / inst[g["name"]]["bks"] * 100)

    rows = []
    for m, byinst in per.items():
        names = sorted(byinst)
        mean_of = {i: st.mean(byinst[i]) for i in names}
        rows.append({
            "model": m, "name": label(m), "instances_solved": len(names),
            "coverage_pct": round(len(names) / len(inst) * 100, 1),
            "model_gap_mean": round(st.mean([mean_of[i] for i in names]), 2),
            "best_of_seeds_mean": round(st.mean([min(byinst[i]) for i in names]), 2),
            "mwkr_same_instances": round(st.mean([rule_gap[i] for i in names]), 2),
            "best_rule_same_instances": round(st.mean([best_gap[i] for i in names]), 2),
            "beats_mwkr_pct": round(sum(1 for i in names if mean_of[i] < rule_gap[i])
                                    / len(names) * 100, 1)})
    rows.sort(key=lambda d: d["model_gap_mean"])

    print(f"\n{'model':46s}{'solved':>8s}{'cover':>8s}{'model gap':>11s}"
          f"{'best seed':>11s}{'MWKR':>8s}{'best rule':>11s}{'beats MWKR':>12s}")
    for d in rows:
        print(f"{d['name'][:44]:46s}{d['instances_solved']:8d}{d['coverage_pct']:7.1f}%"
              f"{d['model_gap_mean']:11.2f}{d['best_of_seeds_mean']:11.2f}"
              f"{d['mwkr_same_instances']:8.2f}{d['best_rule_same_instances']:11.2f}"
              f"{d['beats_mwkr_pct']:11.1f}%")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
