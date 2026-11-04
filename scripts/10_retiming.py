"""Step 10 — re-timing diagnostic.

Keep the ordering the model chose, discard its start and end times, recompute
them so each operation starts when its job and its machine are both free, and
score the result again. A repair can fail in two ways: the set of operations is
wrong, or the ordering contradicts the routes. If feasibility jumps, the
bottleneck was arithmetic; it does not jump.

Run:
    python scripts/10_retiming.py --data /path/to/raw
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations, parse_all           # noqa: E402
from cap_grpo.rawdata import data_root, generations, instances, label  # noqa: E402


def repair(ops, jobs):
    """(status, makespan) with status in ok / wrong_set / cycle."""
    by_job = defaultdict(list)
    for o in ops:
        if o[0] < 0 or o[0] >= len(jobs):
            return "wrong_set", None
        by_job[o[0]].append(o)
    for j, route in enumerate(jobs):
        seq = sorted(by_job.get(j, []), key=lambda o: (o[2], o[5]))
        if len(seq) != len(route):
            return "wrong_set", None
        for o, (machine, dur) in zip(seq, route):
            if o[1] != machine or o[3] != dur:
                return "wrong_set", None

    pos, succ, indeg = {}, defaultdict(list), defaultdict(int)
    nodes = [(j, k) for j, route in enumerate(jobs) for k in range(len(route))]
    for j, route in enumerate(jobs):
        for k, o in enumerate(sorted(by_job[j], key=lambda o: (o[2], o[5]))):
            pos[id(o)] = (j, k)
        for k in range(1, len(route)):
            succ[(j, k - 1)].append((j, k))
            indeg[(j, k)] += 1
    by_machine = defaultdict(list)
    for o in ops:
        by_machine[o[1]].append(o)
    for lst in by_machine.values():
        lst.sort(key=lambda o: (o[2], o[5]))
        for a, b in zip(lst, lst[1:]):
            succ[pos[id(a)]].append(pos[id(b)])
            indeg[pos[id(b)]] += 1

    start = {n: 0 for n in nodes}
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        j, k = n
        end = start[n] + jobs[j][k][1]
        for m in succ[n]:
            start[m] = max(start[m], end)
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if seen != len(nodes):
        return "cycle", None
    return "ok", max(start[(j, k)] + jobs[j][k][1] for j, k in nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    inst = instances(root, "ood")
    per = defaultdict(lambda: {"n": 0, "raw": 0, "ok": 0, "wrong_set": 0, "cycle": 0})
    for g in generations(root, "grid"):
        if g["name"] not in inst:
            continue
        jobs = inst[g["name"]]["jobs"]
        ops = parse_all(g["generation"])
        d = per[g["model"]]
        d["n"] += 1
        d["raw"] += check_violations(g["generation"], jobs)["feasible"]
        status, _ = repair(ops, jobs)
        d["ok" if status == "ok" else status] += 1

    rows = []
    for m, d in per.items():
        rows.append({"model": m, "name": label(m), "generations": d["n"],
                     "valid_raw_pct": round(d["raw"] / d["n"] * 100, 1),
                     "valid_repaired_pct": round(d["ok"] / d["n"] * 100, 1),
                     "recovered_pp": round((d["ok"] - d["raw"]) / d["n"] * 100, 1),
                     "wrong_set_pct": round(d["wrong_set"] / d["n"] * 100, 1),
                     "cycle_pct": round(d["cycle"] / d["n"] * 100, 1)})
    rows.sort(key=lambda r: -r["valid_repaired_pct"])

    print(f"{'model':46s}{'gens':>7s}{'valid raw':>11s}{'repaired':>10s}"
          f"{'recovered':>11s}{'wrong set':>11s}{'cycle':>7s}")
    for r in rows:
        print(f"{r['name'][:44]:46s}{r['generations']:7d}{r['valid_raw_pct']:10.1f}%"
              f"{r['valid_repaired_pct']:9.1f}%{r['recovered_pp']:+10.1f}"
              f"{r['wrong_set_pct']:10.1f}%{r['cycle_pct']:6.1f}%")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
