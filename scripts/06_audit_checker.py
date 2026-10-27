"""Step 6 — Audit the feasibility checker against CP-SAT.

Three tests per instance, none of which involves a language model:

    round trip   CP-SAT solves the instance, the schedule is written in the
                 model's output format, and the checker must accept it
    makespan     the checker's makespan must equal the CP-SAT objective, and
                 must never fall below it
    mutation     five defects are injected into that same optimal schedule and
                 the checker must reject every one

The mutations are: delete an operation, duplicate one, break the arithmetic of
one, overlap two operations on a machine, and move one operation to the wrong
machine.

Run:
    python scripts/06_audit_checker.py                      # FT+LA, 18 instances
    python scripts/06_audit_checker.py --limit 30           # 30 s per instance
    python scripts/06_audit_checker.py --out results/audit.json

Requires ortools:  pip install ortools
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cap_grpo.checker import check_violations          # noqa: E402
from cap_grpo.dataset import load_ood_benchmarks       # noqa: E402


def solve(jobs, n_machines, limit):
    """CP-SAT optimum: returns (makespan, schedule as (job, pos, machine, start, dur), status)."""
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    horizon = sum(d for ops in jobs for _, d in ops)
    starts, ends, per_machine = {}, {}, {m: [] for m in range(n_machines)}
    for j, ops in enumerate(jobs):
        for k, (machine, dur) in enumerate(ops):
            s = model.NewIntVar(0, horizon, f"s{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e{j}_{k}")
            per_machine[machine].append(model.NewIntervalVar(s, dur, e, f"i{j}_{k}"))
            starts[j, k], ends[j, k] = s, e
            if k:
                model.Add(s >= ends[j, k - 1])
    for m in range(n_machines):
        model.AddNoOverlap(per_machine[m])
    span = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(span, [ends[j, len(jobs[j]) - 1] for j in range(len(jobs))])
    model.Minimize(span)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, "no_solution"
    schedule = [(j, k, jobs[j][k][0], solver.Value(starts[j, k]), jobs[j][k][1])
                for j in range(len(jobs)) for k in range(len(jobs[j]))]
    return int(solver.Value(span)), schedule, (
        "optimal" if status == cp_model.OPTIMAL else "feasible")


def render(schedule):
    """Schedule -> the text format the models are asked to produce."""
    return ", ".join(f"J{j}-M{m}: {s}+{d}->{s + d}" for j, _, m, s, d in schedule)


def mutate(schedule, rng):
    """Five defective variants of one optimal schedule, already rendered as text."""
    out = []

    dropped = list(schedule)
    dropped.pop(rng.randrange(len(dropped)))
    out.append(("deleted_operation", render(dropped)))

    duplicated = list(schedule)
    duplicated.append(duplicated[rng.randrange(len(duplicated))])
    out.append(("duplicated_operation", render(duplicated)))

    i = rng.randrange(len(schedule))
    broken = ", ".join(
        f"J{j}-M{m}: {s}+{d}->{s + d + 7 if k == i else s + d}"
        for k, (j, _, m, s, d) in enumerate(schedule))
    out.append(("broken_arithmetic", broken))

    by_machine = sorted(schedule, key=lambda o: (o[2], o[3]))
    pair = next(((a, b) for a, b in zip(by_machine, by_machine[1:]) if a[2] == b[2]), None)
    if pair:
        first, second = pair
        overlapped = [(j, k, m, max(0, first[3] + first[4] - 1), d)
                      if (j, k) == (second[0], second[1]) else (j, k, m, s, d)
                      for j, k, m, s, d in schedule]
        out.append(("machine_overlap", render(overlapped)))

    wrong = list(schedule)
    i = rng.randrange(len(wrong))
    j, k, m, s, d = wrong[i]
    wrong[i] = (j, k, m + 1, s, d)
    out.append(("wrong_machine", render(wrong)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=10.0, help="CP-SAT seconds per instance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = load_ood_benchmarks()
    rng = random.Random(42)
    solved = clean = mutations = 0
    failures = {"round_trip": [], "makespan_mismatch": [], "below_optimum": [], "mutation_missed": []}

    for r in records:
        jobs = [[(int(m), int(d)) for m, d in ops] for ops in r["jobs_spec"]]
        n_machines = max(m for ops in jobs for m, _ in ops) + 1
        span, schedule, status = solve(jobs, n_machines, args.limit)
        if span is None:
            continue
        solved += 1

        verdict = check_violations(render(schedule), jobs)
        if not verdict["feasible"]:
            failures["round_trip"].append(r["name"])
        else:
            clean += 1
        if verdict["makespan"] != span:
            failures["makespan_mismatch"].append(r["name"])
        if verdict["makespan"] is not None and verdict["makespan"] < span:
            failures["below_optimum"].append(r["name"])

        for kind, text in mutate(schedule, rng):
            mutations += 1
            if check_violations(text, jobs)["feasible"]:
                failures["mutation_missed"].append(f"{r['name']}:{kind}")

        print(f"  {r['name']:8s} optimum {span:6d} ({status})", flush=True)

    summary = {
        "instances": len(records), "solved": solved, "round_trip_clean": clean,
        "mutations_checked": mutations,
        "failures": {k: len(v) for k, v in failures.items()},
        "detail": failures,
    }
    print(f"\nround trips clean: {clean}/{solved} | mutations checked: {mutations} | "
          f"failures: {summary['failures']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"written {args.out}")
    return 1 if any(summary["failures"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
