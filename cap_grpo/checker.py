"""JSSP feasibility checker — standalone, no ML dependencies.

One function per constraint, written against the problem definition. A schedule
is feasible exactly when all six counters are zero.

    timing_consistency_violations : start + duration = end for every operation
    missing_op_count              : operations a job owes
    over_op_count                 : operations beyond the route, including
                                    operations for a job the instance lacks
    routing_order_violations      : wrong machine or duration at position k
    precedence_violations         : operation k starts before operation k-1 ends
    machine_capacity_violations   : two operations overlap on one machine

An operation whose arithmetic fails has no trustworthy interval: its end time
contradicts its own start and duration. It already makes the schedule
infeasible through the timing counter, so it is counted once there and then
left out of the interval-based checks and of the makespan, rather than dropped
(which would mislabel it as missing). Its machine and duration are still
checked against the route, since neither depends on the interval.

Schedule format expected (one op per token):
    J<job>-M<machine>: <start>+<duration>-><end>

Example:
    J0-M1: 0+3->3, J1-M0: 0+2->2

The previous implementation is kept in cap_grpo/checker_legacy.py; the two
agree on 99.964 % of 25,025 stored generations. docs/CHECKER.md reports the
audit against CP-SAT.
"""
import re

OP_PATTERN = re.compile(r'J\s*(\d+)\s*-\s*M\s*(\d+)\s*:\s*(\d+)\s*\+\s*(\d+)\s*->\s*(\d+)')


def extract_makespan(text: str):
    """Largest end time written anywhere in the text, or None.

    Kept for reading reference schedules, which are well formed by
    construction. Model output is scored with the makespan returned by
    check_violations, which only trusts operations whose arithmetic holds.
    """
    times = re.findall(r'->\s*(\d+)', text)
    return max(int(t) for t in times) if times else None


def parse_all(text: str):
    """Every operation in written order: (job, machine, start, duration, end, pos)."""
    return [(*map(int, m.groups()), pos) for pos, m in enumerate(OP_PATTERN.finditer(text))]


def parse_ops(text: str):
    """(operations whose arithmetic holds, count of those whose arithmetic fails)."""
    ok, bad = [], 0
    for j, mc, s, d, e, _ in parse_all(text):
        if s + d == e:
            ok.append((j, mc, s, d, e))
        else:
            bad += 1
    return ok, bad


def has_interval(op) -> bool:
    """True when the operation's own numbers agree, so [start, end] means something."""
    return op[2] + op[3] == op[4]


def ops_of_job(ops, job):
    """Operations claimed for one job, ordered by start time, ties by written order."""
    return sorted((o for o in ops if o[0] == job), key=lambda o: (o[2], o[5]))


def count_timing(ops) -> int:
    return sum(1 for o in ops if not has_interval(o))


def count_missing(ops, jobs_spec) -> int:
    return sum(max(0, len(req) - sum(1 for o in ops if o[0] == j))
               for j, req in enumerate(jobs_spec))


def count_over(ops, jobs_spec) -> int:
    """Surplus operations, including every operation of a job outside the instance."""
    total = sum(max(0, sum(1 for o in ops if o[0] == j) - len(req))
                for j, req in enumerate(jobs_spec))
    return total + sum(1 for o in ops if o[0] < 0 or o[0] >= len(jobs_spec))


def count_routing(ops, jobs_spec) -> int:
    total = 0
    for j, req in enumerate(jobs_spec):
        emitted = ops_of_job(ops, j)
        for k in range(min(len(emitted), len(req))):
            machine, dur = req[k]
            if emitted[k][1] != machine or emitted[k][3] != dur:
                total += 1
    return total


def count_precedence(ops, jobs_spec) -> int:
    total = 0
    for j in range(len(jobs_spec)):
        chain = [o for o in ops_of_job(ops, j) if has_interval(o)]
        total += sum(1 for prev, cur in zip(chain, chain[1:]) if cur[2] < prev[4])
    return total


def count_capacity(ops) -> int:
    """Per machine, sweep by start time; an operation starting before the running
    maximum end time of that machine is one violation."""
    by_machine = {}
    for o in ops:
        if has_interval(o):
            by_machine.setdefault(o[1], []).append(o)
    total = 0
    for lst in by_machine.values():
        lst.sort(key=lambda o: (o[2], o[5]))
        busy_until = None
        for o in lst:
            if busy_until is not None and o[2] < busy_until:
                total += 1
            busy_until = o[4] if busy_until is None else max(busy_until, o[4])
    return total


def makespan_of(ops):
    """Latest end time among operations with a trustworthy interval."""
    ends = [o[4] for o in ops if has_interval(o)]
    return max(ends) if ends else None


def check_violations(schedule_str: str, jobs_spec: list) -> dict:
    """Check all JSSP constraints and return a result dictionary.

    Args:
        schedule_str : raw schedule text generated by the model
        jobs_spec    : list of jobs, each a list of (machine, duration) tuples
                       e.g. [[(1, 3), (0, 2)], [(0, 2), (1, 4)]]

    Returns:
        dict with keys:
            feasible                        bool
            makespan                        int or None
            ops_emitted                     int
            ops_expected                    int
            missing_op_count                int
            over_op_count                   int
            routing_order_violations        int
            machine_capacity_violations     int
            timing_consistency_violations   int
            precedence_violations           int
            total_violations                int
            unknown_job_ops                 int   operations of a job the instance lacks
    """
    ops = parse_all(schedule_str)
    counts = {
        "timing_consistency_violations": count_timing(ops),
        "missing_op_count":              count_missing(ops, jobs_spec),
        "over_op_count":                 count_over(ops, jobs_spec),
        "routing_order_violations":      count_routing(ops, jobs_spec),
        "precedence_violations":         count_precedence(ops, jobs_spec),
        "machine_capacity_violations":   count_capacity(ops),
    }
    return {
        "feasible":        all(c == 0 for c in counts.values()),
        "makespan":        makespan_of(ops),
        "ops_emitted":     len(ops),
        "ops_expected":    sum(len(j) for j in jobs_spec),
        "total_violations": sum(counts.values()),
        "unknown_job_ops": sum(1 for o in ops if o[0] < 0 or o[0] >= len(jobs_spec)),
        **counts,
    }
