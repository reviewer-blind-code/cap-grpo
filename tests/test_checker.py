"""Hand-built cases for the feasibility checker, one rule each.

Every case states the expected verdict before it is run. Both the current
checker and the legacy one are scored, so the three corrected rules are visible
as failures of the legacy implementation.

Run:  python tests/test_checker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations as current          # noqa: E402
from cap_grpo.checker_legacy import check_violations as legacy    # noqa: E402

# J0: M0 for 3, then M1 for 2    J1: M1 for 4, then M0 for 1
JOBS = [[(0, 3), (1, 2)], [(1, 4), (0, 1)]]
VALID = "J0-M0: 0+3 -> 3,J0-M1: 4+2 -> 6,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5"

# two single-operation jobs competing for one machine: the only way to test two
# operations touching exactly, without a route conflict
JOBS_TOUCH = [[(0, 3)], [(0, 2)]]

COUNTER = {"timing": "timing_consistency_violations", "missing": "missing_op_count",
           "extra": "over_op_count", "routing": "routing_order_violations",
           "precedence": "precedence_violations", "capacity": "machine_capacity_violations"}

# (id, what it tests, schedule text, expected feasible, expected counter or None, jobs)
CASES = [
    ("C01", "Valid schedule, positive control", VALID, True, None),
    ("C02", "Job operation starts before its predecessor ends",
     "J0-M0: 0+3 -> 3,J0-M1: 2+2 -> 4,J1-M1: 4+4 -> 8,J1-M0: 8+1 -> 9", False, ("precedence", 1)),
    ("C03", "Two operations overlap on one machine",
     "J0-M0: 0+3 -> 3,J0-M1: 0+2 -> 2,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5", False, ("capacity", 1)),
    ("C04", "Two operations touch exactly on one machine, end equals start",
     "J0-M0: 0+3 -> 3,J1-M0: 3+2 -> 5", True, None, JOBS_TOUCH),
    ("C04b", "The same two operations overlapping by one unit",
     "J0-M0: 0+3 -> 3,J1-M0: 2+2 -> 4", False, ("capacity", 1), JOBS_TOUCH),
    ("C05", "Two operations start at the same time on different machines",
     VALID, True, None),
    ("C06", "Idle time inserted, valid but not optimal",
     "J0-M0: 0+3 -> 3,J0-M1: 10+2 -> 12,J1-M1: 0+4 -> 4,J1-M0: 20+1 -> 21", True, None),
    ("C07", "Lines written in scrambled order, times unchanged",
     "J1-M0: 4+1 -> 5,J0-M1: 4+2 -> 6,J1-M1: 0+4 -> 4,J0-M0: 0+3 -> 3", True, None),
    ("C08", "Machines of the two route positions swapped within a job",
     "J0-M1: 0+3 -> 3,J0-M0: 4+2 -> 6,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5", False, ("routing", 2)),
    ("C09", "Duration does not match the route",
     "J0-M0: 0+9 -> 9,J0-M1: 10+2 -> 12,J1-M1: 0+4 -> 4,J1-M0: 20+1 -> 21", False, ("routing", 1)),
    ("C10", "One operation removed",
     "J0-M0: 0+3 -> 3,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5", False, ("missing", 1)),
    ("C11", "One operation duplicated exactly",
     VALID + ",J1-M0: 4+1 -> 5", False, ("extra", 1)),
    ("C12", "Operations for a job the instance does not have",
     VALID + ",J2-M0: 6+3 -> 9,J2-M1: 9+2 -> 11", False, ("extra", 2)),
    ("C13", "Arithmetic does not hold, start plus duration is not the end",
     "J0-M0: 0+3 -> 9,J0-M1: 4+2 -> 6,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5", False, ("timing", 1)),
    ("C14", "Very large numbers, seven digits",
     "J0-M0: 0+3 -> 3,J0-M1: 1000000+2 -> 1000002,J1-M1: 0+4 -> 4,J1-M0: 2000000+1 -> 2000001",
     True, None),
    ("C15", "Empty output", "Solution:\n", False, ("missing", 4)),
    ("C16", "Trailing prose and stray numbers after the schedule",
     VALID + "\nMaximum end completion time or Makespan: 6\nThat is the answer, 42 units saved.",
     True, None),
    ("C17", "Both jobs given each other's full route",
     "J0-M1: 0+4 -> 4,J0-M0: 4+1 -> 5,J1-M0: 0+3 -> 3,J1-M1: 5+2 -> 7", False, ("routing", 4)),
    ("C18", "Operation listed twice with different times",
     "J0-M0: 0+3 -> 3,J0-M0: 6+3 -> 9,J0-M1: 4+2 -> 6,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5",
     False, ("extra", 1)),
    ("C19", "Machine capacity broken by one time unit only",
     "J0-M0: 0+3 -> 3,J0-M1: 3+2 -> 5,J1-M1: 0+4 -> 4,J1-M0: 4+1 -> 5", False, ("capacity", 1)),
    ("C20", "Schedule of a different, larger instance pasted in",
     "J0-M0: 0+5 -> 5,J1-M1: 0+7 -> 7,J2-M2: 0+2 -> 2,J3-M3: 0+9 -> 9", False, None),
]

def run():
    rows, failed = [], 0
    for case in CASES:
        cid, what, text, want_feasible, want_counter = case[:5]
        jobs = case[5] if len(case) > 5 else JOBS
        out = []
        for name, fn in (("current", current), ("legacy", legacy)):
            r = fn(text, jobs)
            ok = r["feasible"] == want_feasible
            got = ""
            if want_counter:
                key, value = want_counter
                got = r[COUNTER[key]]
                ok = ok and got == value
            out.append((name, r["feasible"], got, ok))
        if not out[0][3]:
            failed += 1
        rows.append((cid, what, want_feasible, want_counter, out))
    return rows, failed


def main():
    rows, failed = run()
    width = max(len(r[1]) for r in rows)
    print(f"{'case':6s}{'tests':{width}s}  {'expected':>10s}  {'current':>9s}  {'legacy':>9s}")
    for cid, what, want, counter, out in rows:
        exp = ("feasible" if want else "infeasible") + (f", {counter[0]}={counter[1]}" if counter else "")
        cur = "pass" if out[0][3] else "FAIL"
        leg = "pass" if out[1][3] else "FAIL"
        print(f"{cid:6s}{what:{width}s}  {exp:>10s}  {cur:>9s}  {leg:>9s}")
    n = len(rows)
    print(f"\ncurrent {n - failed}/{n}   legacy "
          f"{sum(1 for r in rows if r[4][1][3])}/{n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
