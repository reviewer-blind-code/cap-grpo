# The feasibility checker

Two independent implementations score every schedule in this repository.

| File | Role |
|---|---|
| `cap_grpo/checker_legacy.py` | the earlier checker, kept so the difference can be measured |
| `cap_grpo/checker.py` | the checker in use, written from the problem definition, one function per constraint |

The six counters are independent functions. A schedule is feasible when, and only when, all six
counters are zero.

| Counter | Rule |
|---|---|
| timing | every emitted operation satisfies `start + duration = end` |
| missing | every job emits as many operations as its route requires |
| surplus | no job emits more than its route requires, and no operation belongs to a job outside the instance |
| routing | operation *i* of a job uses the machine and duration its route gives |
| precedence | consecutive operations of a job do not overlap |
| capacity | operations on one machine do not overlap |

An operation whose arithmetic fails is kept and counted once under timing; it
has no interval, so it is skipped by the precedence, capacity and makespan
steps rather than silently dropped.

## How to run it

```bash
python tests/test_checker.py            # 21 hand-built cases, one rule each
python scripts/06_audit_checker.py      # CP-SAT round trip and five mutation types
```

The audit needs `ortools`; everything else in the checker is dependency-free.

## Audit result

| Test | Coverage | Result |
|---|---|---|
| CP-SAT optimal schedule accepted | 455 of 455 instances | 0 failures |
| Checker makespan equals the CP-SAT objective | 455 instances | 0 mismatches |
| Feasible schedule below the proven optimum | 25,025 generations | 0, as required |
| Injected defects detected | 2,275 mutations | 0 missed |
| Hand-built cases | 21 cases | v2 21/21, v1 20/21 |
| v1 against v2 on real answers | 25,025 generations | agree on 99.964 % |

## The defect the earlier checker had

Operations belonging to a job the instance does not have were skipped
entirely, so they could never make a schedule infeasible. 474 of 25,025 generations contain
such an operation and 9 of them passed v1 as feasible, at most 5 answers for
any one model, about 0.2 points. Every figure reported now is scored with v2,
so none of them carries the defect.

Two further rules differ: an operation whose arithmetic fails is now kept and
counted once under timing rather than dropped and re-reported as missing, and
machine overlap is detected with a running maximum end time rather than between
neighbouring operations only.
