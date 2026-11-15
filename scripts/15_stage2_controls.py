"""Step 15 — the Stage 2 control tests.

Stage 1 showed that the RL stage raises how often the model writes a valid
schedule. It did not show why. Three controls each remove one possible cause,
and all three get exactly what one RL run gets: the same supervised starting
checkpoint, the same 2,000 training records, 500 optimizer updates, and for the
one that samples, the same 8,000 completions.

    test 2-1  the reward is replaced by a draw from N(2.66, 1.93) that never
              reads the schedule, matched to the mean and spread of the real one
    test 2-2  no reward at all: 500 supervised updates on the gold schedules,
              at two learning rates
    test 2-3  the checker is used to filter data instead of to shape a gradient:
              8,000 samples drawn, the valid ones kept, then trained on

Each control is compared with the checkpoint it started from on the same
instance and the same seed, with the two-sided exact sign test used in step 14.
Pairs where both models are valid carry the quality comparison.

Run:
    python scripts/15_stage2_controls.py --data /path/to/raw
    python scripts/15_stage2_controls.py --data /path/to/raw --json out.json
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.rawdata import BASELINE, data_root, generations, label  # noqa: E402

TESTS = [("test_2_1", "test 2-1, an uninformative reward"),
         ("test_2_2", "test 2-2, more supervised training"),
         ("test_2_3", "test 2-3, rejection-sampling fine-tuning")]
CAP = 4096


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)


def index(rows):
    return {(r["model"], r["seed"], r["name"]): r for r in rows}


def summarise(idx, model):
    rows = [v for k, v in idx.items() if k[0] == model]
    feas = [r for r in rows if r["feasible"]]
    gaps = [r["gap_pct"] for r in feas if r["gap_pct"] is not None]
    cut = sum(1 for r in rows if r["gen_tokens"] >= CAP)
    wrong = sum(1 for r in rows if not r["feasible"] and r["gen_tokens"] < CAP)
    return {"model": model, "name": label(model), "answers": len(rows),
            "valid": len(feas), "valid_pct": round(len(feas) / len(rows) * 100, 1),
            "gap_mean": round(st.mean(gaps), 2) if gaps else None,
            "gap_median": round(st.median(gaps), 2) if gaps else None,
            "tokens_mean": round(st.mean(r["gen_tokens"] for r in rows)),
            "cut_off_pct": round(cut / len(rows) * 100, 1),
            "wrong_but_complete_pct": round(wrong / len(rows) * 100, 1)}


def compare(idx, model, base):
    pairs = [(idx[(base, s, i)], idx[(model, s, i)])
             for (m, s, i) in idx if m == model and (base, s, i) in idx]
    wins = sum(1 for a, b in pairs if b["feasible"] and not a["feasible"])
    losses = sum(1 for a, b in pairs if a["feasible"] and not b["feasible"])
    gaps = [b["gap_pct"] - a["gap_pct"] for a, b in pairs
            if a["feasible"] and b["feasible"]
            and a["gap_pct"] is not None and b["gap_pct"] is not None]
    better = sum(1 for g in gaps if g < 0)
    worse = sum(1 for g in gaps if g > 0)
    base_pct = sum(a["feasible"] for a, _ in pairs) / len(pairs) * 100
    run_pct = sum(b["feasible"] for _, b in pairs) / len(pairs) * 100
    return {"model": model, "name": label(model), "baseline": base,
            "pairs": len(pairs), "baseline_pct": round(base_pct, 1),
            "run_pct": round(run_pct, 1), "difference_pp": round(run_pct - base_pct, 1),
            "wins": wins, "losses": losses, "p_value": sign_test(wins, losses),
            "gap_pairs": len(gaps), "gap_better": better, "gap_worse": worse,
            "gap_delta": round(st.mean(gaps), 2) if gaps else None,
            "gap_delta_median": round(st.median(gaps), 2) if gaps else None,
            "gap_sign_p": sign_test(better, worse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--json")
    args = ap.parse_args()
    root = data_root(args.data)

    rows = generations(root, "grid")
    for which, _ in TESTS:
        rows += generations(root, which)
    idx = index(rows)
    models = sorted({k[0] for k in idx})

    controls = [m for m in models if m.startswith("T2-")]
    reference = [m for m in (BASELINE, "V5", "V7") if m in models]

    out = {"baseline": BASELINE,
           "summary": [summarise(idx, m) for m in reference + controls],
           "paired": [compare(idx, m, BASELINE) for m in controls + reference[1:]]}

    print(f"{'model':46s} {'valid':>7s} {'cut off':>8s} {'wrong':>7s} {'gap':>7s}")
    for s in out["summary"]:
        print(f"{s['name'][:44]:46s} {s['valid_pct']:6.1f}% {s['cut_off_pct']:7.1f}% "
              f"{s['wrong_but_complete_pct']:6.1f}% {s['gap_mean']:7.2f}")
    print()
    print(f"{'compared with its supervised start':46s} {'change':>8s} {'w/l':>10s} "
          f"{'p':>10s} {'gap':>8s}")
    for p in out["paired"]:
        print(f"{p['name'][:44]:46s} {p['difference_pp']:+7.1f}pp "
              f"{p['wins']:5d}/{p['losses']:<4d} {p['p_value']:10.2e} "
              f"{p['gap_delta']:+7.2f}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
