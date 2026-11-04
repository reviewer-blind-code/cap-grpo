"""Step 14 — why the hosted models fail.

"0 of 18 feasible" does not say whether the model wrote nothing usable, ran out
of tokens, or produced a complete schedule that breaks a constraint. Every
stored answer is re-scored with this repository's checker and classified:

    valid         a legal schedule
    violates      the full operation set is there, a constraint fails
    stopped early parsed, too few operations, did not reach the token limit
    truncated     the answer reached the token limit

Run:
    python scripts/14_frontier_failures.py --data /path/to/raw
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.checker import check_violations, parse_all          # noqa: E402
from cap_grpo.rawdata import data_root                            # noqa: E402

AT_CAP = 0.98


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    files = sorted((root / "frontier").glob("metrics_*.json"))
    if not files:
        raise SystemExit("no hosted-model results under <data>/frontier/")

    summary = []
    for path in files:
        if "BROKEN" in path.name:
            continue
        d = json.loads(path.read_text())
        if not d.get("results"):
            continue
        cap = d.get("max_tokens")
        name = re.sub(r"^metrics_(openai|groq)_|_benchmarks\.json$", "", path.name)
        kinds, hit = Counter(), 0
        for rec in d["results"]:
            text = rec.get("raw_output") or ""
            ops = parse_all(text)
            truncated = bool(cap and rec.get("gen_tokens", 0) >= cap * AT_CAP)
            hit += truncated
            feasible = bool(rec.get("feasible"))
            required = rec.get("ops_expected")
            if feasible:
                kinds["valid"] += 1
            elif not ops:
                kinds["truncated" if truncated else "no output"] += 1
            elif truncated:
                kinds["truncated"] += 1
            elif required and len(ops) < int(required):
                kinds["stopped early"] += 1
            else:
                kinds["violates"] += 1
        n = len(d["results"])
        summary.append({"model": name, "max_tokens": cap, "instances": n,
                        "valid": kinds["valid"], "violates": kinds["violates"],
                        "stopped_early": kinds["stopped early"],
                        "truncated": kinds["truncated"], "no_output": kinds["no output"],
                        "hit_cap_pct": round(hit / n * 100, 1)})

    print(f"{'model':30s}{'cap':>8s}{'n':>4s}{'valid':>7s}{'violates':>10s}"
          f"{'stopped':>9s}{'truncated':>11s}{'hit cap':>9s}")
    for d in summary:
        print(f"{d['model'][:28]:30s}{d['max_tokens']:8d}{d['instances']:4d}{d['valid']:7d}"
              f"{d['violates']:10d}{d['stopped_early']:9d}{d['truncated']:11d}"
              f"{d['hit_cap_pct']:8.1f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nwritten {args.json}")


if __name__ == "__main__":
    main()
