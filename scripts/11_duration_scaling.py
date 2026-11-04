"""Step 11 — duration scaling.

Multiplying every processing time by a constant leaves the optimal order
unchanged and scales the optimum by the same factor, so anything that changes
is about handling the numbers rather than about scheduling. The long-duration
instances are also divided by ten, with the optimum recomputed.

Run:
    python scripts/11_duration_scaling.py --data /path/to/raw
"""
import argparse
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cap_grpo.rawdata import data_root, label, read_jsonl          # noqa: E402

ORDER = ["x1", "x2", "x5", "x10", "lta_div10"]
TITLE = {"x1": "Base instances x 1", "x2": "Base instances x 2", "x5": "Base instances x 5",
         "x10": "Base instances x 10", "lta_div10": "Long-duration instances / 10"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    root = data_root(args.data)

    rows = []
    for f in sorted((root / "probes").glob("scale_k8*/samples.jsonl")):
        rows += read_jsonl(f)
    if not rows:
        raise SystemExit("no scale samples under <data>/probes/scale_k8*/")

    max_dur = {}
    for r in read_jsonl(root / "instances" / "scale_90.jsonl"):
        max_dur[r["group"]] = max(max_dur.get(r["group"], 0),
                                  max(int(d) for job in r["jobs"] for _, d in job))

    per = defaultdict(lambda: {"n": 0, "valid": 0, "gaps": []})
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
        d = per[(r["group"], r["model"])]
        d["n"] += 1
        d["valid"] += bool(r["feasible"])
        if r["feasible"] and r["gap_pct"] is not None:
            d["gaps"].append(r["gap_pct"])

    print(f"{'set':30s}{'max duration':>14s}" + "".join(f"{label(m)[:20]:>22s}" for m in models))
    for g in ORDER:
        if not any((g, m) in per for m in models):
            continue
        cells = []
        for m in models:
            d = per.get((g, m))
            cells.append(f"{d['valid'] / d['n'] * 100:9.1f}% / gap "
                         f"{st.mean(d['gaps']):5.1f}" if d and d["gaps"]
                         else f"{'-':>22s}")
        print(f"{TITLE.get(g, g):30s}{max_dur.get(g, 0):14d}" + "".join(cells))


if __name__ == "__main__":
    main()
