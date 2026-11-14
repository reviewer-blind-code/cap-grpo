"""Stage 2-1: does an uninformative reward reproduce the RL gain?

Two GRPO runs are trained with --reward-mode random, which draws the reward from
N(2.66, 1.93) without looking at the schedule.  Everything else matches the eight
reward variants.  Both runs are evaluated on the same grid as every other model
(455 instances x 5 seeds, temperature 0.1, vLLM, checker v2) and compared with the
supervised checkpoint they actually started from, using the same two-sided exact
sign test as evaluation/rl_effect.py.

Writes reports/stage2_random_stats.json.
Run from repo root:  python3 analysis/stage2_random.py
"""
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# the random-reward runs started from the rsLoRA checkpoint (config.py default),
# not from the LoRA checkpoint the eight variants use; see MANIFEST.md
START = "SFT-rsLoRA"
RAND = {"RANDs42": "Random reward, seed 42, off-protocol start",
        "RANDs43": "Random reward, seed 43, off-protocol start"}
# the repeat from the LoRA checkpoint every other model starts from
RANDL = {"RANDL42": "Random reward, seed 42", "RANDL43": "Random reward, seed 43"}
# stage 2-2 and 2-3 start from the LoRA checkpoint, like the eight reward designs
CTRL = {"SFTbudget5e6": "Same-budget supervised, lr 5e-6",
        "SFTbudget2e4": "Same-budget supervised, lr 2e-4",
        "RFT": "Rejection-sampling fine-tuning"}
REF = {"SFT-LoRA": "Supervised, LoRA", "SFT-rsLoRA": "Supervised, rsLoRA (the start of both runs)",
       "V3": "Stratified reward (V5)", "V4": "Hybrid reward, CAP-GRPO (V7)",
       "V1": "Stratified reward (V4)"}
RUNS = {"RANDs42": "full_lora_random_n2000_s42", "RANDs43": "full_lora_random_n2000_s43",
        "RANDL42": "full_lora_random_lora_n2000_s42",
        "RANDL43": "full_lora_random_lora_n2000_s43",
        "V3": "full_lora_stratified_n2000_v3"}


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    return min(2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)


def load(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[(d["model"], d["seed"], d["name"])] = d
    return out


def summary(idx, model):
    rows = [v for k, v in idx.items() if k[0] == model]
    feas = [r for r in rows if r["feasible"]]
    gaps = [r["gap_pct"] for r in feas if r["gap_pct"] is not None]
    trunc = sum(1 for r in rows if r["gen_tokens"] >= 4096)
    broken = sum(1 for r in rows if not r["feasible"] and r["gen_tokens"] < 4096)
    return {"model": model, "generations": len(rows),
            "valid": len(feas), "valid_pct": round(len(feas) / len(rows) * 100, 1),
            "gap_mean": round(st.mean(gaps), 2) if gaps else None,
            "gap_median": round(st.median(gaps), 2) if gaps else None,
            "tokens_mean": round(st.mean(r["gen_tokens"] for r in rows)),
            "at_cap_pct": round(trunc / len(rows) * 100, 1),
            "wrong_but_complete_pct": round(broken / len(rows) * 100, 1)}


def paired(idx, model, base):
    pairs = [(idx[(base, s, i)], idx[(model, s, i)])
             for (m, s, i) in idx if m == model and (base, s, i) in idx]
    wins = sum(1 for a, b in pairs if b["feasible"] and not a["feasible"])
    losses = sum(1 for a, b in pairs if a["feasible"] and not b["feasible"])
    base_pct = sum(a["feasible"] for a, _ in pairs) / len(pairs) * 100
    run_pct = sum(b["feasible"] for _, b in pairs) / len(pairs) * 100
    # quality on the pairs both models solve: a negative change is closer to the optimum
    gaps = [b["gap_pct"] - a["gap_pct"] for a, b in pairs
            if a["feasible"] and b["feasible"]
            and a["gap_pct"] is not None and b["gap_pct"] is not None]
    better = sum(1 for g in gaps if g < 0)
    worse = sum(1 for g in gaps if g > 0)
    return {"model": model, "baseline": base, "pairs": len(pairs),
            "baseline_pct": round(base_pct, 1), "run_pct": round(run_pct, 1),
            "difference_pp": round(run_pct - base_pct, 1),
            "wins": wins, "losses": losses, "p_value": sign_test(wins, losses),
            "gap_pairs": len(gaps), "gap_better": better, "gap_worse": worse,
            "gap_delta": round(st.mean(gaps), 2) if gaps else None,
            "gap_delta_median": round(st.median(gaps), 2) if gaps else None,
            "gap_sign_p": sign_test(better, worse)}


def curve(run):
    state = json.loads((ROOT / "results" / "training_logs" / run / "trainer_state.json").read_text())
    pts = [(h["step"], h.get("completion_length"), h.get("reward"), h.get("kl"))
           for h in state["log_history"] if "reward" in h]
    return {"run": run, "steps": state.get("max_steps"),
            "points": [{"step": s, "completion_length": c, "reward": r, "kl": k}
                       for s, c, r, k in pts]}


def main():
    idx = load(ROOT / "results" / "grid_uniform" / "results.jsonl")
    idx.update(load(ROOT / "results" / "grid_random" / "results.jsonl"))
    for extra in ("grid_stage2a", "grid_random_lora"):
        p = ROOT / "results" / extra / "results.jsonl"
        if p.exists():
            idx.update(load(p))

    have = {k for k, _, _ in idx}
    ctrl = [m for m in CTRL if m in have]
    randl = [m for m in RANDL if m in have]
    out = {"names": {**RAND, **RANDL, **CTRL, **REF}, "start": START,
           "summary": [summary(idx, m) for m in randl + list(RAND) + ctrl + list(REF)],
           "paired": [paired(idx, m, START) for m in RAND]
                     + [paired(idx, m, "SFT-LoRA") for m in randl + ctrl + ["V3", "V4"]],
           "curves": {k: curve(v) for k, v in RUNS.items()
                      if (ROOT / "results" / "training_logs" / v / "trainer_state.json").exists()}}
    (ROOT / "reports" / "stage2_random_stats.json").write_text(json.dumps(out, indent=2))
    for s in out["summary"]:
        print(f"{s['model']:12s} valid {s['valid_pct']:5.1f}%  gap mean {s['gap_mean']}  "
              f"tokens {s['tokens_mean']:5d}  at cap {s['at_cap_pct']:5.1f}%")
    for p in out["paired"]:
        print(f"{p['model']:12s} vs {p['baseline']:12s} {p['baseline_pct']:5.1f} -> {p['run_pct']:5.1f} "
              f"({p['difference_pp']:+.1f} pp)  w/l {p['wins']}/{p['losses']}  p={p['p_value']:.2e}"
              f"   gap {p['gap_delta']:+.2f} pp (p={p['gap_sign_p']:.2e})")


if __name__ == "__main__":
    main()
