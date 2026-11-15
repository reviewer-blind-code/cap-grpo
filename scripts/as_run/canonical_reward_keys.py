"""The reward key inside a training log, renamed to the variant it belongs to.

`trainer_state.json` logs the reward under the mode name used in the code, so
the log of variant V1 says `rewards/jssp_reward_uniform` and the log of V8 says
`rewards/jssp_reward_hybrid_v7`, where that `v7` is the reward's old label and
not variant V7. The folder says one thing and the file says another.

This renames that key to the variant, and records the name it was produced
under in a top-level `reward_mode_as_run`, so the original is still readable.
The console `training.log` beside it is never touched.

Run from the repo root, pointing at a folder of training logs:
    python3 analysis/canonical_reward_keys.py ~/Desktop/JSSP-SLM-raw/03_rl_stage
"""
import json
import re
import sys
from pathlib import Path


def variant_of(state_path: Path) -> str:
    """The variant a log belongs to, read from its path."""
    parts = state_path.parts
    for part in parts:
        m = re.match(r"(V[1-8])_", part)
        if m:
            return m.group(1)
    for part, tag in (("test_2_1_random_reward", "T2-1"),
                      ("test_2_2_more_supervised", "T2-2"),
                      ("test_2_3_rejection_sampling", "T2-3")):
        if part in parts:
            leaf = state_path.parent.name
            return tag if leaf == "run" else f"{tag}-{leaf}"
    return state_path.parent.name


def convert(state_path: Path, variant: str) -> int:
    d = json.loads(state_path.read_text())
    if "reward_mode_as_run" in d:
        return 0
    modes, n = set(), 0
    for entry in d.get("log_history", []):
        for key in [k for k in entry if k.startswith("rewards/")]:
            modes.add(key.split("jssp_reward_", 1)[-1])
            entry[f"rewards/{variant}"] = entry.pop(key)
            n += 1
    if modes:
        d["reward_mode_as_run"] = sorted(modes)[0]
        d["variant"] = variant
        state_path.write_text(json.dumps(d, indent=1))
    return n


def main():
    root = Path(sys.argv[1]).expanduser()
    for state in sorted(root.rglob("trainer_state*.json")):
        variant = variant_of(state)
        n = convert(state, variant)
        print(f"{str(state.relative_to(root)):56s} {variant:12s} {n:5d} steps renamed")


if __name__ == "__main__":
    main()
