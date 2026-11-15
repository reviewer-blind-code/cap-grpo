"""One name per model, everywhere a name is written.

Three sets of keys were produced over the project: short ones in the grid,
longer ones in the pass@k run, and a third set for the Stage 2 controls. They
also predate the renumbering of the reward variants, so the key `V1` means
variant V4 and `V4` means V7, which is a trap for anyone reading the files.

This rewrites the `model` field of every raw file to the number used in the
tables and keeps the key it was produced under in `model_as_run`, so nothing
is lost. Only that one field is touched; the rest of each line is untouched
bytes.

Run from the repo root, pointing at the raw-data folder:
    python3 analysis/canonical_names.py ~/Desktop/JSSP-SLM-raw
"""
import json
import re
import sys
from pathlib import Path

CANON = {
    # the grid, sm_heldout, the October evaluation, the chat-prompt probe
    "Base-LLaMA": "Base", "Base": "Base",
    "SFT-LoRA": "SFT-LoRA", "SFT-rsLoRA": "SFT-rsLoRA",
    "Vbin": "V1", "VbinMk": "V2", "StratNoMk": "V3",
    "V1": "V4", "V3": "V5", "V6LoRA": "V6", "V4": "V7", "V7": "V8",
    # pass@k and the sampling probes
    "Vbin-feasonly": "V1", "VbinMk-feasmk": "V2", "StratNoMk-feaspen": "V3",
    "V1-stratified": "V4", "V3-stratified": "V5", "V6LoRA-lenmask": "V6",
    "V4-hybrid": "V7", "V7-hybridover": "V8",
    # the Stage 2 controls
    "RANDL42": "T2-1-seed42", "RANDL43": "T2-1-seed43",
    "SFTbudget5e6": "T2-2-lr5e-6", "SFTbudget2e4": "T2-2-lr2e-4",
    "RFT": "T2-3",
}


def convert(path: Path) -> tuple:
    lines = path.read_text().splitlines(keepends=True)
    out, changed, already, unknown = [], 0, 0, set()
    for line in lines:
        if '"model_as_run"' in line:
            already += 1
            out.append(line)
            continue
        m = re.search(r'"model": "([^"]+)"', line)
        if m and m.group(1) in CANON:
            old = m.group(1)
            out.append(line[:m.start()]
                       + '"model": "%s", "model_as_run": "%s"' % (CANON[old], old)
                       + line[m.end():])
            changed += 1
        else:
            out.append(line)
            if m:
                unknown.add(m.group(1))
    if changed:
        path.write_text("".join(out))
    return len(lines), changed, already, unknown


def main():
    root = Path(sys.argv[1]).expanduser()
    total = 0
    for f in sorted(root.rglob("*.jsonl")):
        n, changed, already, unknown = convert(f)
        total += changed
        if changed or already or unknown:
            print(f"{str(f.relative_to(root)):62s} {n:7d} lines, "
                  f"{changed:7d} renamed, {already:7d} already done"
                  + (f", unknown keys {sorted(unknown)}" if unknown else ""))
    print(f"\n{total} records renamed")


if __name__ == "__main__":
    main()
