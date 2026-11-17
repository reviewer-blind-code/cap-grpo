# The scripts as they ran

These are the files that produced the Stage 2 results, copied from the private
training repository without change. They are written against that repository's
module layout, not against `cap_grpo/`, so they will not run here as they are.
They are included because the exact code that produced a number is part of the
record.

| File | What it did |
|---|---|
| `sft_budget.py` | test 2-2: 500 supervised updates over the same 2,000 records, no reward |
| `sample_rft.py` | test 2-3, first half: draw 8,000 samples, keep the ones the checker accepts |
| `stage2_random.py` | the analysis: validity, gap, token length and the paired sign test |
| `run_grid.py` | the uniform evaluation grid used for every model |
| `run_passk.py` | the pass@k run, with `--store-text` added for the repeat |
| `_stage2a.sh` | the launcher for tests 2-2 and 2-3 and their evaluation |
| `_rerun21.sh` | the launcher for the repeat of test 2-1 |
| `_grid_full.sh` | the launcher that evaluated the epoch's final checkpoint |

`scripts/15_stage2_controls.py` recomputes every Stage 2 number in the README
from the raw data folder, and that one does run here.
