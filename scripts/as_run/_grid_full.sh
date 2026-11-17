#!/bin/bash
# Evaluate the completed-epoch SFT checkpoint (14,406) on the same 455-instance
# grid, 5 seeds, that every other model in the project was measured on.
cd ~/Documents/Starjob
mkdir -p logs/grid_full
source venv-vllm/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "start $(date -Is)" > logs/grid_full/summary.log
python -u evaluation/run_grid.py --models SFT-LoRA-full --seeds 0 1 2 3 4 \
    --out grid_sft_full > logs/grid_full/grid.log 2>&1
echo "exit $? at $(date -Is)" >> logs/grid_full/summary.log
python3 - <<'PY' >> logs/grid_full/summary.log 2>&1
import json, statistics as st
rows=[json.loads(l) for l in open("grid_sft_full/results.jsonl")]
by={}
for r in rows: by.setdefault(r["seed"],[]).append(r)
v=[100*sum(x["feasible"] for x in g)/len(g) for g in by.values()]
print("n rows", len(rows), "seeds", sorted(by))
print("valid %% per seed:", [round(x,1) for x in v])
print("mean valid %.2f  sd %.2f" % (st.mean(v), st.pstdev(v)))
gaps=[r["gap_pct"] for r in rows if r.get("gap_pct") is not None]
print("feasible answers with a gap:", len(gaps), "median gap %.2f" % st.median(gaps))
PY
