"""Step 1 — Download datasets.

Downloads:
    StarJob  : HuggingFace mideavalwisard/Starjob → data/starjob_raw.json
    SM sample: filter to jobs≤10 × machines≤10    → data/starjob_train_sm.jsonl
    OOD      : OR-Library jobshop1.txt             → data/benchmarks/jobshop1.txt

Run:
    python scripts/01_download_data.py

Options:
    --skip-starjob   skip StarJob download (if already downloaded)
    --skip-ood       skip OR-Library download
    --max-jobs N     SM filter: max number of jobs (default 10)
    --max-machines N SM filter: max number of machines (default 10)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cap_grpo.dataset import download_ood, download_starjob, sample_sm
from cap_grpo.config import SM_TRAIN_FILE, JOBSHOP1


def main():
    parser = argparse.ArgumentParser(description="Download StarJob + OOD benchmark data")
    parser.add_argument("--skip-starjob",   action="store_true", help="Skip StarJob download")
    parser.add_argument("--skip-ood",       action="store_true", help="Skip OR-Library download")
    parser.add_argument("--max-jobs",       type=int, default=10, help="SM filter: max jobs")
    parser.add_argument("--max-machines",   type=int, default=10, help="SM filter: max machines")
    args = parser.parse_args()

    raw_path = SM_TRAIN_FILE.parent / "starjob_raw.json"

    if not args.skip_starjob:
        if raw_path.exists():
            print(f"[skip] {raw_path} already exists. Use --skip-starjob to skip or delete to re-download.")
        else:
            download_starjob(output_path=raw_path)

        print()
        sample_sm(
            raw_path=raw_path,
            output_path=SM_TRAIN_FILE,
            max_jobs=args.max_jobs,
            max_machines=args.max_machines,
        )
    else:
        print(f"[skip] StarJob download skipped.")

    print()

    if not args.skip_ood:
        if JOBSHOP1.exists():
            print(f"[skip] {JOBSHOP1} already exists. Delete to re-download.")
        else:
            download_ood(output_path=JOBSHOP1)
    else:
        print("[skip] OR-Library download skipped.")

    print("\nData ready:")
    print(f"  SM train  : {SM_TRAIN_FILE}")
    print(f"  OOD bench : {JOBSHOP1}")
    print("\nNext step: python scripts/02_run_sft.py --model llama")


if __name__ == "__main__":
    main()
