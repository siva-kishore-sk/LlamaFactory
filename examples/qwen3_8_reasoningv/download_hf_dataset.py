"""Download a Hugging Face dataset and save it as JSONL.

Usage examples:
    python datasets/download_hf_dataset.py GipAI/ReasoningV
    python datasets/download_hf_dataset.py GipAI/ReasoningV --split train
    python datasets/download_hf_dataset.py org/name --config default --output-dir datasets/out
    python datasets/download_hf_dataset.py org/name --max-rows 1000

Each split is written to <output-dir>/<dataset_name>[_<config>]_<split>.jsonl.
Requires: pip install datasets
For gated/private datasets, run `huggingface-cli login` or set HF_TOKEN.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# This script lives in a folder named "datasets", which would shadow the
# Hugging Face `datasets` package. Remove the script's folder from sys.path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]

try:
    from datasets import load_dataset, get_dataset_split_names
except ImportError:
    sys.exit("The 'datasets' package is required: pip install datasets")


def to_jsonable(value):
    """Convert values that json can't serialize (bytes, dates, images, ...)."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def write_split(dataset_name, config, split, output_dir, max_rows, streaming, token):
    ds = load_dataset(dataset_name, config, split=split, streaming=streaming, token=token)

    safe_name = dataset_name.replace("/", "__")
    parts = [safe_name] + ([config] if config else []) + [split]
    out_path = output_dir / ("_".join(parts) + ".jsonl")

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            if max_rows is not None and count >= max_rows:
                break
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  {split}: {count} rows written...", flush=True)

    print(f"Saved {count} rows -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset as JSONL.")
    parser.add_argument("dataset", help="Dataset id on the Hub, e.g. GipAI/ReasoningV")
    parser.add_argument("--config", default=None, help="Dataset config/subset name")
    parser.add_argument("--split", default=None,
                        help="Split to download (e.g. train). Default: all splits")
    parser.add_argument("--output-dir", default=_SCRIPT_DIR,
                        help="Directory to write JSONL files (default: this script's folder)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Only write the first N rows of each split")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream rows instead of downloading the full dataset first")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        splits = [args.split]
    else:
        splits = get_dataset_split_names(args.dataset, args.config, token=args.token)
    print(f"Dataset: {args.dataset} | config: {args.config} | splits: {splits}")

    for split in splits:
        write_split(args.dataset, args.config, split, output_dir,
                    args.max_rows, args.streaming, args.token)


if __name__ == "__main__":
    main()
