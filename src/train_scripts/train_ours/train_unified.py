#!/usr/bin/env python3
"""Portable training launcher for the public HNU-SYS-2023 release."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Train PG-LLM-KT on prepared HNU-SYS-2023 instructions")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face model identifier")
    parser.add_argument("--output-dir", default=str(root / "outputs" / "hnu_sys2023"))
    parser.add_argument("--data-dir", default=str(root / "data" / "instructions" / "hnu_sys2023"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved model command without running it")
    arguments, passthrough = parser.parse_known_args()
    data_dir = Path(arguments.data_dir)
    train_file, val_file = data_dir / "train.json", data_dir / "val.json"
    missing = [str(path) for path in (train_file, val_file) if not path.is_file()]
    if missing:
        parser.error("missing prepared instruction files: " + ", ".join(missing) + ". Run `pgllmkt prepare` first.")
    command = [
        sys.executable, str(root / "src" / "model" / "loraAndPredictor.py"),
        "--model_path", arguments.model_path,
        "--data_path", str(train_file), "--val_path", str(val_file),
        "--output_dir", arguments.output_dir,
        "--batch_size", str(arguments.batch_size),
        "--num_epochs", str(arguments.epochs),
        "--max_seq_length", str(arguments.max_seq_length),
        "--learning_rate", str(arguments.learning_rate),
        *passthrough,
    ]
    if arguments.dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    return subprocess.run(command, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
