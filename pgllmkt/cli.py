from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import prepare, preprocess, project_root, split


def doctor() -> int:
    root = project_root()
    required = [root / "data/raw/hnu_sys2023/HNU_SYS_2023.txt", root / "data/raw/hnu_sys2023/question2skill.csv"]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {root}")
    print("Dataset: " + ("ready" if not missing else "missing " + ", ".join(missing)))
    return int(bool(missing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pgllmkt", description="PG-LLM-KT reproducibility toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check the local environment and public data")
    prepare_parser = commands.add_parser("prepare", help="run preprocessing, splitting, and instruction generation")
    prepare_parser.add_argument("--with-graph", action="store_true", help="also build and inject prerequisite graphs")
    commands.add_parser("preprocess", help="normalize the public response data")
    commands.add_parser("split", help="create deterministic learner-level splits")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "preprocess":
        print(json.dumps(preprocess(), indent=2))
    elif args.command == "split":
        split()
    elif args.command == "prepare":
        prepare(with_graph=args.with_graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
