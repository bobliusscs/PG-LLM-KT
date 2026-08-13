from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

DATASET = "hnu_sys2023"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _skills(path: Path) -> dict[int, list[int]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or len(rows.fieldnames) < 2:
            raise ValueError(f"Invalid question-to-skill matrix: {path}")
        question_column, *skill_columns = rows.fieldnames
        result: dict[int, list[int]] = {}
        for row in rows:
            question = int(row[question_column])
            result[question] = [int(skill) for skill in skill_columns if row[skill] in {"1", "1.0"}]
        return result


def preprocess(root: Path | None = None) -> dict[str, int | float]:
    root = root or project_root()
    raw = root / "data" / "raw" / DATASET
    output = root / "data" / "processed" / DATASET
    response_file = raw / "HNU_SYS_2023.txt"
    mapping_file = raw / "question2skill.csv"
    if not response_file.is_file() or not mapping_file.is_file():
        raise FileNotFoundError(f"Expected the public dataset under {raw}")

    question_skills = _skills(mapping_file)
    records: list[tuple[str, list[int], list[int]]] = []
    with response_file.open(encoding="utf-8") as handle:
        for number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                learner, sequence = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed response line {number}") from exc
            interactions = [item.split() for item in sequence.split(",")]
            if any(len(item) != 2 or item[1] not in {"0", "1"} for item in interactions):
                raise ValueError(f"Malformed interaction on line {number}")
            records.append((learner, [int(item[0]) for item in interactions], [int(item[1]) for item in interactions]))

    learners = {learner: index for index, learner in enumerate(sorted({r[0] for r in records}), 1)}
    questions = {qid: index for index, qid in enumerate(sorted({q for _, qs, _ in records for q in qs}), 1)}
    raw_skills = sorted({s for q in questions for s in question_skills.get(q + 1, [])})
    skills = {skill: index for index, skill in enumerate(raw_skills, 1)}
    output.mkdir(parents=True, exist_ok=True)
    maps = output / "map"
    maps.mkdir(exist_ok=True)

    with (output / "user_group_sequences.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for learner, qids, answers in records:
            concepts = []
            for qid in qids:
                values = [skills[s] for s in question_skills.get(qid + 1, []) if s in skills]
                concepts.append(str(values[0]) if len(values) == 1 else "[" + ",".join(map(str, values)) + "]")
            handle.write(f"{learners[learner]}\n")
            handle.write(",".join(str(questions[q]) for q in qids) + "\n")
            handle.write(";".join(concepts) + "\n")
            handle.write(",".join(map(str, answers)) + "\n")

    def write_map(name: str, header: list[str], values: dict) -> None:
        with (maps / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(values.items())

    write_map("user_id_map.csv", ["anonymous_user_id", "internal_user_id"], learners)
    write_map("problem_id_map.csv", ["source_problem_id", "internal_problem_id"], questions)
    write_map("skill_name_map.csv", ["skill_name", "skill_id"], skills)
    lengths = [len(qids) for _, qids, _ in records]
    stats = {"num_users": len(learners), "num_problems": len(questions), "num_skills": len(skills),
             "total_interactions": sum(lengths), "min_sequence_length": min(lengths),
             "max_sequence_length": max(lengths), "mean_sequence_length": sum(lengths) / len(lengths)}
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def split(root: Path | None = None, seed: int = 42, ratios: tuple[float, float, float] = (.7, .1, .2)) -> None:
    root = root or project_root()
    if abs(sum(ratios) - 1.0) > 1e-9 or any(value <= 0 for value in ratios):
        raise ValueError("Split ratios must be positive and sum to 1")
    source = root / "data" / "processed" / DATASET / "user_group_sequences.txt"
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) % 4:
        raise ValueError(f"Processed sequence file is not composed of four-line records: {source}")
    records = [lines[index:index + 4] for index in range(0, len(lines), 4)]
    random.Random(seed).shuffle(records)
    train_end = int(len(records) * ratios[0])
    val_end = train_end + int(len(records) * ratios[1])
    groups = (("train", records[:train_end]), ("val", records[train_end:val_end]), ("test", records[val_end:]))
    output = root / "data" / "splits" / DATASET
    output.mkdir(parents=True, exist_ok=True)
    for name, values in groups:
        text = "\n".join(line for record in values for line in record) + "\n"
        (output / f"{name}_sequences.txt").write_text(text, encoding="utf-8")
    (output / "split_stats.json").write_text(json.dumps({name: len(v) for name, v in groups}, indent=2) + "\n", encoding="utf-8")


def run_script(script: str, *arguments: str, root: Path | None = None) -> None:
    root = root or project_root()
    command = [sys.executable, str(root / script), *arguments]
    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    subprocess.run(command, cwd=root, env=environment, check=True)


def prepare(root: Path | None = None, with_graph: bool = False) -> None:
    root = root or project_root()
    stats = preprocess(root)
    print(f"Preprocessed {stats['num_users']} learners and {stats['total_interactions']} interactions.")
    split(root)
    print("Created deterministic train/validation/test splits.")
    if with_graph:
        run_script("src/process_data/2build_graph/build_prereq_graph.py", "--dataset", DATASET, root=root)
    arguments = ["--input-root", "data/splits", "--output-root", "data/instructions", "--include-datasets", DATASET]
    if with_graph:
        arguments += ["--use-graph-context", "--knowledge-graph-root", "data/knowledge_graph"]
    run_script("src/process_data/4seq2text/instruction_temp.py", *arguments, root=root)
    print("Instruction datasets are ready under data/instructions.")
