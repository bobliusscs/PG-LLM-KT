import csv
from pathlib import Path

from pgllmkt.pipeline import preprocess, split


def fixture(root: Path) -> None:
    raw = root / "data/raw/hnu_sys2023"
    raw.mkdir(parents=True)
    (raw / "HNU_SYS_2023.txt").write_text(
        "user_a\t0 1,1 0\nuser_b\t1 1,0 0\nuser_c\t0 1,1 1\n", encoding="utf-8"
    )
    with (raw / "question2skill.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows([["question_id", "1", "2"], ["1", "1", ""], ["2", "", "1"]])


def test_preprocess_and_split(tmp_path: Path) -> None:
    fixture(tmp_path)
    stats = preprocess(tmp_path)
    assert stats["num_users"] == 3
    assert stats["total_interactions"] == 6
    sequence = tmp_path / "data/processed/hnu_sys2023/user_group_sequences.txt"
    assert len(sequence.read_text(encoding="utf-8").splitlines()) == 12
    split(tmp_path, ratios=(1 / 3, 1 / 3, 1 / 3))
    split_dir = tmp_path / "data/splits/hnu_sys2023"
    assert all((split_dir / f"{name}_sequences.txt").is_file() for name in ("train", "val", "test"))
