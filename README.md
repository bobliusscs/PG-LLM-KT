# PG-LLM-KT

An anonymized, English-documented release of the HNU-SYS-2023 learning dataset and the reproducible processing and modeling code used by PG-LLM-KT.

## Contents

- `data/HNU_SYS_2023.txt`: anonymized learner response sequences. Each line contains a pseudonymous learner ID, a tab, and comma-separated `question_id correctness` pairs (`0` or `1`). Original learner identifiers and personal information are not included.
- `data/question2skill.csv`: question-to-skill incidence matrix. Question and skill identifiers are numeric and intentionally contain no question text or skill names.
- `src/process_data/`: preprocessing, prerequisite-graph construction, dataset splitting, and instruction-template scripts.
- `src/train_scripts/train_ours/train_unified.py`: unified training entry point.
- `src/model/loraAndPredictor.py`: model and prediction modules.

## Reproducibility

Run scripts from the repository root. The scripts accept command-line options and document their expected intermediate files in their module docstrings. A typical pipeline is:

```text
python src/process_data/1data_preprocess/process_hnu_sys2023.py
python src/process_data/2build_graph/build_prereq_graph.py
python src/process_data/3data_split/split_dataset.py
python src/process_data/4seq2text/instruction_temp.py
python src/train_scripts/train_ours/train_unified.py --help
```

Paths and model checkpoints are environment- or argument-dependent; inspect `--help` before running training. Install the dependencies required by the selected script (Python, pandas, PyTorch, Transformers/PEFT, and scikit-learn as applicable).

## Dataset statement

HNU-SYS-2023 is a self-constructed educational interaction dataset. This public release contains only response sequences and a binary question-to-skill matrix. Question wording, answer keys, skill names, knowledge-graph source material, and other potentially identifying metadata are deliberately excluded. Learner identifiers were replaced with stable pseudonyms; no attempt should be made to re-identify learners. Use the data for research and benchmarking, and cite this repository when appropriate.

## License and contact

Unless a separate file states otherwise, code is released under the MIT License. Dataset use is restricted to non-commercial research until the maintainers publish a dedicated data license. Please open an issue for questions or responsible-disclosure requests.
