# PG-LLM-KT

PG-LLM-KT is a reproducible research toolkit for prerequisite-graph enhanced large-language-model knowledge tracing. It includes the anonymized HNU-SYS-2023 interaction dataset, deterministic data preparation, prerequisite graph construction, instruction generation, and the original LoRA prediction architecture.

## Repository layout

```text
PG-LLM-KT/
|-- data/raw/hnu_sys2023/       # Versioned, anonymized public data
|-- pgllmkt/                    # Stable CLI and reproducible pipeline
|-- scripts/                    # One-command wrappers for Windows and Unix
|-- src/process_data/           # Research implementations
|-- src/model/                  # LoRA and prediction-head implementation
|-- src/train_scripts/          # Training orchestration
|-- tests/                      # Fast pipeline tests
`-- docs/DATASET.md             # Dataset card and responsible-use notes
```

Generated data, graphs, model checkpoints, and experiment logs are ignored by Git.

## Quick start: data preparation

Requirements: Python 3.10-3.12, approximately 1 GB free disk space, and no GPU.

### Windows PowerShell

```powershell
git clone https://github.com/bobliusscs/PG-LLM-KT.git
cd PG-LLM-KT
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
.\scripts\prepare.ps1
```

### Linux or macOS

```bash
git clone https://github.com/bobliusscs/PG-LLM-KT.git
cd PG-LLM-KT
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
sh scripts/prepare.sh
```

The one-command script checks the dataset, preprocesses it, creates learner-level 70/10/20 splits with seed 42, and generates LLM instruction JSON files. Outputs are written below `data/processed`, `data/splits`, and `data/instructions`.

To include prerequisite graph construction and graph context in instructions:

```powershell
.\scripts\prepare.ps1 --with-graph
```

Graph construction is CPU intensive. Its thresholds can be adjusted with `PREREQ_MIN_SUPPORT`, `PREREQ_MAX_STUDENTS`, and the other `PREREQ_*` environment variables documented in `build_prereq_graph.py`.

## CLI reference

After `pip install -e .`, use either `pgllmkt` or `python -m pgllmkt.cli`:

```text
pgllmkt doctor              Validate Python, repository paths, and raw data
pgllmkt preprocess          Convert raw interactions to the four-line KT format
pgllmkt split               Produce deterministic learner-level splits
pgllmkt prepare             Run preprocessing, splitting, and instruction generation
pgllmkt prepare --with-graph  Also construct and inject prerequisite graphs
```

Every command resolves paths from the repository location, so it can be run from any working directory.

## Training environment

Training is intentionally separate from the CPU preparation command. Use a recent NVIDIA GPU, matching CUDA drivers, and enough VRAM for the chosen base model. Install the PyTorch build recommended for your CUDA platform first, then install the training extras:

```powershell
pip install -r requirements-train.txt
```

For production experiments, pin the resolved environment after validation:

```powershell
pip freeze > requirements-lock.txt
```

Prepare instruction data, download a Hugging Face-compatible causal language model, and inspect the model entry point:

```powershell
python src/model/loraAndPredictor.py --help
python src/model/loraAndPredictor.py `
  --model_path C:\path\to\base-model `
  --data_path data\instructions\hnu_sys2023\train.json `
  --val_path data\instructions\hnu_sys2023\val.json `
  --output_dir outputs\hnu_sys2023
```

Run `--help` and set batch size, sequence length, LoRA parameters, logging, and loss options for the available hardware. Weights & Biases is imported by the original training implementation; configure it with `wandb login` or use `WANDB_MODE=offline`. Never commit API keys, downloaded models, or checkpoints.

The higher-level `train_unified.py` is retained as research code, but direct use of `loraAndPredictor.py` is the portable public training path. The unified script originated in a multi-dataset internal workflow and should not be treated as the default HNU-SYS-2023 entry point.

## Data contracts and outputs

- Raw responses: one pseudonymous learner per line, followed by a tab and comma-separated `question_id correctness` pairs.
- Processed sequences: four lines per learner: internal learner ID, question IDs, skill IDs, and correctness labels.
- Splits: learners are assigned to exactly one of train, validation, or test; interactions from a learner never cross splits.
- Instructions: JSON arrays containing `system`, `instruction`, `output`, and `dataset_name`.
- Graphs: configuration-specific JSON/CSV edge files under `data/knowledge_graph/hnu_sys2023`.

See [the dataset card](docs/DATASET.md) for provenance, privacy exclusions, intended use, and limitations.

## Development and verification

```powershell
pip install -e ".[dev]"
pytest
python -m compileall -q pgllmkt src
python -m pgllmkt.cli doctor
```

The small test suite validates preprocessing and deterministic learner-level splitting without requiring PyTorch or a GPU. Contributions should follow [CONTRIBUTING.md](CONTRIBUTING.md).

## Troubleshooting

- `Dataset: missing`: confirm both files exist under `data/raw/hnu_sys2023` and run commands inside this clone.
- `ModuleNotFoundError`: activate the virtual environment and run `pip install -e .`; for training, install `requirements-train.txt` too.
- CUDA out of memory: reduce `--batch_size` and `--max_seq_length`, increase gradient accumulation, or choose a smaller base model.
- Garbled console text from a legacy research script: use a UTF-8 terminal (`chcp 65001` on older Windows consoles). The stable `pgllmkt` CLI emits English UTF-8 output.
- Rebuild generated artifacts: delete only the relevant ignored output directory and rerun `pgllmkt prepare`; raw versioned data is never modified.

## License and citation

Code is released under the MIT License. Dataset access is currently limited to non-commercial research as described in the dataset card; the data owner should approve a dedicated data license before formal publication. Cite this repository and the associated paper once its bibliographic record is available.
