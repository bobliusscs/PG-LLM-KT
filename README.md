# PG-LLM-KT
PG-LLM-KT: A Unified Prerequisite-Guided Framework for Knowledge Tracing with Plug-and-play Large Language Models

## Overview

This project presents an innovative approach to Knowledge Tracing (KT) that integrates Large Language Models (LLMs) to enhance student learning modeling and prediction. By combining traditional knowledge tracing methodologies with state-of-the-art LLM techniques, our system provides a comprehensive framework for educational technology research and applications.

The project incorporates multiple knowledge tracing models, including AKT (Attentive Knowledge Tracing), DKT (Deep Knowledge Tracing), GKT (Graph-based Knowledge Tracing), and MGKT (Multi-Skill Graph-based Knowledge Tracing), enhanced with knowledge graph construction and data processing pipelines.

## Key Features

1. **Multi-Model Support**: Integration of various knowledge tracing models for comparative analysis
2. **Knowledge Graph Enhancement**: Construction of prerequisite relationship graphs between concepts
3. **LLM Integration**: Leveraging Large Language Models for knowledge tracing and prediction
4. **Comprehensive Pipeline**: End-to-end workflow from data preprocessing to model evaluation
5. **Scalable Architecture**: Modular design supporting diverse educational datasets

## Technical Approach

Our methodology combines traditional deep learning-based knowledge tracing with graph neural networks and large language models. The system includes:

- **Data Processing Pipeline**: Standardized preprocessing for multiple educational datasets
- **Graph Construction**: Building knowledge graphs based on student learning patterns
- **Sequence-to-Text Conversion**: Transforming learning sequences to text format suitable for LLM training
- **Model Training Framework**: Support for various knowledge tracing architectures
- **Evaluation System**: Comprehensive metrics for performance assessment

## Datasets Supported

The project supports multiple educational datasets:

- ASSISTments 2009 & 2012: Student interaction data from the ASSISTments online learning platform
- EdNet KT1: Knowledge tracing version of the EdNet dataset
- HNU SYS 2023: Hainan Normal University system dataset
- Junyi: Online learning platform data
- KDD Cup 2010: Educational data mining competition dataset

## Evaluation Metrics

Models are evaluated using standard metrics:

- AUC (Area Under Curve)
- Accuracy
- F1 Score
- Precision and Recall
- RMSE (Root Mean Square Error)

## Repository Status

**Important Note**: The complete source code for this project will be made publicly available upon acceptance of the associated research paper. This repository currently contains documentation and structural information for the project.

## Architecture

```
PG_LLM_KT/
├── dataSet/                 # Dataset directory
│   ├── 1raw/               # Raw datasets
│   ├── 2processed/         # Preprocessed datasets
│   ├── 3knowledge_graph/   # Knowledge graphs
│   └── 4data_split/        # Dataset splits
├── models/                 # Model files
│   ├── cache/              # Model cache
│   └── lora_binary_classification/  # LoRA classification models
└── src/                    # Source code (to be released)
    ├── model_structure/    # Model implementations
    ├── process_data/       # Data processing pipeline
    ├── train_scripts/      # Training scripts
    └── test_scripts/       # Testing scripts
```

## Applications

This research contributes to:

- Personalized learning systems
- Intelligent tutoring systems
- Educational data mining
- Student performance prediction
- Adaptive learning environments

## Citation

When the paper is accepted, please cite this work as:

```
[To be updated with citation upon publication]
```

## License

This project will be released under [license type] upon publication.

---

*This repository will be updated with complete code and documentation following the paper acceptance.*
