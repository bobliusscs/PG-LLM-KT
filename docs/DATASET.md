# HNU-SYS-2023 Dataset Card

## Summary

HNU-SYS-2023 contains chronological learner-question interactions collected for educational knowledge-tracing research. This public release provides response correctness and an anonymous question-to-skill incidence matrix.

## Files and schema

`data/raw/hnu_sys2023/HNU_SYS_2023.txt` has one learner per line:

```text
anonymous_learner_id<TAB>question_id correctness,question_id correctness,...
```

Correctness is binary (`0` or `1`). `question2skill.csv` is a sparse binary matrix: the first column is a question ID and each remaining column is an anonymous skill ID.

## Privacy and exclusions

Learner identifiers were replaced with stable pseudonyms. Question wording, answers, skill names, source knowledge graphs, and direct personal identifiers are excluded. IDs are meaningful only inside this release. Users must not attempt re-identification.

## Intended use and limitations

The dataset is intended for non-commercial research in knowledge tracing and educational modeling. It is not representative of all learners or learning environments and must not be used for high-stakes decisions about individuals. The repository does not provide question content, so semantic analyses of questions or skills are outside its scope.

## License

Code is MIT licensed. The dataset is provided for non-commercial research use; redistribution and downstream publication should preserve this dataset card and privacy restrictions. A dedicated data license should be reviewed by the data owner before a formal release.
