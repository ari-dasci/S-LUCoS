# Shared Utilities

This package contains reusable helpers for dataset loading, preprocessing,
metrics, configuration, paths, logging, and visualization. The experiment
folders depend on these utilities to keep the LUCoS and SSMA runners aligned.

## Files

```text
utils/
|--- config_utils.py
|--- dataset.py
|--- evaluation.py
|--- log_utils.py
|--- paths.py
|--- preprocessing.py
`--- visualization.py
```

## Dataset Handling

`dataset.py` defines:

- `Benchmark`: loads OpenML benchmark metadata, filters tasks, and yields
  `Dataset` objects.
- `Dataset`: wraps an OpenML task, resolves classification/regression type,
  loads features and labels, tracks numerical/categorical/binary columns, and
  exposes fold splits.

For the LUCoS paper, `Benchmark` is configured with:

```yaml
suitename: OpenML-CC18
only_classification: true
max_classes: 10
impute_nan: true
```

The resulting paper benchmark contains 67 OpenML-CC18 classification datasets.

OpenML metadata snapshots are stored as Excel files under `datasets/`; raw
OpenML downloads are cached under `datasets/openml/`.

## Paths

`paths.py` centralizes project paths:

```text
project_folder
results_folder
log_folder
datasets_folder
ckpt_folder
tabpfn_ckpt_folder
undersampling_ckpt_folder
ssma_results_folder
```

Generated artifacts are expected under:

- `results/`: logs, raw result CSVs, aggregate tables, and plots.
- `datasets/`: benchmark metadata and OpenML cache.
- `checkpoints/`: encoder outputs, selector masks, evaluator logits, and SSMA
  artifacts.

## Preprocessing and Metrics

`preprocessing.py` provides shared transformations such as one-hot encoding used
by original-space geometric selectors.

`evaluation.py` contains metric helpers, including AUC and imbalance-ratio
utilities used by evaluators and SSMA chromosomes.

The main experiment evaluator computes AUC, ACC, macro-F1, and balanced
accuracy. The paper reports AUC as primary and ACC/macro-F1 as supporting
metrics.

## Configuration and Logging

`config_utils.py` resolves import paths declared in YAML configs and parses fold
arguments passed through the CLI.

`log_utils.py` configures the shared `lucos` logger used across experiment
runners. Logs are written under `results/logs/`.

## Notes for New Experiments

- Keep new generated paths under the centralized folders in `paths.py`.
- Prefer adding new selectors or encoders to the experiment packages, then
  referencing them by `target:` in YAML configs.
- If a new benchmark is added, extend `Benchmark`/`Dataset` rather than
  hard-coding dataset logic in runners.
- Preserve the test-blind LUCoS protocol: representation fitting and selection
  should use training data only unless an experiment is explicitly marked as
  transductive or exploratory.

