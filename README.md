# LUCoS: Latent Unsupervised Context Selection for Tabular Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.27254-b31b1b.svg)](https://arxiv.org/abs/2605.27254)

LUCoS is a label-free context selection framework for Tabular Foundation Models
(TFMs) in low-label tabular classification. It addresses the cold-start setting:
before any labels are available, choose which training instances should be sent
to an oracle for annotation and then used as the in-context support set of a
supervised TFM such as TabPFN.

The main idea is that distances in raw tabular space are often unreliable because
features can be heterogeneous, differently scaled, sparse, missing, categorical,
or nonlinearly interacting. LUCoS replaces this raw geometry with the latent
geometry induced by an unsupervised prior-fitted clustering model. In the paper
implementation, each unlabeled training instance is embedded with the
TabClustPFN PIN Encoder, and K-Medoids selects real representative instances in
that latent space.


## Method Summary

Given an unlabeled training set `D = {x_i}_{i=1}^N` and a labeling budget `K`,
LUCoS performs:

1. **Embed training data only**: compute `z_i = phi(x_i)` using an unsupervised
   PFN encoder. The implementation uses the TabClustPFN PIN Encoder and obtains
   512-dimensional row representations.
2. **Select representative medoids**: run K-Medoids on `{z_i}` with the number
   of medoids fixed to `K`. Medoids are real training instances, so they can be
   sent to an annotator.
3. **Query labels once**: annotate only the selected subset.
4. **Predict in original space**: use the selected labeled rows as the context
   set for a supervised TFM. The paper uses TabPFN-2.5 for downstream
   classification.

Selection is label-free, test-blind, instance-faithful, and single-round. Test
instances are not used during embedding, selection, or subset construction.

## Main Results Reported in the Paper

The paper evaluates LUCoS on 67 OpenML-CC18 classification datasets after
filtering to tasks with at most 10 classes, matching TabPFN-2.5 constraints.
Budgets scale with the number of classes:

```text
K in {1C, 2C, 4C, 8C, 16C, 32C}
```

where `C` is the number of classes.

Key findings:

- TabPFN is strong in few-shot regimes but highly sensitive to the chosen
  context instances.
- A supervised SSMA oracle exposes large headroom for context selection,
  especially at `1C`.
- Original-space K-Medoids helps at the smallest budgets but falls below random
  on many datasets as the budget grows.
- LUCoS ranks first across all six budgets under mean AUC and also transfers
  across ACC and macro-F1 rankings.
- The gain decomposes into two mechanisms: at `1C`, coverage from K-Medoids is
  the main contributor; from `4C` onward, the TabClustPFN latent representation
  becomes the decisive component.
- Robustness checks in the paper include leave-one-dataset-out analysis,
  progressive random dataset removal, and cross-metric rank correlation.

## Repository Layout

```text
.
|--- README.md
|--- requirements.txt
|--- setup.py
|--- datasets/
|   |--- OpenML-CC18.xlsx
|   `--- tabarena-v0.1.xlsx
|--- scripts/
|   |--- setup_external_libs.sh
|   `--- external_libs/
|--- libs/
|   |--- TabClustPFN/
|   |--- RDSS/
|   |--- zcore/
`--- src/lucos/
    |--- unsupervised_context_selection/
         |--- run_experiment_context_selection.py
         |--- (rest of python code)
    |--- SSMA/
         |--- run_ssma_selection.py
         |--- (rest of python code)
    `--- utils/
```

Important generated folders:

- `checkpoints/`: embeddings, selector outputs, evaluator logits, and SSMA
  curves.
- `results/results_LUCoS_<exp_id>/`: raw CSVs, aggregated CSVs, plots, and logs.

These generated folders can become large and are not required to understand the
source code.

## Installation

The project is developed for Python 3.10. A local Conda environment under 
`./env` is recommended.
The project must be installed as a Python package so that all imports work correctly.

```bash
conda create --prefix ./env python=3.10
conda activate ./env
python -m pip install -e .
```

`requirements.txt` mirrors the PyPI dependencies declared in `pyproject.toml`
for environments that prefer requirements-file based installation.

### External Libraries

The repository uses local submodules for methods that are not consumed as normal
PyPI dependencies in this project:

- `libs/TabClustPFN`
- `libs/RDSS`
- `libs/zcore`


Install the supported external libraries with:

```bash
chmod +x scripts/setup_external_libs.sh
scripts/setup_external_libs.sh
```

The script initializes the pinned commits for `RDSS`, `TabClustPFN`, and
`zcore`, applies local packaging files from `scripts/external_libs/`, and
installs those three libraries with `pip install --no-build-isolation`.

### TabClustPFN Checkpoint


The pretrained TabClustPFN checkpoint is not downloaded automatically. Download the pretrained model checkpoint manually from Google Drive:

📥 **[Download Checkpoint](https://drive.google.com/file/d/1GzZBgVXwOB8y9JM89xa7ln-la6v2sUO3/view?usp=sharing)**

 and place it
at:

```text
libs/TabClustPFN/checkpoints/step-10000.ckpt
```
The default LUCoS encoder reads that path unless
`encoder.params.checkpoint_path` is overridden in the YAML configuration.


See [official TabClustPFN repository](https://github.com/Tianqi-Zhao/TabClustPFN) for more details.



### TabPFN Access

Experiments using `TabPFNClassifier` may require accepting the TabPFN license
and authenticating with PriorLabs the first time the model is downloaded. Set
the `TABPFN_TOKEN` environment variable with:

```bash
export TABPFN_TOKEN=...
```

TabPFN and TabICL model downloads also use Hugging Face Hub. Set `HF_TOKEN` to
avoid unauthenticated Hub requests and to access gated model files when needed:

```bash
export HF_TOKEN=...
```

## Reproducing the Paper Experiments

All commands below assume the repository root as the working directory and an
activated environment.

### Main LUCoS Experiment

Runs the label-free comparison from the paper: random selection, original-space
selectors, TabClustPFN-space selectors, and downstream TFM evaluation.

Run a single fold:

```bash
python src/lucos/unsupervised_context_selection/run_experiment_context_selection.py \
  config_exp2_LUCoS.yaml \
  --folds "[0]" \
  --exp_id openml
```


Raw outputs are written to:

```text
results/results_LUCoS_openml/raw/results_LUCoS_openml_f<fold>.csv
```

### Aggregation and Plots

After raw fold results exist:

```bash
cd src/lucos/unsupervised_context_selection/agg_results_and_plot
python aggregate_results.py
python create_plots.py
python create_more_plots_and_figures.py
```

Aggregation defaults are currently hard-coded for `EXPERIMENT_ID = "openml"` in
`aggregate_results.py`.
















### TabPFN Context Sensitivity and SSMA Oracle

The SSMA experiment estimates supervised oracle-like headroom and is not a
deployable cold-start method because it uses labels and validation feedback.

Use the provided SSMA checkpoints if available, then evaluate the fold reported
in the paper with:

```bash
python src/lucos/unsupervised_context_selection/run_experiment_context_selection.py \
  config_exp1_SSMA.yaml \
  --folds "[0]" \
  --exp_id openml
```

To recompute SSMA searches, run:

```bash
python src/lucos/SSMA/run_ssma_selection.py
```

This is computationally expensive. The paper reports fold 0 only for SSMA.

## Configuration Files

Main configurations live in:

```text
src/lucos/unsupervised_context_selection/config/
```

- `config_exp2_LUCoS.yaml`: main LUCoS benchmark.
- `config_exp1_SSMA.yaml`: context sensitivity experiment using balanced random
  selection and SSMA selections.

SSMA-specific configuration lives in:

```text
src/lucos/SSMA/config_ssma.yaml
```

The most important experimental knobs are:

- `benchmark.suitename`: usually `OpenML-CC18`.
- `benchmark.max_classes`: `10` in the paper.
- `experiment.subsampler_nclasses_multipliers`: `[1, 2, 4, 8, 16, 32]`.
- `instance_selection`: representation spaces and selectors.
- `original_space_evaluation`: downstream evaluators.

## Data

The code downloads OpenML tasks through the OpenML Python API and caches them
under `datasets/openml/`. The repository includes Excel metadata snapshots for:

- `datasets/OpenML-CC18.xlsx`

For the paper benchmark, only classification datasets with at most ten classes
are used, yielding 67 OpenML-CC18 datasets.

## Outputs and Checkpoints

The experiment runner is restart-friendly:

- Existing result keys are detected and skipped unless recomputation is enabled.
- Selector outputs are cached as masks under `checkpoints/Undersampling/`.
- Encoder outputs are cached under `checkpoints/encoders/`.
- Evaluator logits can be checkpointed under `checkpoints/evaluators/`.
- Raw result CSV files are backed up with `.bak` files to save them from file corruption while write/read.

K-Medoids can optionally cache full pairwise distance matrices. This can save
time but may consume substantial disk space for large datasets.

## Statistical Analysis

The paper compares methods at the dataset level after averaging runs and folds.
The analysis includes:

- AUC as the primary metric, with ACC and macro-F1 as supporting metrics.
- Mean performance and mean rank across datasets.
- Friedman tests and Nemenyi critical-distance comparisons.
- Paired Wilcoxon signed-rank tests against random selection with
  Benjamini-Hochberg correction.
- Mann-Whitney U tests for Rescue/Boost decomposition analyses.
- Leave-one-dataset-out and progressive dataset-removal robustness checks.

The plotting and aggregation scripts implement these analyses for the result
tables and figures.

## Known Limitations

These limitations come from the paper and current codebase:

- Classification only; regression support is left for future work.
- Main benchmark is limited to tasks with at most 10 classes because of
  TabPFN-2.5 constraints.
- K-Medoids requires pairwise distances and can be expensive on large pools.
- The default representation is TabClustPFN; other tabular embedding models are
  not part of the main claim.

## Citation

```bibtex
@article{ipas2026lucos,
  title   = {LUCoS: Latent Unsupervised Context Selection for Tabular Foundation Models},
  author  = {Ipas, Oroel and Gomez-Trenado, Guillermo and Romero-Zaliz, Rocio and Triguero, Isaac},
  journal = {arXiv preprint arXiv:2605.27254},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
