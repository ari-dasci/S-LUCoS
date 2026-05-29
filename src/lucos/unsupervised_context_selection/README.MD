# Unsupervised Context Selection Experiments

This package contains the experiment code for LUCoS and related baselines. It
implements the paper setting where a small subset of training instances must be
selected before any labels are available, then evaluated as the in-context
support set of a supervised tabular foundation model.

## Scientific Setting

The experiment is decomposed into two independent components:

- **Representation space**: maps raw training rows into the space where
  geometric selection is performed.
- **Instance selector**: chooses `K` representative rows in that space.

LUCoS is the combination:

```text
TabClustPFNEncoder + KMedoidsUnderSampler(init="k-medoids++", metric="euclidean")
```

The selected indices always refer back to the original training rows. Downstream
evaluators receive the original features and labels of the selected rows, not
the latent embeddings.

## Files

```text
unsupervised_context_selection/
|--- config/
|   |--- config_exp1_SSMA.yaml
|   `--- config_exp2_LUCoS.yaml
|--- agg_results_and_plot/
|--- encoders.py
|--- evaluators.py
|--- instance_selection.py
`--- run_experiment_context_selection.py
```

## Main Runner

`run_experiment_context_selection.py` is the main script used for the paper
experiments.

```bash
python src/lucos/unsupervised_context_selection/run_experiment_context_selection.py \
  config_exp2_LUCoS.yaml \
  --folds "[0]" \
  --exp_id openml
```

CLI arguments:

- `config`: YAML file name inside `config/`.
- `--folds`: fold list, for example `"[0]"`, `"[0,1,2]"`, or `"0,1,2"`.
- `--exp_id`: optional override for `config.experiment.exp_id`.

If `experiment.device` is `default`, the script uses `cuda:0` when available
and `cpu` otherwise.

## Main Configurations

### `config_exp2_LUCoS.yaml`

Main LUCoS benchmark:

- Benchmark: OpenML-CC18.
- Filtering: classification only, at most 10 classes.
- Budgets: `K = C * m` for `m in [1, 2, 4, 8, 16, 32]`.
- Spaces:
  - `TabClustPFNEncoder`, producing `TabClustPFNSpace`.
  - `OriginalSpaceEncoder`, producing `OriginalSpace`.
- Selectors:
  - `RandomUnderSamplerUnsupervised`.
  - `KMedoidsUnderSampler` with k-medoids++ and euclidean/cosine distances.
  - `RDSSUnderSampler`.
  - `ZCoreUnderSampler`.
- Evaluators currently listed in the YAML:
  - `TabPFNv2_5`.
  - `TabPFNv3`.
  - `TabICLv2`.

The paper's headline LUCoS result corresponds to TabClustPFN-space K-Medoids
with euclidean distance and TabPFN-2.5 downstream evaluation.

### `config_exp1_SSMA.yaml`

Context sensitivity experiment:

- Original-space balanced random selection.
- SSMA selections loaded from SSMA checkpoints.
- Evaluators: KNN, XGBoost, and TabPFN-2.5.
- Intended for fold 0, matching the paper's supervised-oracle analysis.

## Encoders

Defined in `encoders.py`.

- `OriginalSpaceEncoder`: passes raw features through as a numeric array.
- `TabClustPFNEncoder`: loads
  `libs/TabClustPFN/checkpoints/step-10000.ckpt` by default, preprocesses the
  training split, and extracts row representations using TabClustPFN's column
  embedder and row interactor. The resulting latent space is used only for
  selection.
- `DummyEncoder`: placeholder implementation.

Embedding outputs are cached under:

```text
checkpoints/encoders/<EncoderName>/<suite>/<dataset>_fold<fold>.npy
```

## Instance Selectors

Defined in `instance_selection.py`.

- `RandomUnderSampler`: supports `Unsupervised`, `Balanced`, and
  `Min1PerClass` modes.
- `KMedoidsUnderSampler`: computes pairwise distances, runs
  `sklearn_extra.cluster.KMedoids`, and returns medoid indices. The paper uses
  k-medoids++ initialization and euclidean distance for the main LUCoS method.
- `RDSSUnderSampler`: wraps the local `RDSS` submodule.
- `ZCoreUnderSampler`: computes ZCore scores through the local `zcore`
  submodule and selects the highest-scoring rows.
- `SSMAUnderSampler`: loads previously computed supervised SSMA chromosomes and
  converts them into selected indices for evaluation.

Selector masks are cached under:

```text
checkpoints/Undersampling/<SpaceName>/<SamplerName>/<suite>/<dataset>_fold<fold>.json
```

## Evaluators

Defined in `evaluators.py`.

The base evaluator computes:

- `AUC`
- `ACC`
- `F1`
- `BalACC`

The paper uses AUC as the primary metric and reports ACC and macro-F1 as
supporting metrics. Multiclass AUC is computed one-vs-rest.

Important evaluator classes include:

- `TabPFNv2_5`
- `TabPFNv3`
- `TabICLv2`
- `XGBoost`
- `KNN`
- `LR`

Evaluator logits can be cached under `checkpoints/evaluators/` when enabled.

## Execution Flow

For each dataset and requested fold, the runner:

1. Loads the OpenML task and fold split.
2. Builds the requested budgets from percentages and/or class multipliers.
3. Evaluates each downstream model on the full original training set as an
   `OriginalSpace / All` reference.
4. For every configured representation space:
   - encodes the training split;
   - runs each configured selector for each budget and seed;
   - maps selected indices back to original training rows;
   - evaluates downstream models on the selected context.
5. Appends result rows to the fold CSV.

Result columns are:

```text
dataset, fold, instance_selection_space, instance_selection_algorithm,
n_selected_samples, random_state, evaluator, metric, value
```

## Output Location

For `--exp_id openml`, fold-specific raw outputs are:

```text
results/results_LUCoS_openml/raw/results_LUCoS_openml_f0.csv
results/results_LUCoS_openml/raw/results_LUCoS_openml_f1.csv
...
results/results_LUCoS_openml/raw/results_LUCoS_openml_f9.csv
```

Logs are written to:

```text
results/logs/
```

## Reproducibility Notes

- A deterministic global seed of `42` is set in the main runner.
- Stochastic selectors use `base_seed + run_idx`.
- The main LUCoS config uses five runs for random and K-Medoids selectors, and
  one run for RDSS/ZCore.
- Existing result keys are skipped, so interrupted experiments can usually be
  resumed.
- For publication-quality aggregation, run all configured folds before calling
  the aggregation scripts.

## Practical Notes

- TabClustPFN requires its checkpoint file before `TabClustPFNEncoder` can run.
- K-Medoids may be slow or memory-heavy because it works with pairwise
  distances.
- The selector is intentionally label-free; warnings about missing classes in a
  selected subset are expected in cold-start runs.
