# Aggregation, Statistical Analysis, and Plots

This folder converts raw experiment CSV files into dataset-level aggregates,
tables, and figures for the LUCoS paper.

Run aggregation before plotting. Plotting scripts read the aggregated CSVs
produced by `aggregate_results.py`.

## Files

```text
agg_results_and_plot/
|--- aggregate_results.py
|--- create_plots.py
`--- create_more_plots_and_figures.py
```

## Expected Inputs

For the default `EXPERIMENT_ID = "openml"`, raw inputs are expected under:

```text
results/results_LUCoS_openml/raw/
```

Fold-specific files should follow:

```text
results_LUCoS_openml_f0.csv
results_LUCoS_openml_f1.csv
...
results_LUCoS_openml_f9.csv
```

Each raw row should contain:

```text
dataset, fold, instance_selection_space, instance_selection_algorithm,
n_selected_samples, random_state, evaluator, metric, value
```

## Basic Workflow

From this folder:

```bash
python aggregate_results.py
python create_plots.py
python create_more_plots_and_figures.py
```

Outputs are written under:

```text
results/results_LUCoS_openml/agg/
results/results_LUCoS_openml/plots/
```

## Aggregation Logic

`aggregate_results.py` performs the paper's dataset-level aggregation:

1. Discover raw fold files for the configured experiment id.
2. Validate and normalize result columns.
3. Load benchmark metadata from `datasets/OpenML-CC18.xlsx`.
4. Filter methods, spaces, evaluators, folds, and metrics according to the
   constants declared near the top of the script.
5. Check expected random seeds and warn about missing runs.
6. Average multiple runs per dataset/fold/method/budget.
7. Average folds per dataset.
8. Export wide aggregated tables used by plotting and statistical routines.

Important constants:

- `EXPERIMENT_ID`: defaults to `openml`.
- `C_MULTIPLIERS`: `[1, 2, 4, 8, 16, 32]`.
- `FOLDS_TO_AGGREGATE_LUCoS`: all ten OpenML-CC18 folds.
- `FOLDS_TO_AGGREGATE_SSMA`: fold 0 only.
- `ALL_METHODS_SPACES_TO_AGGREGATE_LUCoS`: methods included in LUCoS tables.
- `EVALUATOR_TO_AGGREGATE_LUCoS`: downstream evaluators included in LUCoS
  aggregation.
- `METRICS_TO_AGGREGATE_PER_EVALUATOR`: metrics exported per evaluator.

For a different experiment id, edit `EXPERIMENT_ID` in `aggregate_results.py`
before aggregating.

## Paper Analyses Implemented Here

The scripts support the following paper analyses:

- Mean score curves across budgets.
- Improvability curves for the SSMA context sensitivity experiment.
- LUCoS versus original-space K-Medoids paired comparisons.
- Mean-rank tables and critical-distance style comparisons.
- AUC, ACC, and macro-F1 summaries.
- Selector-versus-representation gain decomposition.
- Rescue/Boost partition analysis.
- Robustness checks such as leave-one-dataset-out and progressive dataset
  removal.

The exact set of figures generated depends on which raw methods/evaluators were
run and which constants are enabled in the scripts.

## Statistical Tests

The plotting code uses dataset-level values after averaging over folds and
seeds. Tests and summaries include:

- Friedman tests over method ranks.
- Nemenyi post-hoc comparisons through `scikit-posthocs`.
- Paired Wilcoxon signed-rank tests for paired method comparisons.
- Mann-Whitney U tests for Rescue/Boost group separation.
- Spearman correlation between rank vectors across metrics.

## Method Names

The scripts expect the method names produced by the runner, for example:

```text
RandomUnderSamplerUnsupervised
RandomUnderSamplerBalanced
KMedoidsUnderSamplerK-medoids++Euclidean
KMedoidsUnderSamplerK-medoids++Cosine
RDSSUnderSampler
ZCoreUnderSampler
SSMAUnderSampler
```

Representation spaces include:

```text
OriginalSpace
TabClustPFNSpace
```

If a selector or encoder name changes in code, update the constants and style
dictionaries in the aggregation/plotting scripts.

## Practical Notes

- Missing raw folds or missing seeds are reported as warnings; inspect them
  before using generated tables in a paper.
- The current defaults focus on the method subset used in the final LUCoS paper.
- Plot styling is embedded in `create_plots.py` and related scripts.
