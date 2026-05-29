# SSMA Supervised Context Search

This folder contains the adapted Steady-State Memetic Algorithm (SSMA) used in
the paper as a supervised oracle-like diagnostic for TabPFN context sensitivity.
It is not part of the deployable LUCoS method because it uses labels and
validation feedback during search.

## Role in the Paper

The paper first asks whether TabPFN performance depends strongly on which
context instances are selected. To estimate the achievable headroom, SSMA
searches for subsets that maximize validation AUC when used as TabPFN-2.5
context.

The resulting SSMA curve is compared against balanced random selection. In the
paper, SSMA improves random selection most strongly at the smallest budgets,
with the mean AUC gap shrinking as `K` grows from `1C` to `32C`. This motivates
the main LUCoS question: can some of that headroom be recovered without labels?

## Files

```text
SSMA/
|--- Chromosome.py
|--- SSMA.py
|--- config_ssma.yaml
|--- run_ssma_selection.py
|--- ssma_paths.py
|--- ssma_plots.py
`--- ssma_utils.py
```

## Algorithm Overview

Each candidate solution is a binary chromosome over the training candidate pool:

- `1`: instance selected for the context subset.
- `0`: instance not selected.

For each OpenML fold, `run_ssma_selection.py` splits the official training split
again into:

- candidate pool: 80% of the fold training data;
- validation set: 20% of the fold training data.

SSMA optimizes selected subsets on validation AUC while encouraging smaller
subsets. The held-out test split is used only for final evaluation and plotting.

Main components:

- **Initialization**: random chromosomes with at least one instance per class.
- **Parent selection**: binary tournament.
- **Crossover**: uniform crossover controlled by `p_cross`.
- **Mutation**: constrained bit flips with class-coverage repair.
- **Local search**: stochastic first-improvement additions/removals.
- **Replacement**: elitist steady-state replacement.
- **Stopping**: fixed iteration budget or patience without improvement.

## Fitness Function

`Chromosome.evaluate()` computes:

```text
fitness = alpha * validation_auc + (1 - alpha) * reduction
```

where:

```text
reduction = 0.5 * linear_reduction + 0.5 * log_reduction
```

The logarithmic component keeps pressure toward very small subsets. Infeasible
chromosomes, such as those missing a class, receive zero fitness.

The current implementation uses TabPFN-2.5 with `n_estimators=1` inside the
search to reduce cost, then evaluates selected chromosomes separately with the
final classifier settings.

## Configuration

`config_ssma.yaml` controls the search:

```yaml
benchmark:
  suitename: OpenML-CC18
  max_classes: 10
  only_classification: true

ssma:
  alphas: [0.5]
  pop_size: 20
  p_cross: 0.7
  p_mut: 0.01
  local_search_attempts: 10
  patience: 70
  steps_per_alpha: 500

experiment:
  sizes_we_want_to_reduce_to: [250, 180, 100, 50, 20, 10]
  ssma_run: 1
  folds_to_run: [0]
  val_size_split: 0.2
  tabpfn_device: cuda
  tabpfn_n_estimators: 8
```

The helper `get_sizes_we_want_to_reduce_to_in_dataset()` augments the configured
sizes with class-scaled budgets `C * 2^i` and filters them to feasible ranges.

## Running SSMA

Run all datasets from the SSMA config:

```bash
python src/lucos/SSMA/run_ssma_selection.py
```

Run one dataset by OpenML dataset name:

```bash
python src/lucos/SSMA/run_ssma_selection.py electricity
```

This search is expensive. The paper uses SSMA only on fold 0. The source comment
notes that one large dataset/fold can take many hours.

## Outputs

SSMA outputs are organized by run and benchmark suite through `ssma_paths.py`.
The important artifacts are:

- `ssma_best_chromosomes_curve/`: best non-dominated chromosomes by selected
  subset size.
- `ssma_fit_histories/`: population histories and resume checkpoints.
- result summaries used by plotting and by `SSMAUnderSampler`.

`SSMAUnderSampler` in
`src/lucos/unsupervised_context_selection/instance_selection.py` loads these
curves and returns the chromosome with size smaller than or equal to the
requested budget when evaluating `config_exp1_SSMA.yaml`.

## Reproducibility Notes

- SSMA uses labels and validation AUC, so it should never be reported as
  label-free context selection.
- The algorithm is stochastic and can resume from saved histories.
- Chromosomes enforce class coverage during initialization and repair, unlike
  LUCoS and other cold-start selectors that do not know labels.

## Relationship to LUCoS

SSMA answers a diagnostic question: "Is there substantial performance headroom if
we could optimize the context with labels?" LUCoS addresses the deployable
cold-start question: "Can we select a good context without labels by using a
better representation geometry?"

