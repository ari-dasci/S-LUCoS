import logging
import argparse
from pathlib import Path
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split    

from lucos.utils.dataset import Benchmark
from lucos.utils.evaluation import auc_score
from lucos.utils.log_utils import setup_logging
from lucos.utils.config_utils import load_yaml_config
from lucos.SSMA.ssma_utils import load_ssma_results, load_aggregated_ssma_fit_histories, load_and_aggregate_ssma_results, save_results
from lucos.SSMA.ssma_plots import plot_ssma_fit_histories, plot_selection_results
from lucos.SSMA.ssma_paths import modify_paths_for_ssma_run_and_suitename
from lucos.SSMA.SSMA import SSMA_TabPFN

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)


def get_sizes_we_want_to_reduce_to_in_dataset(dataset, sizes_we_want_to_reduce_to):
    n_classes_powers_of_2 = [dataset.num_classes * 2**i for i in range(7)] # 1, 2, 4, 8, 16, 32, 64 times the number of classes
    # Maximum size with which we start the reduction: 25% of train or 250 (the largest size we want to reduce), whichever is smaller
    max_initial_reduction_size = min(sizes_we_want_to_reduce_to[0], int(dataset.x_train_size * 0.25))
    sizes_we_want_to_reduce_to_in_dataset = [max_initial_reduction_size] + sizes_we_want_to_reduce_to + n_classes_powers_of_2
    sizes_we_want_to_reduce_to_in_dataset = [s for s in sizes_we_want_to_reduce_to_in_dataset if dataset.num_classes <= s <= max_initial_reduction_size] # filter in [n_classes, max_initial_reduction_size]
    sizes_we_want_to_reduce_to_in_dataset.sort(reverse=True)
    return sizes_we_want_to_reduce_to_in_dataset


def load_config(config_name_or_path: str):
    return load_yaml_config(
        config_name_or_path,
        "lucos.SSMA",
        repo_relative_dir=Path("src") / "lucos" / "SSMA",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "requested_dataset",
        nargs="?",
        default=None,
        help="Name of the dataset to run. If omitted, all datasets are run.",
    )
    return parser.parse_args()



def evaluate_ssma(clf, X_train, y_train, X_test, y_test, best_chromosomes):
    """
    Evaluates the best chromosomes found by SSMA.
    best_chromosomes: dict {size: Chromosome}
    Returns a dict with mean and std AUCs for validation and test.
    The stds are 0 because each chromosome is evaluated only once here, but it is useful to keep the same structure as in random evaluation.
    results = {
        'val_auc_values': {'size': [], 'auc': []},
        'test_auc_values': {'size': [], 'auc': []},
        'val_auc_mean': {size: val_auc}, # the one calculated during SSMA fit
        'val_auc_std': {size: 0.0},
        'test_auc_mean': {size: test_auc}, # the one recalculated here with the final classifier, which may differ from chrom.test_auc if the classifier is different
        'test_auc_std': {size: 0.0}
    }
    """
    logger.debug(f"Starting SSMA evaluation with {len(best_chromosomes)} best chromosomes")
    results = {'val_auc_values': {'size': [], 'auc': []}, 
               'test_auc_values': {'size': [], 'auc': []},
               'val_auc_mean': {}, 
               'val_auc_std': {}, 
               'test_auc_mean': {}, 
               'test_auc_std': {}}
    for size, chrom in best_chromosomes.items():
        selected_indices = np.where(chrom.genes == 1)[0]
        X_sel = np.array(X_train)[selected_indices]
        y_sel = np.array(y_train)[selected_indices]
        clf.fit(X_sel, y_sel)
        # We are not using extend_logits() here because chromosomes already ensure they have at least one instance of each class
        logits_test = clf.predict_proba(X_test)
        preds = np.argmax(logits_test, axis=1)
        test_auc = auc_score(y_test, logits_test)
        test_acc = accuracy_score(y_test, preds)

        # val_auc is taken from the value calculated during SSMA fit
        # But test_auc is recalculated with the final classifier
        results['val_auc_values']['size'].append(size)
        results['val_auc_values']['auc'].append(chrom.val_auc)
        results['test_auc_values']['size'].append(size)
        results['test_auc_values']['auc'].append(test_auc)
        results['val_auc_mean'][size] = chrom.val_auc
        results['val_auc_std'][size] = 0.0
        results['test_auc_mean'][size] = test_auc
        results['test_auc_std'][size] = 0.0
        logger.debug(f"Evaluated SSMA chromosome: val_auc={chrom.val_auc:.3f} | test_auc={test_auc:.3f} | val_acc={chrom.val_acc:.3f} | test_acc={test_acc:.3f} | reduction={chrom.reduction:.3f} | size={size} | ir={chrom.ir:.3f}")
    
    logger.debug(f"SSMA evaluation complete.")
    logger.debug(f"SSMA selection results: {results}")
    return results


def main():
    args = parse_args()
    requested_dataset = args.requested_dataset

    config = load_config("config_ssma.yaml")

    ssma_run = int(config.experiment.ssma_run)

    benchmark = Benchmark(**config.benchmark)
    modify_paths_for_ssma_run_and_suitename(ssma_run, config.benchmark.suitename)
    final_clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5,
        ignore_pretraining_limits=True,
        device=str(config.experiment.tabpfn_device),
        n_estimators=int(config.experiment.tabpfn_n_estimators),
    )

    for i, dataset in enumerate(benchmark.get_datasets()):
        if requested_dataset is not None and dataset.name != requested_dataset:
            continue
        sizes_we_want_to_reduce_to_in_dataset = get_sizes_we_want_to_reduce_to_in_dataset(
            dataset,
            list(config.experiment.sizes_we_want_to_reduce_to),
        )

        for fold in config.experiment.folds_to_run:
            try:
                X_train_all, y_train_all, X_test, y_test = dataset.get_data_of_fold(fold, numpy_arrays=True)

                setup_logging(f"ssma_run{ssma_run}/ssma_{dataset.name_and_fold}.log")
                logger.debug("\n" * 5 + "=" * 50 + f"\nProcessing DATASET {i+1}/{benchmark.num_of_filtered_datasets} - {dataset.name_and_fold}\n" + "=" * 50 + "\n" * 2)

                # 2. Split the fold splits and create the validation set
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_all,
                    y_train_all,
                    test_size=float(config.experiment.val_size_split),
                    random_state=int(config.experiment.train_val_split_random_state),
                    shuffle=True,
                )
                logger.debug(f"Dataset {dataset.name_and_fold}: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

                ssma = SSMA_TabPFN(
                    dataset_name=dataset.name_and_fold,
                    sizes_we_want_to_reduce_to=sizes_we_want_to_reduce_to_in_dataset,
                    **config.ssma
                    )
                ssma_results_of_fold = load_ssma_results(dataset.name_and_fold)

                if ssma_results_of_fold == {}:
                    # 3. Run selection
                    ssma.fit(X_train, y_train, X_val, y_val, X_test, y_test)

                    # 5. Final validation with TabPFN
                    logger.debug(f"Evaluating SSMA")
                    ssma_results_of_fold = evaluate_ssma(final_clf, X_train, y_train, X_test, y_test, ssma.best_chromosomes_curve)
                else:
                    logger.debug(f"SSMA results for dataset {dataset.name_and_fold} already exist. Skipping computation.")

                # Create visualizations
                #plot_ssma_selection(dataset.name_and_fold, X_train, y_train, X_val, y_val, X_test, y_test, ssma.best_chromosomes_curve, space_name="OriginalSpace")
                plot_ssma_fit_histories(dataset.name_and_fold, ssma.fit_histories, dataset.x_train_size)
                plot_selection_results(dataset.name_and_fold, dataset.x_train_size, dataset.num_classes, results_ssma=ssma_results_of_fold)
                save_results(dataset.name_and_fold, ssma_results=ssma_results_of_fold)
                logger.debug(f"Completed processing for {dataset.name_and_fold}\n\n")

            except Exception as e:
                logger.exception(f"Error processing dataset {dataset.name_and_fold}: {e}")
                continue

        # Aggregate results across folds
        ssma_fit_histories_aggregated = load_aggregated_ssma_fit_histories(dataset.name)
        ssma_results_aggregated = load_and_aggregate_ssma_results(dataset.name, dataset.x_train_size)

        # Plot aggregated results
        dataset_name_agg = dataset.name + "_aggregated"
        logger.debug(f"Plotting aggregated results for dataset {dataset_name_agg}")
        plot_selection_results(dataset_name_agg, dataset.x_train_size, dataset.num_classes, results_ssma=ssma_results_aggregated)
        logger.debug("Plotting aggregated SSMA fit histories")
        plot_ssma_fit_histories(dataset_name_agg, ssma_fit_histories_aggregated, dataset.x_train_size)
        logger.debug(f"Completed aggregated processing for dataset {dataset_name_agg}")
    
    logger.debug("All datasets processed.")


if __name__ == "__main__":
    main()
