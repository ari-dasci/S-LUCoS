import os
import json
import logging
import numpy as np
from lucos.SSMA.Chromosome import Chromosome
import lucos.SSMA.ssma_paths as ssma_paths

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)


###############################################################
#          Functions to load and save results                 #
###############################################################


def convert_to_json(obj):
    if isinstance(obj, Chromosome):
        return {"__Chromosome__": True, **obj.to_dict()}
    elif isinstance(obj, dict):
        return {k: convert_to_json(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json(x) for x in obj]
    else:
        return obj


def convert_from_json(obj):
    if isinstance(obj, dict):
        if "__Chromosome__" in obj:
            return Chromosome.from_dict({k:v for k,v in obj.items() if k != "__Chromosome__"})
        else:
            return {_json_key_to_python(k): convert_from_json(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [convert_from_json(x) for x in obj]
    else:
        return obj


def _json_key_to_python(key):
    if not isinstance(key, str):
        return key

    if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
        return int(key)

    try:
        if any(ch in key for ch in [".", "e", "E"]):
            float_key = float(key)
            return int(float_key) if float_key.is_integer() else float_key
    except ValueError:
        pass

    return key


def save_json_atomic(file_path, data):
    # If the process is interrupted, the file may be left corrupted
    # Save to a temporary file and then rename it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_file_path = str(file_path) + ".tmp"
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json(data), f, ensure_ascii=False)
    os.replace(temp_file_path, file_path)


def load_json_file(file_path, default=None):
    if not os.path.exists(file_path):
        if default is not None:
            logger.debug(f"JSON file not found: {file_path}. Returning provided default value.")
            return default
        else:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return convert_from_json(json.load(f))


def load_aggregated_ssma_fit_histories(dataset_name):
    """
    Aggregate populations step-wise across folds.
    """

    # Aggregate fit_histories across folds
    folds_found = []
    folds_not_found = []
    ssma_fit_histories_aggregated = {}
    for fold in range(10):

        # Load fit histories for this fold
        fit_histories_path = ssma_paths.ssma_fit_histories_folder / f'{dataset_name}_fold{fold}_fit_histories.json'

        if not os.path.exists(fit_histories_path):
            folds_not_found.append(fold)
            continue

        folds_found.append(fold)
        fit_histories = load_json_file(fit_histories_path)

        # For each initial reduction size
        for init_size, history in fit_histories.items():
            if init_size not in ssma_fit_histories_aggregated:
                ssma_fit_histories_aggregated[init_size] = {'step': 0,
                                                        'steps_without_improvement': 0,
                                                        'population': []
                                                        }
            # Aggregate populations step-wise
            for step_idx, pop in enumerate(history['population']):
                if len(ssma_fit_histories_aggregated[init_size]['population']) <= step_idx:
                    ssma_fit_histories_aggregated[init_size]['population'].append(pop)
                else:
                    ssma_fit_histories_aggregated[init_size]['population'][step_idx].extend(pop)
    logger.debug(f"Aggregated fit histories from folds {folds_found}, skipped folds {folds_not_found}")

    return ssma_fit_histories_aggregated


def nadaraya_watson_estimator(x_eval, x_data, y_data, bandwidth=10.0):
    """
    Nadaraya-Watson estimator for kernel regression.
    Parameters:
    - x_eval: points where the regression is evaluated (1D array). Example: np.linspace(min, max, num_points)
    - x_data: input data points (1D array)
    - y_data: output data points (1D array)
    - bandwidth: kernel bandwidth (float)
    Returns:
    - means: estimated y values at x_eval (1D array)
    - stds: estimated standard deviations at x_eval (1D array)
    """
    means = []
    stds = []
    
    for x0 in x_eval:
        # Gaussian kernel: computes weights based on the distance to x0
        weights = np.exp(-0.5 * ((x_data - x0) / bandwidth) ** 2)
        
        # Weighted mean (Weighted Average)
        weighted_mean = np.sum(weights * y_data) / np.sum(weights)
        
        # Weighted standard deviation (for the dispersion band)
        weighted_var = np.sum(weights * (y_data - weighted_mean)**2) / np.sum(weights)
        
        means.append(weighted_mean)
        stds.append(np.sqrt(weighted_var))
        
    return np.array(means), np.array(stds)


def load_and_aggregate_ssma_results(dataset_name, x_train_size):
    """
    Aggregate SSMA results across folds using Nadaraya-Watson estimator.
    Parameters:
    - ssma_results: dict mapping fold -> {'val_auc': {size: auc}, 'test_auc': {size: auc}}
    Returns:
    - aggregated: dict with keys:
        'val_auc_values' and 'test_auc_values': lists of individual size and auc values across folds
        'val_auc_mean' and 'val_auc_std': smoothed mean and std of validation AUC
        'test_auc_mean' and 'test_auc_std': smoothed mean and std of test AUC
    """ 
    aggregated = {
        'val_auc_values': {'size': [], 'auc': []},
        'test_auc_values': {'size': [], 'auc': []}
    }
    folds_found = []
    folds_not_found = []
    # Aggregate val and test AUC across folds
    for fold in range(10):

        # Load SSMA results for this fold
        ssma_results_path = ssma_paths.ssma_selection_results_folder / f'{dataset_name}_fold{fold}_ssma_results.json'

        if not os.path.exists(ssma_results_path):
            folds_not_found.append(fold)
            continue
    
        folds_found.append(fold)
        results = load_json_file(ssma_results_path)

        for sel_size, val_auc in results['val_auc_mean'].items():
            aggregated['val_auc_values']['size'].append(sel_size)
            aggregated['val_auc_values']['auc'].append(val_auc)
        for sel_size, test_auc in results['test_auc_mean'].items():
            aggregated['test_auc_values']['size'].append(sel_size)
            aggregated['test_auc_values']['auc'].append(test_auc)
    
    logger.debug(f"Aggregated SSMA results from folds {folds_found}, skipped folds {folds_not_found}")
    xmin, xmax = min(aggregated['val_auc_values']['size']), max(aggregated['val_auc_values']['size'])
    x_grid = np.linspace(xmin, xmax, 200)
    bandwidth = np.log10(x_train_size) / 100.0  # Adjust the bandwidth according to the size range.

    # Convert x_grid to logarithmic scale so smoothing works better
    x_grid = np.log10(x_grid)
    sizes_val_auc = np.log10(np.array(aggregated['val_auc_values']['size']))
    sizes_test_auc = np.log10(np.array(aggregated['test_auc_values']['size']))
    val_means, val_stds = nadaraya_watson_estimator(x_grid, sizes_val_auc, np.array(aggregated['val_auc_values']['auc']), bandwidth=bandwidth)
    test_means, test_stds = nadaraya_watson_estimator(x_grid, sizes_test_auc, np.array(aggregated['test_auc_values']['auc']), bandwidth=bandwidth)

    # Revert to the original scale
    x_grid = (10**x_grid).astype(int)

    for key, means, stds in [('val_auc', val_means, val_stds), ('test_auc', test_means, test_stds)]:
        aggregated[key+'_mean'] = {}
        aggregated[key+'_std'] = {}
        for xi, mean, std in zip(x_grid, means, stds):
            aggregated[key+'_mean'][xi] = mean
            aggregated[key+'_std'][xi] = std

    return aggregated


def save_results(dataset_name_and_fold, ssma_results={}, random_results={}, kmedoids_results={}):
    """
    Saves results to disk as json files.
    """
    if random_results:
        random_results_path = ssma_paths.random_selection_results_folder / f'{dataset_name_and_fold}_random_results.json'
        save_json_atomic(random_results_path, random_results)
        logger.debug(f"Saved random results to {random_results_path}")

    if ssma_results:
        ssma_results_path = ssma_paths.ssma_selection_results_folder / f'{dataset_name_and_fold}_ssma_results.json'
        save_json_atomic(ssma_results_path, ssma_results)
        logger.debug(f"Saved SSMA results to {ssma_results_path}")

    if kmedoids_results:
        kmedoids_results_path = ssma_paths.kmedoids_selection_results_folder / f'{dataset_name_and_fold}_kmedoids_results.json'
        save_json_atomic(kmedoids_results_path, kmedoids_results)
        logger.debug(f"Saved KMedoids results to {kmedoids_results_path}")


def load_ssma_results(dataset_name_and_fold):
    """
    Loads results from disk if they exist.
    dataset_name_and_fold: dataset name (and fold if applicable)
    Returns:
    - ssma_results or {} if not found.
    """
    ssma_results_path = ssma_paths.ssma_selection_results_folder / f'{dataset_name_and_fold}_ssma_results.json'
    ssma_results = load_json_file(ssma_results_path, default={})

    if ssma_results == {}:
        logger.debug(f"No SSMA results found for dataset {dataset_name_and_fold}.")
    else:
        logger.debug(f"Loaded SSMA results for dataset {dataset_name_and_fold} from disk.")
    
    return ssma_results
        

def load_random_selection_results(dataset_name):
    random_results_path = ssma_paths.random_selection_results_folder / f'{dataset_name}_random_results.json'
    default={'test_auc_mean': {}, 'test_auc_std': {}, 'test_auc_values': {}}
    results = load_json_file(random_results_path, default=default)
    if results['test_auc_mean']:
        logger.debug(f"Loaded cached random selection results for sizes: {sorted(results['test_auc_values'].keys(), reverse=True)}")
    return results


def load_kmedoids_selection_results(dataset_name):
    kmedoids_results_path = ssma_paths.kmedoids_selection_results_folder / f'{dataset_name}_kmedoids_results.json'
    default={'test_auc_mean': {}, 'test_auc_std': {}, 'test_auc_values': {}}
    results = load_json_file(kmedoids_results_path, default=default)
    if results['test_auc_mean']:
        logger.debug(f"Loaded cached KMedoids selection results for sizes: {sorted(results['test_auc_values'].keys(), reverse=True)}")
    else:
        logger.debug(f"No KMedoids results found for dataset {dataset_name}")
    return results
    

def load_best_chromosomes_curve(dataset, run=None):
    target_folder = ssma_paths.ssma_best_chromosomes_curve_folder
    if run is not None:
        target_folder = ssma_paths.ssma_results_folder / dataset.suitename / f'ssma_run{run}' / 'ssma_best_chromosomes_curve'
    best_chromosomes_curve_path = target_folder / f'{dataset.name_and_fold}_best_chromosomes_curve.json'
    return load_json_file(best_chromosomes_curve_path, default={})


######################################################################
#     Functions to compare instance selection curves                 #
######################################################################

def get_chromosome_in_curve_smaller_or_equal_to_n(best_chromosomes_curve, n):
    # Iterate from largest to smallest and return the first one that is <= n
    for chrom_size in sorted(best_chromosomes_curve.keys(), reverse=True):
        if chrom_size <= n:
            return best_chromosomes_curve[chrom_size]
    return None


def find_smaller_or_equeal_subset_auc(random_results, best_chromosomes_curve, n):
    """
    The random subset size closest to n
       - random_subset_size: <= n.
    The ssma subset size closest to random_subset_size
       - ssma_subset_size: <= random_subset_size. 
    """
    random_results = random_results['test_auc_mean']

    for subset_size in sorted(random_results.keys()):
        if subset_size <= n:
            random_subset_auc = random_results[subset_size]
            random_subset_size = subset_size

    chrom = get_chromosome_in_curve_smaller_or_equal_to_n(best_chromosomes_curve, random_subset_size)

    return chrom.test_auc, random_subset_auc, chrom.size, random_subset_size


def get_sizes_we_want_to_compare(x_train_size, n_classes):
    """Get the sizes we want to compare based on the training set size and number of classes.
    
    Args:
        x_train_size (int): The size of the training set.
        n_classes (int): The number of classes in the dataset.
        max_ssma_reduction_size (int): The maximum size for SSMA reduction.
    
    Returns:
        list: A list of sizes to compare.
    """
    max_ssma_reduction_size = 250
    max_ssma_reduction_percent = 0.25
    x_train_size_25 = np.ceil(max_ssma_reduction_percent * x_train_size).astype(int).item()
    max_size_to_consider = min(x_train_size_25, max_ssma_reduction_size)
    n_classes_powers_of_2 = [n_classes * 2**i for i in range(7)]
    size_we_want_to_compare = [size for size in n_classes_powers_of_2 if size <= max_size_to_consider]
    return size_we_want_to_compare
