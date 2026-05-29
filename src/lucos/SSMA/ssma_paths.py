import os
from lucos.utils.paths import ssma_results_folder


"""
I do not really like how this code is structured.
It is a bit confusing that paths are global variables but there is a function that modifies them.
"""


original_space_root = ssma_results_folder.parent
random_selection_results_folder = original_space_root / 'RandomUnderSampler'
kmedoids_selection_results_folder = original_space_root / 'KMedoidsUnderSampler'

os.makedirs(ssma_results_folder, exist_ok=True)
os.makedirs(random_selection_results_folder, exist_ok=True)
os.makedirs(kmedoids_selection_results_folder, exist_ok=True)
ssma_run = 1
suitename = 'OpenML-CC18'

def _build_ssma_paths(run, suitename):
    run_folder = ssma_results_folder / suitename / f'ssma_run{run}'
    return {
        'ssma_results_folder_with_run': run_folder,
        'ssma_fit_histories_folder': run_folder / 'ssma_fit_histories',
        'ssma_selection_results_folder': run_folder / 'ssma_selection_results',
        'ssma_best_chromosomes_curve_folder': run_folder / 'ssma_best_chromosomes_curve',
        'ssma_plots_folder': run_folder / 'ssma_plots',
    }


_paths = _build_ssma_paths(ssma_run, suitename)
ssma_results_folder_with_run = _paths['ssma_results_folder_with_run']
ssma_fit_histories_folder = _paths['ssma_fit_histories_folder']
ssma_selection_results_folder = _paths['ssma_selection_results_folder']
ssma_best_chromosomes_curve_folder = _paths['ssma_best_chromosomes_curve_folder']
ssma_plots_folder = _paths['ssma_plots_folder']

def modify_paths_for_ssma_run_and_suitename(run, suitename):
    # This function is a bit dirty because it modifies global variables
    global ssma_run, ssma_results_folder_with_run, ssma_fit_histories_folder, ssma_selection_results_folder, ssma_best_chromosomes_curve_folder, ssma_plots_folder
    ssma_run = run
    paths = _build_ssma_paths(ssma_run, suitename)
    ssma_results_folder_with_run = paths['ssma_results_folder_with_run']
    ssma_fit_histories_folder = paths['ssma_fit_histories_folder']
    ssma_selection_results_folder = paths['ssma_selection_results_folder']
    ssma_best_chromosomes_curve_folder = paths['ssma_best_chromosomes_curve_folder']
    ssma_plots_folder = paths['ssma_plots_folder']
    os.makedirs(ssma_plots_folder, exist_ok=True)
    os.makedirs(ssma_selection_results_folder, exist_ok=True)
    os.makedirs(ssma_fit_histories_folder, exist_ok=True)
    os.makedirs(ssma_best_chromosomes_curve_folder, exist_ok=True)
