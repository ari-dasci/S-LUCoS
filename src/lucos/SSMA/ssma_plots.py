import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

import lucos.SSMA.ssma_paths as ssma_paths
from lucos.SSMA.ssma_utils import get_chromosome_in_curve_smaller_or_equal_to_n, get_sizes_we_want_to_compare

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)



def plot_ssma_selection_in_axes(ax, X_train_2d, y_train, X_val_2d, y_val, X_test_2d, y_test, selected_indices, title, x_label="", y_label=""):
    from matplotlib.lines import Line2D

    # Colors per class (simple mapping using matplotlib colormap)
    unique_classes = np.unique(y_train)
    cmap = plt.get_cmap("tab10")
    class_to_color = {
        c: cmap(i % 10) for i, c in enumerate(sorted(unique_classes))
    }

    ax.clear()
    # Plot training points (non-selected) and selected separately
    for cls in unique_classes:
        color = class_to_color[cls]

        # Training data for this class
        mask_train = y_train == cls
        # Selected indices within this class
        selected_in_cls = np.intersect1d(np.where(mask_train)[0], selected_indices)
        non_selected_in_cls = np.setdiff1d(np.where(mask_train)[0], selected_indices)

        # Non-selected training points (small circles)
        ax.scatter(
            X_train_2d[non_selected_in_cls, 0],
            X_train_2d[non_selected_in_cls, 1],
            c=[color],
            marker="o",
            s=15,
            alpha=0.4,
            edgecolors="none",
            label=None,
        )
        # Val data for this class (squares)
        mask_val = y_val == cls
        ax.scatter(
            X_val_2d[mask_val, 0],
            X_val_2d[mask_val, 1],
            c=[color],
            marker="s",
            s=15,
            alpha=0.6,
            linewidths=1.0,
            label=None,
        )
        # Test data for this class (crosses)
        mask_test = y_test == cls
        ax.scatter(
            X_test_2d[mask_test, 0],
            X_test_2d[mask_test, 1],
            c=[color],
            marker="x",
            s=30,
            alpha=0.8,
            linewidths=1.0,
            label=None,
        )

    # Draw the stars on top
    for cls in unique_classes:
        color = class_to_color[cls]
        # Training data for this class
        mask_train = y_train == cls
        # Selected indices within this class
        selected_in_cls = np.intersect1d(np.where(mask_train)[0], selected_indices)
        non_selected_in_cls = np.setdiff1d(np.where(mask_train)[0], selected_indices)

        # Selected training instances (large stars, bold edge)
        ax.scatter(
            X_train_2d[selected_in_cls, 0],
            X_train_2d[selected_in_cls, 1],
            c=[color],
            marker="*",
            s=120,
            alpha=0.95,
            edgecolors="black",
            linewidths=1.0,
            label=None,
        )

    # Legend (train/test & selection markers)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Train (non-selected)",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=6,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Train (selected by SSMA)",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=12,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Val",
            markerfacecolor="gray",
            markeredgecolor="none",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label="Test",
            markersize=6,
        ),
    ]

    ax.legend(handles=legend_elements, loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()



def get_pca_and_tsne_embeddings(X_train, X_val, X_test):
    t = time()
    # Fit PCA and t-SNE on full data (train + test) to obtain consistent projections
    X_full = np.vstack([X_train, X_val, X_test])
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    embeddings = {}

    # PCA embedding
    pca = PCA(n_components=2, random_state=42)
    X_full_2d_pca = pca.fit_transform(X_full)
    embeddings["pca"] = (
        X_full_2d_pca[:n_train], # train
        X_full_2d_pca[n_train:n_train + n_val], # val
        X_full_2d_pca[n_train + n_val:], # test
    )
    print(f"Computed PCA in {time() - t:.2f} seconds")
    t = time()
    # t-SNE embedding
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_full_2d_tsne = tsne.fit_transform(X_full)
    embeddings["tsne"] = (
        X_full_2d_tsne[:n_train], # train
        X_full_2d_tsne[n_train:n_train + n_val], # val
        X_full_2d_tsne[n_train + n_val:], # test
    )
    print(f"Computed t-SNE in {time() - t:.2f} seconds")
    return embeddings


def plot_ssma_selection(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, best_chromosomes, space_name="original"):
    """
    TODO: allow empty X_val, y_val (no validation set)
    Creates and saves:
    - A GIF showing the selection evolution over different sizes in best_chromosomes
    - A static plot with subplots for each size in best_chromosomes
    We calculate 2D embeddings only once
    """
    logger.debug(f"Plotting selection for dataset {dataset_name}")

    # Get 2D embeddings
    embeddings = get_pca_and_tsne_embeddings(X_train, X_val, X_test)
  
    n_classes = len(np.unique(y_train))
    # Get chroms with sizes in n_classes * 2^k
    sizes_we_want_to_compare = get_sizes_we_want_to_compare(X_train.shape[0], n_classes)
    chroms_to_plot = [get_chromosome_in_curve_smaller_or_equal_to_n(best_chromosomes, size) for size in sizes_we_want_to_compare]
    chroms_to_plot = {c.count_active(): c for c in chroms_to_plot} # remove duplicates
    chroms_to_plot = dict(sorted(chroms_to_plot.items())) # sort
    
    # Prepare output folder
    os.makedirs(ssma_paths.ssma_plots_folder, exist_ok=True)

    #### GIF ####
    gif_path = ssma_paths.ssma_plots_folder / f"{dataset_name}_{space_name}.gif"
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    writer = animation.PillowWriter(fps=1)
    with writer.saving(fig, gif_path, dpi=100):
        for size, chrom in chroms_to_plot.items():
            for i, (emb_name, (X_train_2d, X_val_2d, X_test_2d)) in enumerate(embeddings.items()):
                selected_indices = np.where(chrom.genes == 1)[0]
                title = f"{dataset_name}_{space_name} {emb_name} \n SSMA selection (size={size}, {size / X_train.shape[0] * 100:.1f}%)"
                plot_ssma_selection_in_axes(axs[i], X_train_2d, y_train, X_val_2d, y_val, X_test_2d, y_test, selected_indices, title=title, y_label=emb_name)
            writer.grab_frame()
        plt.close(fig)
        logger.debug(f"Saved SSMA selection GIF to {gif_path}")


    #### STATIC PLOTS ####
    plot_path = ssma_paths.ssma_plots_folder / f"{dataset_name}_{space_name}.png"
    fig, axs = plt.subplots(2, len(chroms_to_plot), figsize=(5*len(chroms_to_plot), 10))
    # Create one plot per embedding type and size
    for i, (emb_name, (X_train_2d, X_val_2d, X_test_2d)) in enumerate(embeddings.items()):
        for j, (size, chrom) in enumerate(chroms_to_plot.items()):
            selected_indices = np.where(chrom.genes == 1)[0]
            title = f"size={size} ({size / X_train.shape[0] * 100:.1f}%)"
            title = f"{dataset_name}_{space_name} {emb_name}\n" + title if len(chroms_to_plot)//2 == j else title
            y_label = emb_name if j == 0 else ""
            plot_ssma_selection_in_axes(axs[i, j], X_train_2d, y_train, X_val_2d, y_val, X_test_2d, y_test, selected_indices, title=title, y_label=y_label)

    plt.savefig(plot_path)
    plt.close(fig)
    logger.debug(f"Saved SSMA selection plot to {plot_path}")




def plot_selection_results(dataset_name, X_train_size, n_classes, results_ssma ={}, results_random={}, results_kmedoids={}):
    """
    TODO: this function should receive the pandas dataframe I am building to aggregate all results
    Plots the comparison of test AUC between random selection and SSMA selection methods.
    One line for random selection (with mean and std shaded area) and
    two lines for SSMA selection (validation and test).
    """
    logger.debug(f"Plotting selection results for dataset {dataset_name}")
    os.makedirs(ssma_paths.ssma_plots_folder, exist_ok=True)
    plt.figure()

    if 'test_auc_mean' in results_random:
        plt.plot(list(results_random['test_auc_mean'].keys()), list(results_random['test_auc_mean'].values()), color='blue', label='Random Selection', marker='o')
        plt.fill_between(list(results_random['test_auc_mean'].keys()), 
                        [m - s for m, s in zip(results_random['test_auc_mean'].values(), results_random['test_auc_std'].values())],
                        [m + s for m, s in zip(results_random['test_auc_mean'].values(), results_random['test_auc_std'].values())],
                        color='blue', alpha=0.2)
    
    if 'test_auc_mean' in results_kmedoids:
        plt.plot(list(results_kmedoids['test_auc_mean'].keys()), list(results_kmedoids['test_auc_mean'].values()), color='red', label='K-Medoids Selection (Test)', marker='o')
        plt.fill_between(list(results_kmedoids['test_auc_mean'].keys()), 
                        [m - s for m, s in zip(results_kmedoids['test_auc_mean'].values(), results_kmedoids['test_auc_std'].values())],
                        [m + s for m, s in zip(results_kmedoids['test_auc_mean'].values(), results_kmedoids['test_auc_std'].values())],
                        color='red', alpha=0.2)
        
    if 'val_auc_values' in results_ssma:
        # Scatter the individual points from each fold, if any
        plt.scatter(list(results_ssma['val_auc_values']['size']), list(results_ssma['val_auc_values']['auc']), color='orange', marker='o', alpha=0.5)
    plt.plot(list(results_ssma['val_auc_mean'].keys()), list(results_ssma['val_auc_mean'].values()), color='orange', label='SSMA Selection (Val)')
    plt.fill_between(list(results_ssma['val_auc_mean'].keys()),
                    [m - s for m, s in zip(results_ssma['val_auc_mean'].values(), results_ssma['val_auc_std'].values())],
                    [m + s for m, s in zip(results_ssma['val_auc_mean'].values(), results_ssma['val_auc_std'].values())],
                    color='orange', alpha=0.2)
    
    if 'test_auc_values' in results_ssma:
        # Scatter the individual points from each fold, if any
        plt.scatter(list(results_ssma['test_auc_values']['size']), list(results_ssma['test_auc_values']['auc']), color='green', marker='o', alpha=0.5)
    plt.plot(list(results_ssma['test_auc_mean'].keys()), list(results_ssma['test_auc_mean'].values()), color='green', label='SSMA Selection (Test)')
    plt.fill_between(list(results_ssma['test_auc_mean'].keys()), 
                    [m - s for m, s in zip(results_ssma['test_auc_mean'].values(), results_ssma['test_auc_std'].values())],
                    [m + s for m, s in zip(results_ssma['test_auc_mean'].values(), results_ssma['test_auc_std'].values())],
                    color='green', alpha=0.2)

    plt.xscale('log')
    xmin, xmax = plt.xlim()
    plt.xlim(xmax, xmin)
    xticks_reduction_percentages = [1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01] # Hardcoded because these look good in the logarithmic plot
    xticks_selection_sizes = [int(X_train_size * p) for p in xticks_reduction_percentages]
    x_ticks = [f"{s}\n{round(p*100)}%" for s,p in zip(xticks_selection_sizes, xticks_reduction_percentages)]
    plt.xticks(xticks_selection_sizes + [n_classes], x_ticks + [f'{n_classes}'])
    plt.xlabel(f'Size of Training Data Used')
    plt.ylabel('Test AUC')
    plt.title(f'{dataset_name} - SSMA vs Random Selection (TabPFN) \n x_train size={X_train_size} \n(n_classes={n_classes})')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(ssma_paths.ssma_plots_folder / f'{dataset_name}.png')
    plt.close()
    logger.debug(f"Saved selection results plot to {ssma_paths.ssma_plots_folder / f'{dataset_name}.png'}")


def plot_ssma_fit_histories(dataset_name_and_fold, ssma_fit_histories, X_train_size):
    """
    Plots the evolution of fitness, reduction, and AUC during SSMA training.
    
    For each initial_selection_size in the fit_histories, creates a separate subplot with:
    - Fitness: orange line for validation, green line for test
    - Reduction: orange line for validation, green line for test
    - AUC: orange line for validation, green line for test
    - Alpha: red line
    
    Args:
        dataset_name_and_fold: Name of the dataset and fold for the plot title
        ssma_fit_histories: Dictionary mapping initial_selection_size -> fit_history dict
    """
    logger.debug(f"Plotting SSMA fit histories for dataset {dataset_name_and_fold}")
    os.makedirs(ssma_paths.ssma_plots_folder, exist_ok=True)
    
    # remove empty fit histories
    ssma_fit_histories = {k: v for k, v in ssma_fit_histories.items() if len(v['population']) > 0}

    n_initial_selection_size = len(ssma_fit_histories)
    fig, axes = plt.subplots(n_initial_selection_size, 1, figsize=(12, 5 * n_initial_selection_size))
    
    # Handle case where there's only one reduction size (axes is not an array)
    if n_initial_selection_size == 1:
        axes = [axes]
    
    for idx, (initial_selection_size, fit_history) in enumerate(sorted(ssma_fit_histories.items())):
        ax = axes[idx]
        
        steps = np.arange(len(fit_history['population']))   
        best_chrom_of_population = [pop[0] for pop in fit_history['population']]
        for key, color, label in [('fitness',          'black',       'Fitness'), 
                                  ('reduction',        'deepskyblue', 'Reduction'), 
                                  ('log_reduction',    'blue',        'Log Reduction'), 
                                  ('linear_reduction', 'cyan',        'Linear Reduction'), 
                                  ('val_auc',          'orange',      'Val AUC'), 
                                  ('test_auc',         'green',       'Test AUC'), 
                                  ('alpha',            'red',         'Alpha')]:
            if len(fit_history['population']) == 0 or key not in fit_history['population'][0][0].__dict__:
                continue
            
             # Extract values per step
            populations = fit_history['population']
            values_per_step = [[chrom.__getattribute__(key) for chrom in pop] for pop in populations]
            best_chrom_metric = [chrom.__getattribute__(key) for chrom in best_chrom_of_population]
            mean_values = [np.mean(v) for v in values_per_step]
            std_values = [np.std(v) for v in values_per_step]
            ax.plot(steps, mean_values, color=color, label=label, linewidth=2, linestyle='-')
            ax.plot(steps, best_chrom_metric, color=color, linewidth=2, linestyle='--')
            # fill between mean ± std
            ax.fill_between(steps, 
                            np.array(mean_values) - np.array(std_values),
                            np.array(mean_values) + np.array(std_values),
                            color=color, alpha=0.2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, 1)
        ax.set_title(f'{dataset_name_and_fold} - Population metrics (dashed line: best chrom) - Initial Selection Size: {initial_selection_size} ({round(initial_selection_size/X_train_size*100)}%)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ssma_paths.ssma_plots_folder / f'{dataset_name_and_fold}_fit_histories.png', dpi=150)
    logger.debug(f"Saved fit histories plot to {ssma_paths.ssma_plots_folder / f'{dataset_name_and_fold}_fit_histories.png'}")
    plt.close()
