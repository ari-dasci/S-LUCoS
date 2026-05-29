import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from scipy.stats import friedmanchisquare, wilcoxon

from lucos.unsupervised_context_selection.agg_results_and_plot.aggregate_results import (
    C_MULTIPLIERS, \
    RESULTS_LUCoS_EXPERIMENT, \
    EXPERIMENT_RESULTS_FOLDER, \
    METRICS_TO_AGGREGATE_PER_EVALUATOR, \
    _latex_escape, \
    get_data_from_column, \
    read_aggregated_results
)

logger = logging.getLogger("lucos")


# SSMA: Supervised Selection: RandomUnderSamplerBalanced, SSMAUnderSampler
METHOD_SPACE_EVALUATOR_RENAME_AND_STYLE_SSMA = {
    ("OriginalSpace", "SSMAUnderSampler",           "TabPFNv2_5"): {"color": "black",      "linestyle": ":",  "label": "TabPFNv2.5 (SSMA)"},
    ("OriginalSpace", "RandomUnderSamplerBalanced", "TabPFNv2_5"): {"color": "black",      "linestyle": "-",  "label": "TabPFNv2.5"},
    ("OriginalSpace", "RandomUnderSamplerBalanced", "KNN"):        {"color": "tab:blue",   "linestyle": "-",  "label": "KNN"},
    ("OriginalSpace", "RandomUnderSamplerBalanced", "XGBoost"):    {"color": "tab:orange", "linestyle": "-",  "label": "XGBoost",},
}

# LUCoS: Unsupervised Selection
METHOD_SPACE_RENAME_AND_STYLE_LUCoS = {
    ("OriginalSpace",    "RandomUnderSamplerUnsupervised"):           {"color": "black", "linestyle": "-", "label": "Random Unsupervised"},
    ("OriginalSpace",    "KMedoidsUnderSamplerK-medoids++Euclidean"): {"color": "tab:orange", "linestyle": "-", "label": "Original-Space | KMedoids-Euc"},
    ("OriginalSpace",    "KMedoidsUnderSamplerK-medoids++Cosine"):    {"color": "tab:orange", "linestyle": "--", "label": "Original-Space | KMedoids-Cos"},
    ("TabClustPFNSpace", "KMedoidsUnderSamplerK-medoids++Euclidean"): {"color": "steelblue",  "linestyle": "-", "label": "LUCoS-Euc (our proposal)"},
    ("TabClustPFNSpace", "KMedoidsUnderSamplerK-medoids++Cosine"):    {"color": "steelblue",  "linestyle": "--", "label": "LUCoS-Cos (our proposal)"},
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"] # Times New Roman is not installed, so use DejaVu Serif, which is similar and ships with matplotlib
plt.rcParams["mathtext.fontset"] = "dejavuserif"


DEFAULT_SPACE_COLORS = {
    "OriginalSpace": "tab:orange",
    "TabClustPFNSpace": "steelblue",
}

def build_plot_output_path(section: str, suffix: str) -> Path:
    output_folder = EXPERIMENT_RESULTS_FOLDER / "plots" / str(section)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder / f"{RESULTS_LUCoS_EXPERIMENT}_{suffix}"


def format_plot_output_path(path: str | Path) -> str:
    path = Path(path)
    plots_root = EXPERIMENT_RESULTS_FOLDER / "plots"
    try:
        return f".../plots/{path.resolve().relative_to(plots_root.resolve())}"
    except ValueError:
        path_parts = path.parts
        if "plots" in path_parts:
            plots_idx = path_parts.index("plots")
            return ".../" + "/".join(path_parts[plots_idx:])
        return str(path)


def build_lucos_plot_output_path(
    evaluator_name: str,
    metric_name: str,
    suffix: str,
    lucos_km_metric_name: str | None = None,
) -> Path:
    section = f"LUCoS/{evaluator_name}/{metric_name}"
    if lucos_km_metric_name is not None:
        section = f"{section}/{lucos_km_metric_name}"
    return build_plot_output_path(section, suffix)


def get_lucos_method_styles_for_evaluator(evaluator_name: str) -> dict[tuple[str, str, str], dict]:
    return {
        (space, method, str(evaluator_name)): style.copy()
        for (space, method), style in METHOD_SPACE_RENAME_AND_STYLE_LUCoS.items()
    }


def has_metric_for_evaluator(agg_df: pd.DataFrame, evaluator_name: str, metric_name: str) -> bool:
    return any(
        column_data is not None
        and column_data["evaluator_name"] == str(evaluator_name)
        and column_data["metric_name"] == str(metric_name)
        and column_data["suffix"] == ""
        for column_name in agg_df.columns
        for column_data in [get_data_from_column(str(column_name))]
    )

def get_plot_file_metric_prefix(evaluator_name: str, metric_name: str) -> str:
    return f"{evaluator_name}_{metric_name}"

def metric_column_getter(c: int, space_name: str, method_name: str, evaluator_name: str, metric: str) -> str:
    return f"{c}C_{space_name}_{method_name}_{evaluator_name}_{metric}"


def _format_title_with_n(title: str, n_by_c: dict[str, int]) -> str:
    valid_n_by_c = {c_label: n for c_label, n in n_by_c.items() if n > 0}
    if not valid_n_by_c:
        return title

    n_values = list(valid_n_by_c.values())
    if len(set(n_values)) == 1:
        n_text = f"N={n_values[0]} datasets"
    else:
        n_text = "N=" + ", ".join(f"{c_label}:{n}" for c_label, n in valid_n_by_c.items())

    return f"{title}\n{n_text}" if title else n_text


def _format_improvability_ylabel(metric_name: str) -> str:
    metric_text = _latex_escape(str(metric_name))
    return (
        rf"$\mathrm{{Improvability}} = \left(1 - "
        rf"\frac{{\mathrm{{{metric_text}}}(\mathcal{{Q}}_K)}}"
        rf"{{\mathrm{{{metric_text}}}(\mathcal{{X}}_{{\mathrm{{full}}}})}}\right)\cdot 100\%$"
    )


def plot_metric_scatterplot(
    out_df: pd.DataFrame,
    x_space: str,
    x_method: str,
    x_evaluator: str,
    y_space: str,
    y_method: str,
    y_evaluator: str,
    metric: str,
    output_path: Path,
    title: str,
    arrow_labels: tuple[str, str] = ("OriginalSpace\nstronger", "TabClustPFN\nstronger"),
) -> None:
    x_template = f"{{multiplier}}C_{x_space}_{x_method}_{x_evaluator}_{metric}"
    y_template = f"{{multiplier}}C_{y_space}_{y_method}_{y_evaluator}_{metric}"

    valid_points_by_c: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    wilcoxon_p_by_c: dict[int, float] = {}
    mean_auc_diff_by_c: dict[int, float] = {}
    all_x = []
    all_y = []

    for multiplier in C_MULTIPLIERS:
        x_col = x_template.format(multiplier=multiplier)
        y_col = y_template.format(multiplier=multiplier)
        if x_col not in out_df.columns or y_col not in out_df.columns:
            logger.warning("Missing %s scatter columns for %sC: %s, %s", metric, multiplier, x_col, y_col)
            continue

        subset = out_df[[x_col, y_col]].copy()
        subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
        subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
        subset = subset.dropna(subset=[x_col, y_col])
        if subset.empty:
            continue

        x_vals = subset[x_col].to_numpy(dtype=float)
        y_vals = subset[y_col].to_numpy(dtype=float)
        valid_points_by_c[multiplier] = (x_vals, y_vals)
        mean_auc_diff_by_c[multiplier] = float(np.mean(y_vals) - np.mean(x_vals))
        all_x.extend(x_vals.tolist())
        all_y.extend(y_vals.tolist())

        try:
            _, p_value = wilcoxon(x_vals, y_vals, zero_method="wilcox", alternative="two-sided")
            wilcoxon_p_by_c[multiplier] = float(p_value)
        except ValueError as exc:
            logger.warning(
                "Could not compute Wilcoxon p-value for %sC in %s scatter (%s vs %s): %s",
                multiplier,
                metric,
                f"{x_space}/{x_method}",
                f"{y_space}/{y_method}",
                exc,
            )

    if not valid_points_by_c:
        logger.warning("No valid %s points found to plot. Scatterplot not saved.", metric)
        return

    all_values = np.array(all_x + all_y, dtype=float)
    min_val = float(np.nanmin(all_values))
    max_val = float(np.nanmax(all_values))
    span = max(max_val - min_val, 1e-6)
    margin = 0.05 * span
    lo = min_val - margin
    hi = max_val + margin

    fig, axes = plt.subplots(
        1,
        len(C_MULTIPLIERS),
        figsize=(4.4 * len(C_MULTIPLIERS), 6),
        sharex=True,
        sharey=True,
    )
    if len(C_MULTIPLIERS) == 1:
        axes = [axes]

    for idx, multiplier in enumerate(C_MULTIPLIERS):
        ax = axes[idx]
        p_value = wilcoxon_p_by_c.get(multiplier)
        mean_auc_diff = mean_auc_diff_by_c.get(multiplier)
        mean_auc_diff_text = f"{mean_auc_diff:.4f}" if mean_auc_diff is not None else "N/A"
        if p_value is not None:
            ax.set_title(
                f"K = {multiplier}⋅num_classes\n"
                f"Wilcoxon $p$={p_value:.3g}\n"
                f"Mean {metric} diff: {mean_auc_diff_text}",
                fontsize=16,
            )
        else:
            ax.set_title(
                f"K = {multiplier}⋅num_classes\n"
                f"Wilcoxon $p$-N/A\n"
                f"Mean {metric} diff: {mean_auc_diff_text}",
                fontsize=16,
            )
        ax.grid(False)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="both", labelsize=12)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color="black")

        if multiplier in valid_points_by_c:
            x_vals, y_vals = valid_points_by_c[multiplier]
            above_diag_mask = y_vals > x_vals
            below_diag_mask = y_vals < x_vals

            if np.any(above_diag_mask):
                ax.scatter(
                    x_vals[above_diag_mask],
                    y_vals[above_diag_mask],
                    alpha=0.65,
                    s=38,
                    color=DEFAULT_SPACE_COLORS[y_space],
                    edgecolors=DEFAULT_SPACE_COLORS[y_space],
                    linewidths=0.8,
                )

            if np.any(below_diag_mask):
                ax.scatter(
                    x_vals[below_diag_mask],
                    y_vals[below_diag_mask],
                    alpha=0.65,
                    s=38,
                     color=DEFAULT_SPACE_COLORS[x_space],
                    edgecolors=DEFAULT_SPACE_COLORS[x_space],
                    linewidths=0.8,
                )

            on_diag_mask = ~(above_diag_mask | below_diag_mask)
            if np.any(on_diag_mask):
                ax.scatter(
                    x_vals[on_diag_mask],
                    y_vals[on_diag_mask],
                    alpha=0.65,
                    s=38,
                    color="gray",
                    edgecolors="gray",
                    linewidths=0.8,
                )

            ax.annotate(
                arrow_labels[1],
                xy=(lo, hi),
                xytext=(lo + 0.24 * (hi - lo), hi - 0.204 * (hi - lo)),
                xycoords="data",
                textcoords="data",
                color=DEFAULT_SPACE_COLORS[y_space],
                fontsize=12,
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "->", "color": DEFAULT_SPACE_COLORS[y_space], "lw": 1.4},
            )
            ax.annotate(
                arrow_labels[0],
                xy=(hi, lo),
                xytext=(lo + 0.80 * (hi - lo), lo + 0.24 * (hi - lo)),
                xycoords="data",
                textcoords="data",
                color=DEFAULT_SPACE_COLORS[x_space],
                fontsize=12,
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "->", "color": DEFAULT_SPACE_COLORS[x_space], "lw": 1.4},
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                alpha=0.75,
            )

    fig.suptitle(title)

    def _display_method_name(method_name: str) -> str:
        return str(method_name).split("UnderSampler", 1)[0]

    fig.supxlabel(f"{_display_method_name(x_method)} in {x_space}", fontsize=20)
    fig.supylabel(f"{_display_method_name(y_method)} in {y_space}", fontsize=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.92))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s scatterplot to %s", metric, format_plot_output_path(output_path))


def plot_metric_vs_c_boxplot_diff_two_methods(
    out_df: pd.DataFrame,
    output_path: Path,
    metric: str,
    x_space: str = "OriginalSpace",
    x_method: str = "KMedoidsUnderSamplerK-medoids++Cosine",
    x_evaluator: str = "TabPFNv2_5",
    x_color: str = None,
    y_space: str = "TabClustPFNSpace",
    y_method: str = "KMedoidsUnderSamplerK-medoids++Cosine",
    y_evaluator: str = "TabPFNv2_5",
    y_color: str = None,
    title: str = "Boxplot of metric diff per C",
    ylabel: str = "",
    showfliers: bool = False,
) -> None:
    records = []
    p_values_by_c: dict[str, float] = {}
    n_by_c: dict[str, int] = {}
    for c in C_MULTIPLIERS:
        col1 = metric_column_getter(c, x_space, x_method, x_evaluator, metric)
        col2 = metric_column_getter(c, y_space, y_method, y_evaluator, metric)
        c_label = f"{c}C"
        
        if col1 not in out_df.columns or col2 not in out_df.columns:
            logger.warning("Metric columns for %sC not found: %s, %s", c, col1, col2)
            continue
        
        val1 = pd.to_numeric(out_df[col1], errors="coerce")
        val2 = pd.to_numeric(out_df[col2], errors="coerce")
        paired_values = pd.concat([val1, val2], axis=1).dropna()
        n_by_c[c_label] = len(paired_values)
        diffs = paired_values.iloc[:, 1] - paired_values.iloc[:, 0]

        if len(paired_values) >= 2:
            try:
                if np.isclose(diffs.to_numpy(dtype=float), 0.0).all():
                    p_values_by_c[c_label] = 1.0
                else:
                    _, p_value = wilcoxon(paired_values.iloc[:, 1], paired_values.iloc[:, 0])
                    p_values_by_c[c_label] = float(p_value)
            except ValueError as exc:
                logger.warning("Wilcoxon test for %s %s could not be computed (%s)", metric, c_label, exc)
        
        for v in diffs:
            records.append({"C": c_label, "metric_diff": float(v)})

    if not records:
        logger.warning("No paired metric values found for metric difference boxplot. Aborting.")
        return

    df_plot = pd.DataFrame.from_records(records)
    c_labels = [f"{c}C" for c in C_MULTIPLIERS if f"{c}C" in set(df_plot["C"])]
    data_by_c = [
        pd.to_numeric(df_plot.loc[df_plot["C"] == c_label, "metric_diff"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for c_label in c_labels
    ]

    fig, ax = plt.subplots(figsize=(len(c_labels) * 1.2, 7))
    positions = np.arange(len(c_labels), dtype=float)
    box_width = 0.55
    boxplot = ax.boxplot(
        data_by_c,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=showfliers,
        showmeans=True,
        meanline=True,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.2, "zorder": 3},
        medianprops={"color": "black", "linewidth": 1.4, "zorder": 4},
        meanprops={"color": "black", "linewidth": 1.4, "linestyle": "--", "zorder": 4},
        whiskerprops={"color": "black", "linewidth": 1.1, "zorder": 3},
        capprops={"color": "black", "linewidth": 1.1, "zorder": 3},
    )

    for position, values, box in zip(positions, data_by_c, boxplot["boxes"], strict=False):
        q1, q3 = np.nanpercentile(values, [25, 75])
        box.set_facecolor("none")
        box.set_zorder(3)

        x_left = position - (box_width / 2)
        if q1 < 0:
            negative_top = min(q3, 0.0)
            if negative_top > q1:
                ax.add_patch(
                    Rectangle(
                        (x_left, q1),
                        box_width,
                        negative_top - q1,
                        facecolor=x_color or DEFAULT_SPACE_COLORS.get(x_space, "tab:orange"),
                        edgecolor="none",
                        alpha=0.65,
                        zorder=2,
                    )
                )
        if q3 > 0:
            positive_bottom = max(q1, 0.0)
            if q3 > positive_bottom:
                ax.add_patch(
                    Rectangle(
                        (x_left, positive_bottom),
                        box_width,
                        q3 - positive_bottom,
                        facecolor=y_color or DEFAULT_SPACE_COLORS.get(y_space, "steelblue"),
                        edgecolor="none",
                        alpha=0.65,
                        zorder=2,
                    )
                )

    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.035
    for position, c_label, upper_cap in zip(positions, c_labels, boxplot["caps"][1::2], strict=False):
        p_value = p_values_by_c.get(c_label)
        if p_value is None or pd.isna(p_value):
            continue

        cap_y = float(np.nanmax(upper_cap.get_ydata()))
        p_value_text = rf"p={p_value:.3g}"
        if p_value < 0.05:
            p_value_text = rf"$\mathbf{{p={p_value:.3g}}}$"
        ax.text(
            position,
            cap_y + y_offset,
            p_value_text,
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax.set_ylim(top=max(ax.get_ylim()[1], max(float(np.nanmax(cap.get_ydata())) for cap in boxplot["caps"][1::2]) + y_offset * 3.0))

    ax.set_xticks(positions)
    ax.set_xticklabels(c_labels)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Selection size (×C)", fontsize=24)
    ax.set_ylabel(ylabel or f"{metric} diff (TabClustPFN - Original)", fontsize=24)
    ax.set_title(_format_title_with_n(title, n_by_c), fontsize=24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Saved %s diff boxplot to %s", metric, format_plot_output_path(output_path))



def plot_metric_vs_c_lineplot(
    out_df: pd.DataFrame,
    metric: str,
    figure_size: tuple[float, float],
    selected_method_styles: dict[tuple[str, str, str], dict],
    output_path: Path,
    title: str = "",
    ylabel: str = "",
    show_std: bool = False,
    std_alpha: float = 0.18,
    legend_inside: bool = False,
) -> None:

    series_stats: dict[tuple[str, str, str], np.ndarray] = {}
    series_std: dict[tuple[str, str, str], np.ndarray] = {}
    n_by_c: dict[str, int] = {}
    plotted_columns_by_c: dict[int, list[str]] = {c: [] for c in C_MULTIPLIERS}
    for space, selector, evaluator in selected_method_styles:
        mean_values = []
        std_values = []
        for c in C_MULTIPLIERS:
            i_col = metric_column_getter(c, space, selector, evaluator, metric)
            if i_col not in out_df.columns:
                mean_values.append(np.nan)
                std_values.append(np.nan)
                continue

            plotted_columns_by_c[c].append(i_col)
            improv = pd.to_numeric(out_df[i_col], errors="coerce").dropna()
            if improv.empty:
                mean_values.append(np.nan)
                std_values.append(np.nan)
            else:
                mean_values.append(float(improv.mean()))
                std_values.append(float(improv.std(ddof=1)) if len(improv) > 1 else 0.0)

        series_stats[(space, selector, evaluator)] = np.asarray(mean_values, dtype=float)
        series_std[(space, selector, evaluator)] = np.asarray(std_values, dtype=float)

    has_any_series = any(not np.isnan(mean_arr).all() for mean_arr in series_stats.values())
    if not has_any_series:
        logger.warning(
            "No valid improvability values found for improvability-vs-C plot. Plot %s not saved.",
            format_plot_output_path(output_path),
        )
        return

    for c, columns in plotted_columns_by_c.items():
        if not columns:
            continue
        numeric_subset = out_df[columns].apply(pd.to_numeric, errors="coerce")
        n_by_c[f"{c}C"] = int(numeric_subset.notna().any(axis=1).sum())

    fig, ax = plt.subplots(figsize=figure_size)
    c_arr = np.asarray(C_MULTIPLIERS, dtype=float)
    other_entries: list[tuple[object, str]] = []

    for (space, selector, evaluator), style in selected_method_styles.items():
        mean_arr = series_stats.get((space, selector, evaluator))
        if mean_arr is None or np.isnan(mean_arr).all():
            continue

        label = style.get("label", f"{space} | {selector} | {evaluator}")
        color = style.get("color", "black")
        if show_std:
            std_arr = series_std.get((space, selector, evaluator))
            if std_arr is not None:
                valid_std = ~np.isnan(mean_arr) & ~np.isnan(std_arr)
                if valid_std.any():
                    ax.fill_between(
                        c_arr[valid_std],
                        mean_arr[valid_std] - std_arr[valid_std],
                        mean_arr[valid_std] + std_arr[valid_std],
                        color=color,
                        alpha=std_alpha,
                        linewidth=0,
                        zorder=1,
                    )
        line = ax.plot(
            c_arr,
            mean_arr,
            linewidth=2.2,
            linestyle=style.get("linestyle", "-"),
            marker=None,
            color=color,
            alpha=float(style.get("alpha", 1.0)),
            label=label,
            zorder=2,
        )[0]
        valid_points = ~np.isnan(mean_arr)
        ax.scatter(
            c_arr[valid_points],
            mean_arr[valid_points],
            s=34,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=float(style.get("alpha", 1.0)),
            zorder=3,
        )
        other_entries.append((line, label))

    ax.set_xscale("log")
    ax.set_xlabel("Selection size (×C)", fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    if metric.endswith("_I"):
        # If it is an Improvability metric, format the Y axis as a percentage
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.set_title(_format_title_with_n(title, n_by_c), fontsize=18)
    ax.set_xticks(C_MULTIPLIERS)
    ax.set_xticklabels([f"{c}C" for c in C_MULTIPLIERS])
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(False)

    if other_entries:
        handles, labels = zip(*other_entries, strict=False)
        if legend_inside:
            ax.legend(
                handles,
                labels,
                loc="best",
                frameon=True,
                framealpha=0.92,
                handlelength=2.2,
                labelspacing=0.5,
                fontsize=18,
            )
        else:
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.06, 0.5),
                borderaxespad=0.0,
                frameon=True,
                handlelength=2.2,
                labelspacing=0.5,
                fontsize=18,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Saved improvability-vs-C plot to %s", format_plot_output_path(output_path))


def plot_posthoc_cd_diagram_per_c(
    out_df: pd.DataFrame,
    output_path: Path,
    metric: str,
    selected_space_method_pairs: dict[tuple[str, str, str], tuple[str, str, str]],
) -> None:
    fig, axes = plt.subplots(
        len(C_MULTIPLIERS),
        1,
        figsize=(14, max(3.2 * len(C_MULTIPLIERS), 4.0)),
        squeeze=False,
    )
    axes = axes.ravel().tolist()

    for ax, c in zip(axes, C_MULTIPLIERS):
        col_map = {}
        for space_name, method_name, evaluator_name in selected_space_method_pairs.keys():
            col = metric_column_getter(c, space_name, method_name, evaluator_name, metric)
            if col in out_df.columns:
                label = f"{space_name} | {method_name} | {evaluator_name}"
                col_map[label] = col

        if len(col_map) < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Post-Hoc {c}C: not enough methods", ha="center", va="center")
            continue

        comp_df = out_df[list(col_map.values())].copy()
        comp_df = comp_df.rename(columns={v: k for k, v in col_map.items()})
        for col in comp_df.columns:
            comp_df[col] = pd.to_numeric(comp_df[col], errors="coerce")
        comp_df = comp_df.dropna(axis=0, how="any")

        if comp_df.shape[0] < 2:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Post-Hoc {c}C: not enough complete datasets after NaN filtering", ha="center", va="center")
            continue

        avg_ranks = comp_df.rank(axis=1, ascending=False, method="average").mean(axis=0)

        friedman_p = np.nan
        try:
            values_per_model = [comp_df[col].to_numpy(dtype=float) for col in comp_df.columns]
            _, friedman_p = friedmanchisquare(*values_per_model)
        except ValueError as exc:
            logger.warning("C=%s: Friedman test could not be computed (%s)", c, exc)

        try:
            p_values = sp.posthoc_nemenyi_friedman(comp_df)
            sp.critical_difference_diagram(
                ranks=avg_ranks,
                sig_matrix=p_values,
                ax=ax,
            )
            ax.set_title(
                f"{c}C | Avg rank by dataset (lower is better) | "
                f"N={len(comp_df)} datasets | Friedman p={friedman_p:.4g}"
            )
        except Exception as exc:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Post-Hoc {c}C: could not build CD diagram ({exc})", ha="center", va="center")
            logger.warning("C=%s: failed to build critical difference diagram (%s)", c, exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Saved post-hoc critical difference diagrams to %s", format_plot_output_path(output_path))


def main() -> None:

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("lucos").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    ###############################################################
    #                         SSMA plots                          #
    ###############################################################

    agg_df = read_aggregated_results("SSMA")
    logger.info("Loaded aggregated SSMA results")

    plot_metric_vs_c_lineplot(
        agg_df,
        "AUC_I",
        (8, 7.2),
        legend_inside=True,
        output_path=build_plot_output_path("SSMA", "SSMA_AUC_I_vs_C.pdf"),
        selected_method_styles=METHOD_SPACE_EVALUATOR_RENAME_AND_STYLE_SSMA,
        title="",
        #ylabel="AUC Improvability (lower is better)",
        ylabel=_format_improvability_ylabel("AUC"),
    )

    ###############################################################
    #                         LUCoS plots                          #
    ###############################################################

    LUCoS_space = "TabClustPFNSpace"
    agg_df = read_aggregated_results("LUCoS")
    logger.info("Loaded aggregated LUCoS results")
    
    
    for evaluator, metrics_of_evaluator in METRICS_TO_AGGREGATE_PER_EVALUATOR.items():
        for metric in metrics_of_evaluator:

            if not has_metric_for_evaluator(agg_df, evaluator, metric):
                logger.info("Skipping %s/%s plots because no aggregated columns were found.", evaluator, metric)
                continue


            ###########################################################################
            #            PLOTS COMPARING ALL SELECTION METHODS                       #
            ###########################################################################

            metric_method_styles = get_lucos_method_styles_for_evaluator(evaluator)
            metric_file_prefix = get_plot_file_metric_prefix(evaluator, metric)

            # METRIC VS C LINEPLOTS
            plot_metric_vs_c_lineplot(
                agg_df,
                metric,
                (7, 8),
                legend_inside=True,
                output_path=build_lucos_plot_output_path(evaluator, metric, f"LUCoS_{metric_file_prefix}_vs_C_lineplot.pdf"),
                selected_method_styles=metric_method_styles,
                title="",
                ylabel=f"{metric} (higher is better)"
            )

            # METRIC IMPROVABILITY VS C LINEPLOTS
            plot_metric_vs_c_lineplot(
                agg_df,
                f"{metric}_I",
                (7.2, 7),
                legend_inside=True,
                output_path=build_lucos_plot_output_path(evaluator, metric, f"LUCoS_{metric_file_prefix}_I_vs_C_lineplot.pdf"),
                selected_method_styles=metric_method_styles,
                title="",
                #ylabel=f"{metric} Improvability (lower is better)"
                ylabel=_format_improvability_ylabel(metric),
            )

            # CRITICAL DIFFERENCE DIAGRAMS - FRIEDMAN + NEMENYI
            plot_posthoc_cd_diagram_per_c(
                agg_df,
                metric=metric,
                output_path=build_lucos_plot_output_path(evaluator, metric, f"LUCoS_{metric_file_prefix}_posthoc_cd_by_c.pdf"),
                selected_space_method_pairs=metric_method_styles,
            )



            ###########################################################################
            #            PLOTS COMPARING ONLY TWO SELECTION METHODS                  #
            #      COMPARE TABCLUSTPFN-KM AGAINST THE ORIGINALS (RANDOM AND KM)      #
            #     AND PLOT BOTH VERSIONS: WITH EUCLIDEAN AND WITH COSINE             #
            ###########################################################################

            LUCoS_KM_metrics = {"Cosine": "KMedoidsUnderSamplerK-medoids++Cosine", "Euclidean": "KMedoidsUnderSamplerK-medoids++Euclidean"}
            for lucos_km_metric_name, metric_lucos_method in LUCoS_KM_metrics.items():

                # SCATTERPLOT: LUCoS vs Original-RANDOM
                plot_metric_scatterplot(
                    agg_df,
                    x_space="OriginalSpace",
                    x_method="RandomUnderSamplerUnsupervised",
                    x_evaluator=evaluator,
                    y_space=LUCoS_space,
                    y_method=metric_lucos_method,
                    y_evaluator=evaluator,
                    metric=metric,
                    output_path=build_lucos_plot_output_path(
                        evaluator,
                        metric,
                        f"LUCoS_{metric_file_prefix}_random_vs_tabclust_scatterplot.pdf",
                        lucos_km_metric_name,
                    ),
                    title="",
                    arrow_labels=("Original-Random\nstronger", "TabClustPFN-KM\nstronger"),
                )

                # SCATTERPLOT: LUCoS vs Original-KM
                plot_metric_scatterplot(
                    agg_df,
                    x_space="OriginalSpace",
                    x_method=metric_lucos_method,
                    x_evaluator=evaluator,
                    y_space=LUCoS_space,
                    y_method=metric_lucos_method,
                    y_evaluator=evaluator,
                    metric=metric,
                    output_path=build_lucos_plot_output_path(
                        evaluator,
                        metric,
                        f"LUCoS_{metric_file_prefix}_km_vs_tabclust_scatterplot.pdf",
                        lucos_km_metric_name,
                    ),
                    title="",
                    arrow_labels=("Original-KM\nstronger", "TabClustPFN-KM\nstronger"),
                )

                # METRIC DIFFERENCE BOXPLOT: LUCoS vs Original-KM (FIGURE 3 OF PAPER)
                plot_metric_vs_c_boxplot_diff_two_methods(
                    agg_df,
                    output_path=build_lucos_plot_output_path(
                        evaluator,
                        metric,
                        f"LUCoS_{metric_file_prefix}_km_diff_boxplot.pdf",
                        lucos_km_metric_name,
                    ),
                    metric=metric,
                    x_space="OriginalSpace",
                    x_method=metric_lucos_method,
                    x_evaluator=evaluator,
                    y_space=LUCoS_space,
                    y_method=metric_lucos_method,
                    y_evaluator=evaluator,
                    title="",
                    ylabel=rf"$\Delta${metric}= {metric}$_{{\text{{LUCoS}}}}$ - {metric}$_{{\text{{Original-KM}}}}$"
                )

                # METRIC DIFFERENCE BOXPLOT: LUCoS vs Original-RANDOM
                plot_metric_vs_c_boxplot_diff_two_methods(
                    agg_df,
                    output_path=build_lucos_plot_output_path(
                        evaluator,
                        metric,
                        f"LUCoS_{metric_file_prefix}_rnd_diff_boxplot.pdf",
                        lucos_km_metric_name,
                    ),
                    metric=metric,
                    x_space="OriginalSpace",
                    x_method="RandomUnderSamplerUnsupervised",
                    x_evaluator=evaluator,
                    x_color="#808080",
                    y_space=LUCoS_space,
                    y_method=metric_lucos_method,
                    y_evaluator=evaluator,
                    title="",
                    ylabel=rf"$\Delta${metric}= {metric}$_{{\text{{LUCoS}}}}$ - {metric}$_{{\text{{Original-Random}}}}$"
                )

if __name__ == "__main__":
    main()
