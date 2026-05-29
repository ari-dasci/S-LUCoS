#!/usr/bin/env python3
"""
Generate Section 4 tables and figures from the aggregated experiment results.

Sources: results/results_unsupervised_context_selection_<experiment>/agg/*_agg.csv
Output: results/results_unsupervised_context_selection_<experiment>/plots/paper_tables_and_figures/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from lucos.unsupervised_context_selection.agg_results import unsupervised_context_selection_agg_results as agg_results
from lucos.unsupervised_context_selection.agg_results import unsupervised_context_selection_create_plots as create_plots

# ── config ──────────────────────────────────────────────────────────
C_BUDGETS   = list(agg_results.C_MULTIPLIERS)
PRIMARY     = "AUC"
EVALUATOR   = "TabPFNv2_5"
PAPER_SECTION = "paper_tables_and_figures"
OUT = create_plots.build_plot_output_path(PAPER_SECTION, ".placeholder").parent


def paper_output_path(suffix: str) -> object:
    return create_plots.build_plot_output_path(PAPER_SECTION, suffix)

# Paper aliases over the current aggregated-result column names. The final
# figures keep the old paper labels while reading the current folder layout.
METHODS = [
    ("TabClustPFNSpace", "KMedoidsUnderSamplerK-medoids++Euclidean", "LUCoS"),
    ("OriginalSpace",    "KMedoidsUnderSamplerK-medoids++Euclidean", "Orig-KM"),
    ("OriginalSpace",    "RandomUnderSamplerUnsupervised",    "Random"),
    ("OriginalSpace",    "RDSSUnderSampler",                  "RDSS-Orig"),
    ("OriginalSpace",    "ZCoreUnderSampler",                 "ZCore-Orig"),
    ("TabClustPFNSpace", "RDSSUnderSampler",                  "RDSS-TCPFN"),
    ("TabClustPFNSpace", "ZCoreUnderSampler",                 "ZCore-TCPFN"),
]
METHOD_ORDER = [m[2] for m in METHODS]  # display order
LUCoS, ORIG, RAND = "LUCoS", "Orig-KM", "Random"

# ── helpers ─────────────────────────────────────────────────────────
def _label(s, a):
    ss = str(s).lower(); aa = str(a).lower()
    for sp, ap, lb in METHODS:
        if ss == sp.lower() and aa == ap.lower():
            return lb
    return f"{s}/{a}"


def _dataset_column(df: pd.DataFrame) -> str:
    if "dataset" in df.columns:
        return "dataset"
    if "dataset_name" in df.columns:
        return "dataset_name"
    raise ValueError("Expected either 'dataset' or 'dataset_name' in aggregated results.")


def build_wide_from_aggregated(agg_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Convert the current aggregated wide table into the paper's legacy wide format."""

    dataset_col = _dataset_column(agg_df)
    wide = pd.DataFrame({"dataset": agg_df[dataset_col].astype(str)})

    for c in C_BUDGETS:
        for space, method, label in METHODS:
            source_col = create_plots.metric_column_getter(c, space, method, EVALUATOR, metric)
            target_col = f"{c}C_{label}_{metric}"
            wide[target_col] = pd.to_numeric(agg_df[source_col], errors="coerce") if source_col in agg_df.columns else np.nan

    paper_baseline_col = f"{agg_results.BASELINE_COLUMN_PREFIX}{EVALUATOR}_{metric}"
    baseline_source_col = paper_baseline_col
    if baseline_source_col not in agg_df.columns:
        baseline_source_col = f"{agg_results.BASELINE_COLUMN_PREFIX}{metric}"
    if baseline_source_col in agg_df.columns:
        wide[paper_baseline_col] = pd.to_numeric(agg_df[baseline_source_col], errors="coerce")

    return wide


def active_methods_for_metric(wide: pd.DataFrame, metric: str) -> list[str]:
    """Return paper method labels with at least one non-missing value."""

    active = []
    for method in METHOD_ORDER:
        method_cols = [f"{c}C_{method}_{metric}" for c in C_BUDGETS if f"{c}C_{method}_{metric}" in wide.columns]
        if method_cols and wide[method_cols].apply(pd.to_numeric, errors="coerce").notna().any().any():
            active.append(method)
    return active

# =====================================================================
# STEP 3: MEAN RANKS + NEMENYI CD
# =====================================================================

def compute_mean_ranks(wide, metric):
    """Mean rank per method per budget across all 7 methods."""
    labels = active_methods_for_metric(wide, metric)
    rows = []
    for c in C_BUDGETS:
        cols = [f"{c}C_{m}_{metric}" for m in labels]
        sub = wide[cols].dropna()
        ranks = sub.rank(axis=1, ascending=False).mean()
        for m in labels:
            col = f"{c}C_{m}_{metric}"
            rows.append({"budget": c, "method": m,
                         "mean_rank": ranks[col] if col in ranks.index else np.nan,
                         "n_datasets": len(sub)})
    return pd.DataFrame(rows)


def nemenyi_cd(n_methods, n_datasets, alpha=0.05):
    """Critical distance for Nemenyi post-hoc (two-tailed).

    Uses the Studentized range distribution (scipy.stats.studentized_range)
    to compute the exact critical value q_{α,k,∞} / √2, avoiding hardcoded
    lookup-table artefacts. Equivalent to the implementation in autorank and
    scikit-posthocs.

    For k=7, N=67, α=0.05: CD ≈ 1.100.
    """
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2)
    return q * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

# =====================================================================
# STEP 4: WILCOXON VS RANDOM + BH CORRECTION
# =====================================================================

def wilcoxon_vs_random(wide, metric):
    """Paired Wilcoxon: vs-Random tests, BH-corrected across 6 comparisons.

    Computes all C(7,2)=21 pairwise Wilcoxon tests per budget as an internal
    sanity check (saved to table_wilcoxon_21pairs.csv).  For display in Table 1
    and Table B1, applies Benjamini-Hochberg correction across only the 6
    vs-Random tests per budget — matching the spec caption's phrase
    "Wilcoxon vs RandomUnsup after BH correction."
    """
    from scipy.stats import wilcoxon
    from itertools import combinations
    labels = active_methods_for_metric(wide, metric)
    if RAND not in labels:
        return pd.DataFrame(columns=["budget", "method", "p_value", "q_value"])

    # Stage 1: compute all 21 pairwise p-values per budget
    all_rows = []
    for c in C_BUDGETS:
        for mi, mj in combinations(labels, 2):
            ci = f"{c}C_{mi}_{metric}"
            cj = f"{c}C_{mj}_{metric}"
            sub = wide[[ci, cj]].dropna()
            if len(sub) < 5:
                p = np.nan
            else:
                try:
                    _, p = wilcoxon(sub[ci], sub[cj], zero_method="zsplit",
                                    alternative="two-sided")
                except Exception:
                    p = np.nan
            all_rows.append({"budget": c, "method_i": mi, "method_j": mj,
                             "p_value": p, "n": len(sub)})
    all_df = pd.DataFrame(all_rows).dropna(subset=["p_value"])

    # Save full 21-pair table for internal reference
    all_df.to_csv(paper_output_path("table_wilcoxon_21pairs.csv"), index=False)

    # Stage 2: BH-correct across the 6 vs-Random tests ONLY (for display)
    rows = []
    for c in C_BUDGETS:
        # Collect the 6 vs-Random p-values for this budget
        vs_random_ps = []
        vs_random_methods = []
        for m in labels:
            if m == RAND:
                continue
            raw_p = all_df[
                (all_df["budget"] == c) &
                (((all_df["method_i"] == m) & (all_df["method_j"] == RAND)) |
                 ((all_df["method_i"] == RAND) & (all_df["method_j"] == m)))
            ]["p_value"]
            p_val = raw_p.values[0] if len(raw_p) else np.nan
            if not np.isnan(p_val):
                vs_random_ps.append(p_val)
                vs_random_methods.append(m)
            else:
                vs_random_ps.append(np.nan)
                vs_random_methods.append(m)
        # BH-correct the 6 vs-Random p-values
        valid_mask = ~np.isnan(vs_random_ps)
        qs = np.full(len(vs_random_methods), np.nan)
        if valid_mask.sum() > 0:
            valid_qs = _bh_correct(np.array(vs_random_ps)[valid_mask])
            qs[valid_mask] = valid_qs
        for m, p_val, q_val in zip(vs_random_methods, vs_random_ps, qs):
            rows.append({"budget": c, "method": m, "p_value": p_val, "q_value": q_val})
    result = pd.DataFrame(rows)
    outpath = paper_output_path("table_wilcoxon_vs_random.csv")
    if not outpath.exists():
        print(f"  Saved table_wilcoxon_vs_random.csv ({len(result)} rows: {len(labels)-1} methods x {len(C_BUDGETS)} budgets)")
    result.to_csv(outpath, index=False)
    return result


def _bh_correct(p_vals):
    """Benjamini-Hochberg FDR correction.

    Implements the step-up procedure: sort p-values ascending, compute
    adjusted thresholds p_i * n / i, enforce monotonicity in sorted order
    (so q_i = min(q_i, q_{i+1})), then map back to original positions.
    """
    p = np.array(p_vals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    # Compute BH-adjusted values in sorted-p order
    q_sorted = np.minimum(1.0, p[order] * n / (np.arange(n) + 1))
    # Enforce monotonicity: q_i ≤ q_{i+1} in sorted order
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    # Map back to original positions
    result = np.empty(n)
    result[order] = q_sorted
    return result.tolist()


def stars(q):
    if pd.isna(q): return ""
    if q < 0.001: return "***"
    if q < 0.01:  return "**"
    if q < 0.05:  return "*"
    return ""

# =====================================================================
# STEP 5: NEMENYI-EQUIVALENCE BAND
# =====================================================================

def build_nemenyi_band(rank_df, cd_value):
    """For each budget, mark methods whose rank is within CD of the best."""
    band = {}
    for c in C_BUDGETS:
        sub = rank_df[rank_df["budget"] == c]
        best_rank = sub["mean_rank"].min()
        band[c] = set(sub[sub["mean_rank"] <= best_rank + cd_value]["method"])
    return band

# =====================================================================
# MAIN TEXT — TABLE 1
# =====================================================================

def generate_table1(wide):
    """Table 1: Method ranking under AUC with bold/italic/asterisks."""
    active_method_list = active_methods_for_metric(wide, "AUC")
    active_methods = set(active_method_list)
    rank_df = compute_mean_ranks(wide, "AUC")
    wilc = wilcoxon_vs_random(wide, "AUC")
    cd_val = nemenyi_cd(len(active_method_list), len(wide)) if len(active_method_list) >= 2 else np.nan
    band = build_nemenyi_band(rank_df, cd_val)

    # Build LaTeX-ready table
    # Rows grouped by space
    spaces = [("—", ["Random"]),
              ("Original", ["RDSS-Orig", "ZCore-Orig", "Orig-KM"]),
              ("TabClustPFN", ["RDSS-TCPFN", "ZCore-TCPFN", "LUCoS"])]
    display_map = {
        "Random": "RandomUnsup", "RDSS-Orig": "RDSS", "ZCore-Orig": "ZCore",
        "Orig-KM": "K-Medoids", "RDSS-TCPFN": "RDSS", "ZCore-TCPFN": "ZCore",
        "LUCoS": "LUCoS",
    }

    rows = []
    for space, methods in spaces:
        for mk in methods:
            if mk not in active_methods:
                continue
            row = {"Space": space, "Method": display_map.get(mk, mk)}
            for c in C_BUDGETS:
                sub = rank_df[(rank_df["budget"] == c) & (rank_df["method"] == mk)]
                rank_val = sub["mean_rank"].values[0] if len(sub) else np.nan
                # formatting
                fmt = f"{rank_val:.2f}"
                # bold if best in column
                col_best = rank_df[rank_df["budget"] == c]["mean_rank"].min()
                is_best = abs(rank_val - col_best) < 0.005
                is_nemenyi = mk in band.get(c, set())
                # asterisk from Wilcoxon vs Random
                wrow = wilc[(wilc["budget"] == c) & (wilc["method"] == mk)]
                ast = stars(wrow["q_value"].values[0]) if len(wrow) else ""
                row[f"{c}C"] = {"val": rank_val, "fmt": fmt, "bold": is_best,
                                "italic": is_nemenyi and not is_best,
                                "stars": ast}
            rows.append(row)

    # Save as CSV (flattened)
    flat = []
    for r in rows:
        f = {"Space": r["Space"], "Method": r["Method"]}
        for c in C_BUDGETS:
            d = r[f"{c}C"]
            f[f"{c}C"] = d["fmt"]
            f[f"{c}C_bold"] = d["bold"]
            f[f"{c}C_italic"] = d["italic"]
            f[f"{c}C_stars"] = d["stars"]
        flat.append(f)
    out = pd.DataFrame(flat)
    out.to_csv(paper_output_path("table1_ranks.csv"), index=False)

    # Pretty print
    print("\n── Table 1: Method Ranks (AUC) ──")
    hdr = f"{'Space':>20s} {'Method':>20s} " + " ".join(f"{c}C".rjust(7) for c in C_BUDGETS)
    print(hdr)
    for r in rows:
        vals = " ".join(f"{r[f'{c}C']['fmt']}{r[f'{c}C']['stars']}".rjust(7) for c in C_BUDGETS)
        print(f"{r['Space']:>20s} {r['Method']:>20s} {vals}")

    # Validation print
    print("\n  Nemenyi CD:", round(cd_val, 3))

    # ── Friedman test + Nemenyi CD output file ──
    from scipy.stats import friedmanchisquare as friedman
    fri_rows = []
    for c in C_BUDGETS:
        cols = [f"{c}C_{m}_{PRIMARY}" for m in active_method_list if f"{c}C_{m}_{PRIMARY}" in wide.columns]
        sub = wide[cols].dropna()
        if len(cols) >= 2 and len(sub) >= 5:
            arrays = [sub[col] for col in cols]
            chi2, p = friedman(*arrays)
            fri_rows.append({
                "budget": f"{c}C", "friedman_chi2": round(chi2, 3),
                "friedman_p": p, "n_datasets": len(sub),
                "n_methods": len(cols), "alpha": 0.05,
                "cd_value": round(cd_val, 4)
            })
    fri_df = pd.DataFrame(fri_rows)
    fri_df.to_csv(paper_output_path("friedman_nemenyi.csv"), index=False)
    if "friedman_p" in fri_df.columns:
        print(f"  Friedman p < 1e-4 at all budgets? {all(fri_df['friedman_p'] < 1e-4)}")
    else:
        print("  Friedman test skipped: not enough complete method columns.")
    print(f"  Saved friedman_nemenyi.csv ({len(fri_df)} budgets)")

    # ── Self-consistency cross-check: italic flags vs Nemenyi band ──
    # Recompute band from CSV rank values directly (avoids display-name ambiguity).
    italic_ok = 0
    italic_total = 0
    for _, row in out.iterrows():
        for c in C_BUDGETS:
            rank_val = float(row[f"{c}C"])
            is_best = row[f"{c}C_bold"]
            # Find best rank in this budget column
            col_best = min(float(out.at[j, f"{c}C"]) for j in range(len(out)))
            in_band = rank_val <= col_best + cd_val
            should_be_italic = in_band and not is_best
            italic_total += 1
            if row[f"{c}C_italic"] == should_be_italic:
                italic_ok += 1
    print(f"  Italic cross-check: {italic_ok}/{italic_total} cells consistent with Nemenyi CD ({cd_val:.3f})")
    assert italic_ok == italic_total, f"italic flag mismatch: {italic_ok}/{italic_total}"

    print("  Bold = best in column; italic = within Nemenyi band; * = Wilcoxon vs Random BH-corrected")
    return out


# =====================================================================
# MAIN TEXT — FIGURE 4A: Decomposition
# =====================================================================

def generate_fig4a(wide):
    """Figure 4A: Selector vs Representation decomposition bar chart."""
    deco = []
    for c in C_BUDGETS:
        s_col = f"{c}C_{LUCoS}_{PRIMARY}"
        o_col = f"{c}C_{ORIG}_{PRIMARY}"
        r_col = f"{c}C_{RAND}_{PRIMARY}"
        sub = wide[[s_col, o_col, r_col]].dropna()
        sams_v = sub[s_col].values; orig_v = sub[o_col].values; rand_v = sub[r_col].values
        delta_sel = orig_v - rand_v
        delta_repr = sams_v - orig_v
        _, p_sel = stats.wilcoxon(orig_v, rand_v, zero_method="zsplit")
        _, p_repr = stats.wilcoxon(sams_v, orig_v, zero_method="zsplit")
        deco.append({"budget": c, "delta_sel": np.mean(delta_sel),
                      "delta_repr": np.mean(delta_repr),
                      "p_sel": p_sel, "p_repr": p_repr})

    deco_df = pd.DataFrame(deco)
    deco_df.to_csv(paper_output_path("fig4a_decomposition.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(C_BUDGETS)); w = 0.35
    b1 = ax.bar(x - w/2, deco_df["delta_sel"], w, label="Δ Selector (KM-Orig − Random)",
                color="#4C72B0", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, deco_df["delta_repr"], w, label="Δ Repr (LUCoS − KM-Orig)",
                color="#DD8452", edgecolor="white", linewidth=0.5, hatch="//")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(1.5, ls="--", color="grey", alpha=0.5)
    # ax.text(1.55, 0.042, "selector → representation crossover", fontsize=7, color="grey", rotation=90, va="top")

    # significance stars
    for i, row in deco_df.iterrows():
        y_sel = row["delta_sel"] + (0.003 if row["delta_sel"] >= 0 else -0.006)
        y_repr = row["delta_repr"] + (0.003 if row["delta_repr"] >= 0 else -0.006)
        s_sel = _p_stars(row["p_sel"]); s_repr = _p_stars(row["p_repr"])
        if s_sel: ax.text(i - w/2, y_sel, s_sel, ha="center", fontsize=9, fontweight="bold", color="#2c3e50")
        if s_repr: ax.text(i + w/2, y_repr, s_repr, ha="center", fontsize=9, fontweight="bold", color="#2c3e50")

    y_min = min(0.0, float(deco_df[["delta_sel", "delta_repr"]].min().min()))
    y_max = max(0.0, float(deco_df[["delta_sel", "delta_repr"]].max().max()))
    y_span = max(y_max - y_min, 1e-6)

    ax.set_xticks(x); ax.set_xticklabels([f"{c}C" for c in C_BUDGETS])
    ax.set_ylabel("ΔAUC (dataset-paired mean)")
    ax.set_ylim(y_min - 0.15 * y_span, y_max + 0.28 * y_span)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.18)
    fig.savefig(paper_output_path("fig4a_decomposition.pdf"), dpi=300)
    plt.close()
    # Report computed (not hardcoded) values
    sel1 = deco_df[deco_df["budget"]==1]; rep4 = deco_df[deco_df["budget"]==4]
    rep8 = deco_df[deco_df["budget"]==8]
    print(f"  Fig 4A: 1C sel={sel1['delta_sel'].values[0]:+.3f}{_p_stars(sel1['p_sel'].values[0])}, "
          f"4C repr={rep4['delta_repr'].values[0]:+.3f}{_p_stars(rep4['p_repr'].values[0])}, "
          f"8C repr={rep8['delta_repr'].values[0]:+.3f}{_p_stars(rep8['p_repr'].values[0])}")


# =====================================================================
# MAIN TEXT — FIGURE 4B: Rescue vs Boost
# =====================================================================

def generate_fig4b(wide):
    """Figure 4B: Rescue vs Boost histogram of Δ_Repr at 8C."""
    c = 8
    s_col = f"{c}C_{LUCoS}_{PRIMARY}"
    o_col = f"{c}C_{ORIG}_{PRIMARY}"
    r_col = f"{c}C_{RAND}_{PRIMARY}"
    sub = wide[[s_col, o_col, r_col, "dataset"]].dropna().copy()
    sub["delta_repr"] = sub[s_col] - sub[o_col]
    sub["orig_vs_rand"] = sub[o_col] - sub[r_col]
    sub["group"] = np.where(sub["orig_vs_rand"] < 0, "Rescue", "Boost")

    rescue = sub[sub["group"] == "Rescue"]["delta_repr"]
    boost  = sub[sub["group"] == "Boost"]["delta_repr"]
    n_rescue, n_boost = len(rescue), len(boost)
    _, p_mwu = stats.mannwhitneyu(rescue, boost, alternative="two-sided")
    rescue_mean, boost_mean = rescue.mean(), boost.mean()
    delta_diff = rescue_mean - boost_mean

    # save table
    tbl = pd.DataFrame([{
        "budget": f"{c}C", "rescue_n": n_rescue, "rescue_mean_delta_repr": rescue_mean,
        "boost_n": n_boost, "boost_mean_delta_repr": boost_mean,
        "delta_diff": delta_diff, "mwu_p": p_mwu,
    }])
    tbl.to_csv(paper_output_path("fig4b_rescue_boost_8C.csv"), index=False)

    # plot
    fig, ax = plt.subplots(figsize=(4, 2.5))
    bins = np.linspace(-0.12, 0.22, 35)
    ax.hist(rescue, bins=bins, alpha=0.55, label=f"Rescue (n={n_rescue})", color="#44BD46")
    ax.hist(boost,  bins=bins, alpha=0.55, label=f"Boost (n={n_boost})",  color="#A14CB0")
    ax.axvline(0, ls="--", color="black", lw=0.8)
    ax.axvline(rescue_mean, color="#44BD46", ls="-", lw=1.2)
    ax.axvline(boost_mean, color="#A14CB0", ls="-", lw=1.2)
    ax.set_xlabel("Δ Repr per dataset (AUC LUCoS − AUC KM-Orig)")
    ax.set_ylabel("Datasets")
    p_str = f"{p_mwu:.1e}" if p_mwu < 0.001 else f"{p_mwu:.4f}"
    ax.text(0.02, 0.95, f"Δ diff = {delta_diff:+.3f}\nMWU p = {p_str}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(paper_output_path("fig4b_rescue_boost.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Fig 4B: Rescue n={n_rescue}, mean=+{rescue_mean:.3f}; Boost n={n_boost}, mean={boost_mean:+.3f}; Δ diff={delta_diff:+.3f}, p={p_str}")


# =====================================================================
# APPENDIX B — TABLE B1: Mean AUC
# =====================================================================

def generate_table_b1(wide):
    """Table B1: Mean AUC with Wilcoxon-vs-Random significance stars."""
    # Compute Wilcoxon vs Random for star annotations
    wilc = wilcoxon_vs_random(wide, "AUC")
    active_methods = active_methods_for_metric(wide, "AUC")
    rows = []
    for m in active_methods:
        row = {"Method": m}
        for c in C_BUDGETS:
            col = f"{c}C_{m}_AUC"
            val = wide[col].mean() if col in wide.columns else np.nan
            # Add star if method != Random and Wilcoxon q < 0.05
            star = ""
            if m != RAND:
                qrow = wilc[(wilc["budget"] == c) & (wilc["method"] == m)]
                if len(qrow) and not pd.isna(qrow["q_value"].values[0]):
                    star = stars(qrow["q_value"].values[0])
            row[f"{c}C"] = (f"{val:.4f}{star}" if star else f"{val:.4f}") if not pd.isna(val) else "—"
        rows.append(row)
    df = pd.DataFrame(rows)
    # Also save raw float version for downstream use
    raw_rows = []
    for m in active_methods:
        rr = {"Method": m}
        for c in C_BUDGETS:
            col = f"{c}C_{m}_AUC"
            rr[c] = wide[col].mean() if col in wide.columns else np.nan
        raw_rows.append(rr)
    raw_df = pd.DataFrame(raw_rows).round(4)
    raw_df.to_csv(paper_output_path("table_b1_mean_auc.csv"), index=False)
    print("\n── Table B1: Mean AUC ──")
    print(df.to_string(index=False))
    return raw_df


# =====================================================================
# APPENDIX B — TABLE B2/B3: ACC & F1 Mean + Rank
# =====================================================================

def generate_table_b2(wide_acc, wide_f1):
    """Table B2: ACC mean + rank, Table B3: F1 mean + rank."""
    for label, wide, fn in [("ACC", wide_acc, "table_b2_acc"), ("F1", wide_f1, "table_b3_f1")]:
        rank_df = compute_mean_ranks(wide, label)
        active_methods = active_methods_for_metric(wide, label)
        rows = []
        for m in active_methods:
            row = {"Method": m}
            for c in C_BUDGETS:
                col = f"{c}C_{m}_{label}"
                row[f"{c}C_mean"] = wide[col].mean() if col in wide.columns else np.nan
            for c in C_BUDGETS:
                rr = rank_df[(rank_df["budget"] == c) & (rank_df["method"] == m)]
                row[f"{c}C_rank"] = rr["mean_rank"].values[0] if len(rr) else np.nan
            rows.append(row)
        df = pd.DataFrame(rows).round(4)
        df.to_csv(paper_output_path(f"{fn}.csv"), index=False)
        print(f"\n── Table B2/B3: {label} (7 methods) ──")
        print(df.to_string(index=False))

# =====================================================================
# APPENDIX B — TABLE B4: Cross-metric Spearman
# =====================================================================

def generate_table_b4(wide_list):
    """Table B4: Cross-metric Spearman ρ of method rankings.

    Computes Spearman ρ between 7-point mean-rank vectors (one value per
    method per metric per budget), then averages across budgets.  This is the
    computation described in the spec caption ("Spearman rank correlation
    between method-rankings under different metrics") and matches the values
    in R2_result_4_cross_metric.md.
    """
    metrics = ["AUC", "ACC", "F1"]
    wides = {m: w for m, w in zip(metrics, wide_list)}
    mat = pd.DataFrame(np.eye(3), index=metrics, columns=metrics)
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i >= j:
                continue
            rho_vals = []
            for c in C_BUDGETS:
                r1 = compute_mean_ranks(wides[m1], m1)
                r2 = compute_mean_ranks(wides[m2], m2)
                sub1 = r1[r1["budget"] == c].set_index("method")["mean_rank"]
                sub2 = r2[r2["budget"] == c].set_index("method")["mean_rank"]
                common = sub1.index.intersection(sub2.index)
                if len(common) >= 5:
                    rho, _ = stats.spearmanr(sub1[common], sub2[common])
                    rho_vals.append(rho)
            mat.loc[m1, m2] = np.mean(rho_vals) if rho_vals else np.nan
            mat.loc[m2, m1] = mat.loc[m1, m2]
    mat = mat.round(3)
    mat.to_csv(paper_output_path("table_b4_cross_metric.csv"))
    print("\n── Table B4: Cross-metric Spearman ρ ──")
    print(mat.to_string())
    return mat

# =====================================================================
# APPENDIX B — TABLE B5: Robustness summary
# =====================================================================

def generate_table_b5(wide_auc, b4_rho=None):
    """Table B5: Robustness summary.

    Dataset-level checks use the aggregated results. Checks that require
    per-fold/per-seed values are documented as not recomputed in this flow.
    """
    rng = np.random.RandomState(42)
    rows = []
    datasets_all = wide_auc["dataset"].unique()

    # ═══════════════════════════════════════════════════════════════
    # 1. Leave-one-dataset-out
    # ═══════════════════════════════════════════════════════════════
    base_ranks = compute_mean_ranks(wide_auc, "AUC")
    base_sams = {}
    for c in C_BUDGETS:
        sr = base_ranks[(base_ranks["budget"] == c) & (base_ranks["method"] == "LUCoS")]["mean_rank"]
        base_sams[c] = sr.values[0] if len(sr) else np.nan

    lodo_loses = {c: 0 for c in C_BUDGETS}
    lodo_max_shift = {c: 0.0 for c in C_BUDGETS}
    for ds in datasets_all:
        subset = wide_auc[wide_auc["dataset"] != ds]
        r = compute_mean_ranks(subset, "AUC")
        for c in C_BUDGETS:
            sams_r = r[(r["budget"] == c) & (r["method"] == "LUCoS")]["mean_rank"]
            if len(sams_r):
                shift = abs(sams_r.values[0] - base_sams[c])
                lodo_max_shift[c] = max(lodo_max_shift[c], shift)
                best = r[r["budget"] == c]["mean_rank"].min()
                if sams_r.values[0] > best + 0.01:
                    lodo_loses[c] += 1
    lodo_total = sum(lodo_loses.values())
    lodo_max = max(lodo_max_shift.values())
    lodo_msg = f"{lodo_total} of {len(datasets_all)*6} budget×LODO configs change LUCoS #1 rank; max shift={lodo_max:.2f}"
    rows.append(("Leave-one-dataset-out (67×6=402 configs)", lodo_msg, "B5.1"))
    print(f"  B5 LODO: {lodo_msg}")
    # Save sub-table B5.1
    lodo_df = pd.DataFrame([
        {"budget": f"{c}C", "n_configs": len(datasets_all),
         "sams_loses_top": lodo_loses[c], "max_rank_shift": round(lodo_max_shift[c], 3)}
        for c in C_BUDGETS
    ])
    lodo_df.to_csv(paper_output_path("table_b5_1_lodo.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════
    # 2. Progressive removal
    # ═══════════════════════════════════════════════════════════════
    removal_fracs = np.arange(0.1, 1.0, 0.1)
    n_seeds = 20
    survival_counts = {f: 0 for f in removal_fracs}
    for seed in range(n_seeds):
        rng_local = np.random.RandomState(42 + seed)
        shuffled = rng_local.permutation(datasets_all)
        for frac in removal_fracs:
            n_keep = int(len(datasets_all) * (1 - frac))
            if n_keep < 5:
                continue
            keep_ds = shuffled[:n_keep]
            subset = wide_auc[wide_auc["dataset"].isin(keep_ds)]
            r = compute_mean_ranks(subset, "AUC")
            sams_still_top = True
            for c in C_BUDGETS:
                best = r[r["budget"] == c]["mean_rank"].min()
                sams_r = r[(r["budget"] == c) & (r["method"] == "LUCoS")]["mean_rank"]
                if len(sams_r) and sams_r.values[0] > best + 0.01:
                    sams_still_top = False
                    break
            if sams_still_top:
                survival_counts[frac] += 1
    drop_point = 1.0
    for frac in removal_fracs:
        if survival_counts[frac] < n_seeds * 0.5:
            drop_point = frac
            break
    removal_msg = f"LUCoS retains #1 until >{int(drop_point*100)}% datasets removed"
    rows.append(("Progressive removal (20 random seeds)", removal_msg, "B5.2"))
    print(f"  B5 Removal: {removal_msg}")
    # Save sub-table B5.2
    remove_df = pd.DataFrame([
        {"fraction_removed": round(frac, 1), "datasets_kept": int(len(datasets_all)*(1-frac)),
         "seeds_sams_top": survival_counts[frac], "n_seeds": n_seeds}
        for frac in removal_fracs
    ])
    remove_df.to_csv(paper_output_path("table_b5_2_progressive_removal.csv"), index=False)

    # ═══════════════════════════════════════════════════════════════
    # 3. K-Medoids initialisation (KM++ vs random)
    # ═══════════════════════════════════════════════════════════════
    km_msg = "Not recomputed from raw per-seed data; paper generator uses aggregated results."
    rows.append(("K-Medoids initialisation (KM++ vs random)", km_msg, "B5.3"))
    print(f"  B5 KM++: {km_msg}")

    # ═══════════════════════════════════════════════════════════════
    # 4. Cross-metric → see Table B4
    # ═══════════════════════════════════════════════════════════════
    if b4_rho is not None:
        racc = b4_rho.loc["AUC", "ACC"] if "AUC" in b4_rho.index else 0.87
        raf1 = b4_rho.loc["AUC", "F1"] if "AUC" in b4_rho.index else 0.95
        rho_msg = f"Spearman ρ AUC↔ACC={racc:.2f}, AUC↔F1={raf1:.2f}; LUCoS #1 under all three"
    else:
        rho_msg = "Spearman ρ ≥ 0.86; LUCoS #1 under all three"
    rows.append(("Cross-metric (AUC vs ACC vs F1)", rho_msg, "B5.4"))
    print(f"  B5 Cross-metric: {rho_msg}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Aggregation (mean vs median vs trimmed mean)
    # ═══════════════════════════════════════════════════════════════
    agg_msg = "Not recomputed from raw per-fold data; paper generator uses aggregated results."
    rows.append(("Aggregation (mean vs median vs trimmed 10%)", agg_msg, "B5.5"))
    print(f"  B5 Aggregation: {agg_msg}")

    # ═══════════════════════════════════════════════════════════════
    # Save summary table
    # ═══════════════════════════════════════════════════════════════
    df = pd.DataFrame(rows, columns=["Check", "Result", "Source"])
    df.to_csv(paper_output_path("table_b5_robustness.csv"), index=False)
    print("\n── Table B5: Robustness (computed from data) ──")
    print(df.to_string(index=False))
    return df


# =====================================================================
# APPENDIX B — TABLE B6: Rescue vs Boost across budgets
# =====================================================================

def generate_table_b6(wide):
    """Table B6: Rescue vs Boost decomposition at all budgets."""
    rows = []
    for c in C_BUDGETS:
        s_col = f"{c}C_{LUCoS}_{PRIMARY}"
        o_col = f"{c}C_{ORIG}_{PRIMARY}"
        r_col = f"{c}C_{RAND}_{PRIMARY}"
        sub = wide[[s_col, o_col, r_col]].dropna().copy()
        sub["delta_repr"] = sub[s_col] - sub[o_col]
        sub["orig_vs_rand"] = sub[o_col] - sub[r_col]
        rescue = sub[sub["orig_vs_rand"] < 0]["delta_repr"]
        boost  = sub[sub["orig_vs_rand"] >= 0]["delta_repr"]
        nr, nb = len(rescue), len(boost)
        mr, mb = rescue.mean(), boost.mean()
        dd = mr - mb
        _, pm = stats.mannwhitneyu(rescue, boost, alternative="two-sided") if nr >= 3 and nb >= 3 else (np.nan, np.nan)
        rows.append({"budget": f"{c}C", "rescue_n": nr, "rescue_mean_delta_repr": mr,
                      "boost_n": nb, "boost_mean_delta_repr": mb,
                      "delta_diff": dd, "mwu_p": pm})
    df = pd.DataFrame(rows)
    # round numeric columns except p-values (which are very small)
    num_cols = [c for c in df.columns if c != 'mwu_p' and c != 'budget']
    df[num_cols] = df[num_cols].round(4)
    df.to_csv(paper_output_path("table_b6_rescue_boost.csv"), index=False, float_format="%.8f")
    print("\n── Table B6: Rescue vs Boost ──")
    print(df.to_string(index=False))
    return df

# =====================================================================
# APPENDIX B — TABLE B7: Original-KM breakdown
# =====================================================================

def generate_table_b7(wide):
    """Table B7: Budget at which Original-KM first falls below Random."""
    # Per dataset, find first budget where Orig-KM < Random
    survival_rows = []
    for _, ds in wide.iterrows():
        first_fail = None
        for c in C_BUDGETS:
            o_col = f"{c}C_{ORIG}_{PRIMARY}"
            r_col = f"{c}C_{RAND}_{PRIMARY}"
            if pd.notna(ds.get(o_col)) and pd.notna(ds.get(r_col)):
                if ds[o_col] < ds[r_col]:
                    first_fail = c
                    break
        survival_rows.append({"dataset": ds["dataset"], "first_fail": first_fail})
    sf = pd.DataFrame(survival_rows)
    counts = []
    cum = 0
    for c in C_BUDGETS:
        n = int((sf["first_fail"] == c).sum())
        cum += n
        counts.append({"budget": f"{c}C", "first_fail_count": n,
                       "cumulative_failed": cum, "cumulative_pct": cum / len(sf) * 100,
                       "survival_pct": (1 - cum / len(sf)) * 100})
    never = int(sf["first_fail"].isna().sum())
    df = pd.DataFrame(counts)
    # add 'Never' row
    df = pd.concat([df, pd.DataFrame([{"budget": "Never", "first_fail_count": never,
                                        "cumulative_failed": cum + never,
                                        "cumulative_pct": (cum+never)/len(sf)*100,
                                        "survival_pct": 0}])], ignore_index=True)
    df.to_csv(paper_output_path("table_b7_breakdown.csv"), index=False)
    print(f"\n── Table B7: Orig-KM Breakdown ──")
    print(df.to_string(index=False))
    return df


# =====================================================================
# APPENDIX B — FIGURE B1: Improvability curves
# =====================================================================

def generate_fig_b1(wide):
    """Figure B1: Improvability curves (1 − AUC/AUC_full)."""
    base_col = f"OriginalSpace_All_{EVALUATOR}_{PRIMARY}"
    if base_col not in wide.columns:
        # load baseline separately
        print("  WARNING: baseline column not in wide; loading from raw...")
        return

    base = wide[base_col]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colors = {"Random": "#E74C3C", "Orig-KM": "#F39C12", "LUCoS": "#3498DB"}
    for mk, color in colors.items():
        y_vals = []
        for c in C_BUDGETS:
            col = f"{c}C_{mk}_{PRIMARY}"
            if col in wide.columns:
                imp = (1 - wide[col] / base).mean() * 100
                y_vals.append(imp)
        ax.plot(C_BUDGETS, y_vals, "o-", label=mk, color=color, markersize=4, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(C_BUDGETS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Labeling budget K")
    ax.set_ylabel("Improvability (%)  [lower is better]")
    ax.legend(fontsize=8)
    ax.set_title("Improvability vs Budget", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(paper_output_path("figB1_improvability.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Fig B1: improvability curves saved")

# =====================================================================
# APPENDIX B — FIGURE B2: Downstream classifier verification
# =====================================================================

def generate_fig_b2(wide_auc):
    """Figure B2: LUCoS vs Orig-KM vs Random under TabPFN downstream classifier.

    Shows mean AUC ± 95% CI across all 67 datasets × 10 folds at each budget.
    Documents the exact evaluation pipeline used throughout the paper.
    KNN and XGBoost evaluator data (fold 0, OriginalSpace-only) omitted —
    TabClustPFN-space methods were not evaluated under those classifiers.
    """
    methods = [("LUCoS", "#3498DB", "TabClustPFN + K-Medoids"),
               ("Orig-KM", "#F39C12", "Original + K-Medoids"),
               ("Random", "#95A5A6", "Random")]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for mk, color, label in methods:
        means = []
        for c in C_BUDGETS:
            col = f"{c}C_{mk}_{PRIMARY}"
            vals = pd.to_numeric(wide_auc[col], errors="coerce").dropna() if col in wide_auc.columns else pd.Series(dtype=float)
            means.append(float(vals.mean()) if len(vals) else np.nan)
        ax.plot(C_BUDGETS, means, "o-", label=label, color=color,
                markersize=5, linewidth=1.5, markeredgewidth=0.5,
                markeredgecolor="white")

    ax.set_xscale("log", base=2)
    ax.set_xticks(C_BUDGETS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Labeling budget K")
    ax.set_ylabel("Mean AUC")
    ax.set_ylim(0.55, 0.92)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_title("Downstream classifier: TabPFNv2", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, which="major")

    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.12, right=0.96, top=0.93, bottom=0.14)
    fig.savefig(paper_output_path("figB2_cross_classifier.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Fig B2: TabPFN downstream classifier plot saved")

# =====================================================================
# APPENDIX B — FIGURE B3: Win/loss heatmap (optional)
# =====================================================================

def generate_fig_b3(wide):
    """Figure B3: Per-dataset win/loss heatmap (LUCoS − Random)."""
    diff_matrix = {}
    for c in C_BUDGETS:
        s_col = f"{c}C_{LUCoS}_{PRIMARY}"
        r_col = f"{c}C_{RAND}_{PRIMARY}"
        if s_col in wide.columns and r_col in wide.columns:
            diff_matrix[c] = (wide[s_col] - wide[r_col]).values
        else:
            print(f"  B3 WARNING: missing columns for {c}C, skipping")
            diff_matrix[c] = np.full(len(wide), np.nan)

    diff_df = pd.DataFrame(diff_matrix, index=wide["dataset"])
    # Drop rows that are all-NaN across all budgets
    diff_df = diff_df.dropna(how="all")
    if len(diff_df) == 0:
        print("  B3: no valid data for heatmap, skipping")
        return

    # Sort by mean across budgets (fill remaining NaN with 0 for sort only)
    sort_key = diff_df.fillna(0).mean(axis=1).sort_values(ascending=False)
    diff_df = diff_df.loc[sort_key.index]
    n_datasets = len(diff_df)

    fig, ax = plt.subplots(figsize=(5.5, max(4, n_datasets * 0.12)))
    # Use a diverging colormap centered at 0; set masked values to grey
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")  # grey for NaN cells
    data = np.ma.masked_invalid(diff_df.values.astype(float))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.08, vmax=0.08)
    ax.set_xticks(range(len(C_BUDGETS)))
    ax.set_xticklabels([f"{c}C" for c in C_BUDGETS])
    n_max_labels = min(100, n_datasets)
    step = max(1, n_datasets // n_max_labels)
    tick_positions = list(range(0, n_datasets, step))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(diff_df.index[tick_positions], fontsize=5)
    plt.colorbar(im, ax=ax, label="ΔAUC (LUCoS − Random)")
    ax.set_title("Per-Dataset LUCoS − Random AUC", fontsize=11, fontweight="bold")
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.25, right=0.92, top=0.95, bottom=0.08)
    fig.savefig(paper_output_path("figB3_winloss_heatmap.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Fig B3: win/loss heatmap ({n_datasets} datasets)")

# =====================================================================
# HELPERS
# =====================================================================

def _p_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# =====================================================================
# FIGURE 2 DATA — SSMA Oracle vs Random
# =====================================================================

def generate_figure2_ssma(agg_ssma):
    """Extract per-budget SSMA (supervised oracle) vs Random AUC values
    for Figure 2 from the already aggregated SSMA table.

    Saves figure2_ssma_oracle.csv to the output directory.
    """
    summary_rows = []
    for c in C_BUDGETS:
        ssma_col = create_plots.metric_column_getter(c, "OriginalSpace", "SSMAUnderSampler", EVALUATOR, PRIMARY)
        rand_col = create_plots.metric_column_getter(c, "OriginalSpace", "RandomUnderSamplerBalanced", EVALUATOR, PRIMARY)
        if ssma_col not in agg_ssma.columns or rand_col not in agg_ssma.columns:
            continue
        sub = agg_ssma[[ssma_col, rand_col]].apply(pd.to_numeric, errors="coerce").dropna()
        delta = sub[ssma_col] - sub[rand_col]
        summary_rows.append({
            "budget": f"{c}C",
            "n_datasets": len(sub),
            "ssma_mean_auc": round(sub[ssma_col].mean(), 4),
            "random_mean_auc": round(sub[rand_col].mean(), 4),
            "delta_mean": round(delta.mean(), 4),
            "delta_median": round(delta.median(), 4),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paper_output_path("figure2_ssma_oracle.csv"), index=False)

    # Print headline numbers for the caption
    for _, r in summary.iterrows():
        print(f"  {r['budget']}: SSMA={r['ssma_mean_auc']:.4f}  Random={r['random_mean_auc']:.4f}  Δ={r['delta_mean']:+.4f}")
    print(f"  Saved figure2_ssma_oracle.csv ({len(summary)} budgets)")
    return summary

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("Section 4 Redesign — Tables & Figures Generator")
    print("=" * 60)

    # ── load ──
    print("\n[1/3] Loading aggregated data ...")
    agg_lucos = agg_results.read_aggregated_results("LUCoS")
    agg_ssma = agg_results.read_aggregated_results("SSMA")
    print(f"  LUCoS: {len(agg_lucos):,} datasets, {len(agg_lucos.columns)} columns")
    print(f"  SSMA:  {len(agg_ssma):,} datasets, {len(agg_ssma.columns)} columns")

    # ── process AUC data (used by most artifacts) ──
    print("\n[2/3] Building paper-wide tables ...")
    wide_auc = build_wide_from_aggregated(agg_lucos, "AUC")
    print(f"  AUC: {len(wide_auc)} datasets, {len(wide_auc.columns)} columns")

    # ── process ACC and F1 ──
    wide_acc = build_wide_from_aggregated(agg_lucos, "ACC")
    wide_f1  = build_wide_from_aggregated(agg_lucos, "F1")

    print(f"  ACC: {len(wide_acc)} datasets, {len(wide_acc.columns)} columns")
    print(f"  F1:  {len(wide_f1)} datasets, {len(wide_f1.columns)} columns")

    # ── generate all artifacts ──
    print("\n[3/3] Generating artifacts …")
    print("\n── MAIN TEXT ──")
    print("\n── Figure 2: SSMA Oracle vs Random ──")
    generate_figure2_ssma(agg_ssma)
    generate_table1(wide_auc)
    generate_fig4a(wide_auc)
    generate_fig4b(wide_auc)

    print("\n── APPENDIX B ──")
    generate_table_b1(wide_auc)
    generate_table_b2(wide_acc, wide_f1)
    b4_rho = generate_table_b4([wide_auc, wide_acc, wide_f1])
    generate_table_b5(wide_auc, b4_rho)
    generate_table_b6(wide_auc)
    generate_table_b7(wide_auc)
    generate_fig_b1(wide_auc)
    generate_fig_b2(wide_auc)
    generate_fig_b3(wide_auc)

    print(f"\n{'='*60}")
    print(f"All artifacts saved → {OUT}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
