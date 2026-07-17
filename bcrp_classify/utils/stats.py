"""RM-ANOVA, Tukey HSD, and Conover-Friedman CD diagrams for CV model comparisons."""

import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import studentized_range
from statsmodels.stats.anova import AnovaRM

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SUBJECT PREPARATION
# ============================================================================

def prepare_subject_column(df):
    """
    Ensure Subject column exists, creating from Repeat+Fold if needed.
    
    Args:
        df: DataFrame with CV results
        
    Returns:
        DataFrame with Subject column
    """
    if "Subject" in df.columns:
        return df

    if {"Repeat", "Fold"}.issubset(df.columns):
        df = df.copy()
        df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)
        return df

    raise ValueError("No 'Subject' column found and cannot build from 'Repeat'+'Fold'.")


# ============================================================================
# REPEATED MEASURES ANOVA
# ============================================================================

def perform_repeated_measures_anova(df, depvar, subject, within):
    """
    Perform repeated measures ANOVA.
    
    Args:
        df: DataFrame with data
        depvar: Dependent variable (metric column)
        subject: Subject identifier column
        within: Within-subjects factor (model column)
        
    Returns:
        AnovaResults object with .anova_table attribute
        
    Example:
    """

    rm = AnovaRM(
        data=df,
        depvar=depvar,
        subject=subject,
        within=[within]
    )
    return rm.fit()


# ============================================================================
# TUKEY HSD POST-HOC TEST
# ============================================================================

def tukey_hsd_pairwise(df, metric_col, group_col, subject_col="Subject"):
    """
    Perform Tukey HSD pairwise comparisons for repeated measures.
    
    Args:
        df: DataFrame with CV results
        metric_col: Metric to compare (e.g., 'ROC_AUC')
        group_col: Grouping variable (e.g., 'Model')
        subject_col: Subject identifier (e.g., 'Subject')
        
    Returns:
        DataFrame with pairwise comparisons
    """
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    n_subjects = df[subject_col].nunique()

    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups for pairwise comparison, got {n_groups}"
        )
    if n_subjects < 2:
        raise ValueError(
            f"Need at least 2 subjects for repeated measures, got {n_subjects}"
        )

    group_means = df.groupby(group_col)[metric_col].mean()
    subject_means = df.groupby(subject_col)[metric_col].mean()
    grand_mean = df[metric_col].mean()

    # Subject x Treatment interaction residuals (proper RM error term)
    residuals = []
    for subj in df[subject_col].unique():
        subj_data = df[df[subject_col] == subj]
        for grp in groups:
            grp_vals = subj_data[subj_data[group_col] == grp][metric_col].values
            if len(grp_vals) > 0:
                residuals.append(
                    grp_vals[0] - subject_means[subj] - group_means[grp] + grand_mean
                )

    df_interaction = (n_subjects - 1) * (n_groups - 1)
    mse = np.sum(np.array(residuals) ** 2) / df_interaction if df_interaction > 0 else 0
    se = np.sqrt(mse / n_subjects)

    results = []
    for g1, g2 in combinations(groups, 2):
        diff = abs(group_means[g1] - group_means[g2])
        q_stat = diff / se if se > 0 else 0

        # Studentized range distribution p-value
        df_error = (n_subjects - 1) * (n_groups - 1)
        # scipy's studentized_range.sf gives upper tail probability (p-value)
        p_value = studentized_range.sf(q_stat, n_groups, df_error)

        results.append({
            "Group1": g1,
            "Group2": g2,
            "MeanDiff": diff,
            "Q-stat": q_stat,
            "p-value": p_value,
            "Significant": p_value < 0.05
        })

    return pd.DataFrame(results)


# ============================================================================
# PAIRWISE MATRIX
# ============================================================================

def create_pairwise_matrix(tukey_results, alpha=0.05):
    """
    Convert Tukey results to symmetric p-value matrix.
    
    Args:
        tukey_results: DataFrame from tukey_hsd_pairwise()
        alpha: Significance level (not used, kept for compatibility)
        
    Returns:
        Symmetric DataFrame of p-values
    """
    groups = sorted(set(tukey_results["Group1"].tolist() + tukey_results["Group2"].tolist()))
    pmat = pd.DataFrame(1.0, index=groups, columns=groups, dtype=float)

    for _, row in tukey_results.iterrows():
        g1, g2, pval = row["Group1"], row["Group2"], row["p-value"]
        # Ensure p-value is numeric
        pval = float(pval)
        pmat.loc[g1, g2] = pval
        pmat.loc[g2, g1] = pval

    for g in groups:
        pmat.loc[g, g] = 1.0

    return pmat


# ============================================================================
# COMPLETE STATISTICAL PIPELINE
# ============================================================================

def compare_models_rm_anova(df, metric_col, model_col, subject_col="Subject", alpha=0.05):
    """
    Complete statistical comparison pipeline: RM-ANOVA then Tukey HSD.

    Args:
        df: DataFrame with CV results
        metric_col: Metric to compare (e.g., 'ROC_AUC')
        model_col: Column with model names
        subject_col: Column with subject IDs
        alpha: Significance level

    Returns:
        dict with 'anova', 'tukey', and 'pairwise_matrix'
    """
    df = prepare_subject_column(df)

    anova_result = perform_repeated_measures_anova(
        df, depvar=metric_col, subject=subject_col, within=model_col
    )
    tukey_result = tukey_hsd_pairwise(df, metric_col, model_col, subject_col)
    pairwise_matrix = create_pairwise_matrix(tukey_result, alpha)

    return {
        "anova": anova_result,
        "tukey": tukey_result,
        "pairwise_matrix": pairwise_matrix,
    }


# ============================================================================
# CONVENIENCE WRAPPERS
# ============================================================================

class StatisticalAnalyzer:
    """Convenience wrapper for RM-ANOVA and Tukey HSD statistical functions."""
    
    def __init__(self, alpha=0.05):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def run_anova(self, df, descriptor, metric):
        """
        Run repeated measures ANOVA for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            
        Returns:
            ANOVA results object
        """
        df_sub = df[df['Descriptor'] == descriptor].copy()
        df_sub = prepare_subject_column(df_sub)
        anova_result = perform_repeated_measures_anova(
            df_sub,
            depvar=metric,
            subject='Subject',
            within='Model'
        )
        
        return anova_result
    
    def tukey_hsd(self, df, descriptor, metric):
        """
        Run Tukey HSD test for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            
        Returns:
            DataFrame with pairwise comparisons
        """
        df_sub = df[df['Descriptor'] == descriptor].copy()
        df_sub = prepare_subject_column(df_sub)
        tukey_result = tukey_hsd_pairwise(
            df_sub,
            metric_col=metric,
            group_col='Model',
            subject_col='Subject'
        )
        
        return tukey_result
    
    def run_analysis(self, df, descriptor, metric, alpha=None):
        """
        Run complete statistical analysis for one descriptor.
        
        Args:
            df: DataFrame with all CV results
            descriptor: Descriptor to analyze
            metric: Metric to analyze
            alpha: Significance level (uses instance alpha if not provided)
            
        Returns:
            dict with statistical results
        """
        if alpha is None:
            alpha = self.alpha
        
        df_sub = df[df['Descriptor'] == descriptor].copy()
        results = compare_models_rm_anova(
            df_sub, 
            metric_col=metric,
            model_col='Model',
            subject_col='Subject',
            alpha=alpha
        )
        
        return results
    
    def save_results(self, results, output_dir, descriptor, metric):
        """
        Save statistical results to files.
        
        Args:
            results: Results dict from compare_models_rm_anova()
            output_dir: Output directory
            descriptor: Descriptor name
            metric: Metric name
        """
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{descriptor}_{metric}"
        
        tukey_file = output_dir / f"{prefix}_tukey.csv"
        results['tukey'].to_csv(tukey_file, index=False)

        pmat_file = output_dir / f"{prefix}_pairwise_matrix.csv"
        results['pairwise_matrix'].to_csv(pmat_file)

        logger.info(f"\n{prefix} ANOVA:\n{results['anova']}")


# ============================================================================
# TOP-10 MODEL COMPARISON
# ============================================================================

def plot_sign_plot(pmat, metric_display, save_path, mean_scores=None):
    """
    Plot a Conover-Friedman significance heatmap using scikit-posthocs' sign_plot.

    Args:
        pmat: symmetric p-value DataFrame from Conover-Friedman post-hoc
        metric_display: display name for metric (e.g. "AUROC")
        save_path: output path for the figure
        mean_scores: Series of mean metric values for ordering (best at top-left)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    if mean_scores is not None:
        order = mean_scores.sort_values(ascending=False).index.tolist()
        order = [o for o in order if o in pmat.index]
    else:
        order = list(pmat.index)

    pmat_ordered = pmat.loc[order, order]

    n = len(order)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.95), max(7, n * 0.85)))

    heatmap_args = {
        "linewidths": 0.5,
        "linecolor": "0.7",
        "clip_on": True,
        "square": True,
    }

    sp.sign_plot(pmat_ordered, **heatmap_args, ax=ax)

    ax.set_title(
        f"Conover-Friedman Significance Plot — {metric_display}\n"
        f"(ordered by mean {metric_display}, best at top-left)",
        fontsize=12, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cd_diagram(df_top, metric_col, metric_display, save_path):
    """
    Single-metric Critical Difference diagram using the Pat Walters approach:
    wide-format pivot, percent ranks, Conover-Friedman post-hoc.

    Args:
        df_top: DataFrame with columns [Subject, Combo, <metric_col>]
        metric_col: metric column name (e.g. "MCC")
        metric_display: display name for metric (e.g. "MCC")
        save_path: output path for the figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    df_wide = df_top.pivot_table(
        index="Subject", columns="Combo",
        values=metric_col, aggfunc="mean",
    )
    avg_rank = df_wide.rank(axis=1, pct=True).mean()
    pc = sp.posthoc_conover_friedman(df_wide, p_adjust="holm")

    fig, ax = plt.subplots(figsize=(12, 5))
    sp.critical_difference_diagram(avg_rank, pc, ax=ax)
    ax.set_title(metric_display, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined_cd(df, metrics, save_path):
    """
    Plot critical difference diagrams for multiple metrics in a single figure.

    Uses the Pat Walters approach: pivot to wide format, then run
    Conover-Friedman on the wide-format DataFrame.

    Args:
        df: DataFrame with columns [Subject, Combo, <metric_cols>]
        metrics: dict mapping metric column names to display names
                 e.g. {"MCC": "MCC", "ROC_AUC": "AUROC", "GMean": "G-Mean"}
        save_path: output path for the combined figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    metric_cols = list(metrics.keys())
    n_metrics = len(metric_cols)
    figure, axes = plt.subplots(n_metrics, 1, sharex=True, sharey=False,
                                figsize=(16, 3 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for i, (metric_col, metric_display) in enumerate(metrics.items()):
        df_wide = df.pivot_table(
            index="Subject", columns="Combo",
            values=metric_col, aggfunc="mean",
        )
        avg_rank = df_wide.rank(axis=1, pct=True).mean()
        pc = sp.posthoc_conover_friedman(df_wide, p_adjust="holm")
        sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
        axes[i].set_title(metric_display)

    plt.tight_layout()
    figure.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


# ============================================================================
# CRITICAL DIFFERENCE DIAGRAMS
# ============================================================================

def select_top_n(df, metric_col, top_n=10):
    """
    Select top N descriptor+model combos by mean metric.

    Args:
        df: DataFrame with a "Combo" column (Descriptor + Model)
        metric_col: metric column name (e.g. "MCC")
        top_n: number of top combos to select

    Returns:
        (top_names, top_means): list of combo names and Series of their means
    """
    combo_means = (
        df.groupby("Combo")[metric_col]
        .mean()
        .sort_values(ascending=False)
    )
    top_names = combo_means.head(top_n).index.tolist()
    return top_names, combo_means.loc[top_names]


def run_tukey_top10(df, metric_col, top_n=10, alpha=0.05):
    """
    Run Tukey HSD on top-N combos using statsmodels pairwise_tukeyhsd.

    Args:
        df: DataFrame with columns [Combo, <metric_col>]
        metric_col: metric column name (e.g. "MCC")
        top_n: number of top combos to select
        alpha: significance level

    Returns:
        (tukey_result, melt_df, best_combo, anova_p, top_means)
    """
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    top_names, top_means = select_top_n(df, metric_col, top_n)

    df_top = df[df["Combo"].isin(top_names)].copy()

    melt_df = df_top[["Combo", metric_col]].copy()
    melt_df.columns = ["group", "value"]

    best_combo = top_means.index[0]

    groups = [v["value"].values for _, v in melt_df.groupby("group")]
    anova_p = f_oneway(*groups)[1] if len(groups) >= 2 else 1.0

    tukey = pairwise_tukeyhsd(
        endog=melt_df["value"],
        groups=melt_df["group"],
        alpha=alpha,
    )

    return tukey, melt_df, best_combo, anova_p, top_means


def plot_tukey_simultaneous(tukey, best_combo, metric_col, metric_display,
                            anova_p, top_means, out_dir, top_n, alpha=0.05,
                            melt_df=None, order=None):
    """
    Create the plot_simultaneous CI plot (blue/grey/red) for top-N combos.

    Blue  = the best method (highest mean)
    Grey  = methods NOT significantly different from the best
    Red   = methods significantly different from the best

    Args:
        tukey: statsmodels TukeyHSDResults object
        best_combo: name of best combo (comparison reference)
        metric_col: metric column name (e.g. "MCC")
        metric_display: display name for metric (e.g. "MCC")
        anova_p: ANOVA p-value
        top_means: Series of mean metric values for top combos
        out_dir: output directory (str or Path)
        top_n: number of top combos
        alpha: significance level
        melt_df: DataFrame with columns ["group", "value"] (required when order is given)
        order: list of combo names from best to worst (e.g. top-10 by MCC);
               controls y-axis order so best model appears at the top

    Returns:
        Path to saved figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from pathlib import Path

    out_dir = Path(out_dir)
    n_groups = len(top_means)
    fig_height = max(5, n_groups * 0.65)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    if order is not None and melt_df is not None:
        # Prefix group names with zero-padded rank so statsmodels' alphabetical
        # sort equals the desired MCC order; rank 01 (best) ends up at y=0
        # (bottom), then we invert so it appears at the top.
        rank_map = {name: f"{i+1:02d} {name}" for i, name in enumerate(order)}
        inv_map  = {v: k for k, v in rank_map.items()}
        df_r = melt_df.copy()
        df_r["group"] = df_r["group"].map(lambda x: rank_map.get(x, x))
        best_r = rank_map.get(best_combo, best_combo)
        tukey_r = pairwise_tukeyhsd(endog=df_r["value"], groups=df_r["group"], alpha=alpha)
        tukey_r.plot_simultaneous(comparison_name=best_r, ax=ax, figsize=(12, fig_height))
        ax.set_yticklabels(
            [inv_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()]
        )
    else:
        tukey.plot_simultaneous(comparison_name=best_combo, ax=ax, figsize=(12, fig_height))

    ax.set_title("")
    ax.invert_yaxis()   # best model (rank 01 / lowest y) moves to top
    ax.set_xlabel(metric_display, fontsize=14)
    ax.tick_params(labelsize=11)

    plt.tight_layout()

    out_path = out_dir / f"top{top_n}_{metric_col}_tukey_simultaneous.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path
