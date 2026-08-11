"""Publication-quality plot helpers: heatmaps, boxplots, p-value grids, and external-validation bar plots."""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import seaborn as sns
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 18,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.frameon': False,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# ============================================================================
# P-VALUE HEATMAP
# ============================================================================

def plot_pvalue_heatmap(pvalue_matrix, title="P-value Heatmap",
                        alpha_levels=[0.0, 0.001, 0.01, 0.05, 1.0],
                        colors=["#00441b", "#238b45", "#99d8c9", "#fee0d2"],
                        figsize=(10, 8), save_path=None):
    """
    Create p-value heatmap with custom binning.

    Args:
        pvalue_matrix: Symmetric matrix of p-values
        title: Plot title
        alpha_levels: Significance level boundaries
        colors: Colors for each bin
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    # Ensure values are numeric — warn if any coercion occurs
    pvalue_matrix = pvalue_matrix.copy()
    numeric_matrix = pvalue_matrix.apply(pd.to_numeric, errors='coerce')
    coerced = numeric_matrix.isna() & pvalue_matrix.notna()
    if coerced.any().any():
        bad_cells = [
            (pvalue_matrix.index[i], pvalue_matrix.columns[j])
            for i in range(len(pvalue_matrix))
            for j in range(len(pvalue_matrix.columns))
            if coerced.iloc[i, j]
        ]
        warnings.warn(
            f"Non-numeric p-values at {len(bad_cells)} cell(s) coerced to 1.0 "
            f"(non-significant): {bad_cells[:5]}"
        )
    pvalue_matrix = numeric_matrix.fillna(1.0).astype(np.float64)
    
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(alpha_levels, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pvalue_matrix.values, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(pvalue_matrix.columns)))
    ax.set_yticks(range(len(pvalue_matrix.index)))
    ax.set_xticklabels(pvalue_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(pvalue_matrix.index)

    ax.tick_params(labelsize=12)

    cbar = plt.colorbar(im, ax=ax, boundaries=alpha_levels, ticks=alpha_levels)
    cbar.set_label('p-value', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


# ============================================================================
# GENERAL HEATMAP
# ============================================================================

def plot_heatmap(data, title="Heatmap", xlabel=None, ylabel=None,
                 cmap="RdYlGn", annot=False, fmt=".3f", figsize=(10, 8),
                 vmin=None, vmax=None, cbar_label=None, save_path=None):
    """
    Create general heatmap.

    Args:
        data: DataFrame or 2D array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap
        annot: Show annotations
        fmt: Annotation format
        figsize: Figure size
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cbar_label: Colorbar label
        save_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': cbar_label} if cbar_label else None,
        ax=ax
    )

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


# ============================================================================
# BOX PLOTS
# ============================================================================

def plot_boxplots(data, x, y, hue=None, title="Box Plot",
                 xlabel=None, ylabel=None, figsize=(12, 6),
                 palette="Set2", order=None, save_path=None):
    """
    Create box plots.

    Args:
        data: DataFrame
        x: X-axis column
        y: Y-axis column
        hue: Grouping variable
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        palette: Color palette
        order: Order of categories
        save_path: Path to save figure
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Only pass palette if hue is provided (avoids seaborn warning)
    boxplot_kwargs = {
        'data': data,
        'x': x,
        'y': y,
        'hue': hue,
        'order': order,
        'ax': ax
    }
    
    # Add palette only when hue is provided
    if hue is not None:
        boxplot_kwargs['palette'] = palette
    
    sns.boxplot(**boxplot_kwargs)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.xticks(rotation=45, ha='right')
    if hue:
        plt.legend(title='', frameon=False, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fig, ax


# ============================================================================
# CONVENIENCE WRAPPER CLASS
# ============================================================================

class Visualizer:
    """Convenience wrapper for plotting functions."""
    
    def __init__(self, output_dir, dpi=300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        set_publication_style()
        if self.dpi != 300:
            plt.rcParams['savefig.dpi'] = self.dpi
    
    def plot_tukey_heatmap(self, pvalue_matrix, metric, descriptor, model_order=None):
        """
        Create and save Tukey p-value heatmap.

        Args:
            pvalue_matrix: Symmetric p-value matrix
            metric: Metric name
            descriptor: Descriptor name
            model_order: Order of models on axes

        Returns:
            Path to saved figure
        """
        if model_order is not None:
            # Keep only models present in the matrix, in the requested order
            order = [m for m in model_order if m in pvalue_matrix.index]
            pvalue_matrix = pvalue_matrix.reindex(index=order, columns=order)

        save_path = self.output_dir / f"{descriptor}_{metric}_tukey_pvalue_heatmap.png"

        plot_pvalue_heatmap(
            pvalue_matrix,
            title=f"Tukey HSD P-values: {descriptor} - {metric}",
            save_path=save_path
        )
        
        return save_path
    
    def plot_heatmap(self, df, metric, descriptor_order=None, model_order=None):
        """
        Create heatmap of metric across descriptors and models.
        
        Args:
            df: Master table DataFrame
            metric: Metric name (e.g., 'ROC_AUC')
            descriptor_order: Order of descriptors
            model_order: Order of models
            
        Returns:
            Path to saved figure or None if metric not found
        """
        # Try with _mean suffix first, then without
        metric_col = None
        if f"{metric}_mean" in df.columns:
            metric_col = f"{metric}_mean"
        elif metric in df.columns:
            metric_col = metric
        else:
            logger.info(f"Warning: Neither '{metric}_mean' nor '{metric}' found in DataFrame")
            logger.info(f"Available columns: {df.columns.tolist()}")
            save_path = self.output_dir / f"{metric}_heatmap_NOT_CREATED.png"
            return save_path
        
        pivot_data = df.pivot(
            index='Model',
            columns='Descriptor',
            values=metric_col
        )
        
        if descriptor_order:
            pivot_data = pivot_data.reindex(columns=descriptor_order)
        if model_order:
            pivot_data = pivot_data.reindex(index=model_order)
        
        display_name = {'ROC_AUC': 'AUROC'}.get(metric, metric)
        save_path = self.output_dir / f"{metric}_heatmap.png"

        plot_heatmap(
            pivot_data,
            title=f"{display_name} Comparison",
            xlabel="Descriptor",
            ylabel="Model",
            cmap="RdYlGn",
            annot=True,
            fmt=".3f",
            figsize=(10, 8),
            vmin=pivot_data.min().min(),
            vmax=pivot_data.max().max(),
            cbar_label=display_name,
            save_path=save_path
        )
        
        return save_path
    
    def plot_boxplot(self, df, descriptor, metric, model_order=None):
        """
        Create and save box plot for descriptor-metric combination.
        
        Args:
            df: DataFrame with per-fold results
            descriptor: Descriptor name
            metric: Metric name
            model_order: Order of models on x-axis
            
        Returns:
            Path to saved figure
        """
        df_sub = df[df['Descriptor'] == descriptor].copy()
        
        if metric not in df_sub.columns:
            logger.info(f"Warning: {metric} not in DataFrame")
            return None
        
        display_name = {'ROC_AUC': 'AUROC'}.get(metric, metric)
        save_path = self.output_dir / f"{descriptor}_{metric}_boxplot.png"

        plot_boxplots(
            data=df_sub,
            x='Model',
            y=metric,
            title=f"{descriptor} - {display_name} Distribution",
            xlabel="Model",
            ylabel=display_name,
            figsize=(14, 6),
            palette="Set2",
            order=model_order,
            save_path=save_path
        )
        
        return save_path
    
    def plot_comparison(self, df, metrics=None, descriptor_order=None, model_order=None):
        """
        Create comparison plot (grouped minmax heatmap).
        
        Args:
            df: Master table DataFrame
            metrics: List of metrics to include
            descriptor_order: Order of descriptors
            model_order: Order of models
            
        Returns:
            Path to saved figure
        """
        if metrics is None:
            metrics = ['ROC_AUC', 'MCC', 'GMean']
        
        save_path = self.output_dir / "comparison_minmax_heatmap.png"
        
        grouped_minmax_heatmap(
            df,
            metrics=metrics,
            descriptor_order=descriptor_order if descriptor_order else df['Descriptor'].unique().tolist(),
            model_order=model_order,
            save_path=save_path
        )
        
        return save_path
    
    def create_combined_boxplot(self, df, metric, descriptor_order=None, model_order=None):
        """
        Create combined boxplot showing all descriptors together.
        
        Args:
            df: DataFrame with per-fold results
            metric: Metric to plot
            descriptor_order: Order of descriptors
            model_order: Order of models
            
        Returns:
            Path to saved figure
        """
        save_path = self.output_dir / f"all_descriptors_{metric.lower()}_boxplot.png"
        
        return create_combined_boxplot(
            df,
            metric,
            descriptor_order=descriptor_order,
            model_order=model_order,
            save_path=save_path
        )
    
# ============================================================================
# GROUPED MIN-MAX HEATMAP
# ============================================================================

def grouped_minmax_heatmap(
    df,
    metrics,
    descriptor_order=None,
    model_order=None,
    add_separators=True,
    cmap_name="viridis",
    save_path=None,
):
    """
    Create grouped min-max scaled heatmap.

    Args:
        df: DataFrame with columns ['Descriptor', 'Model'] + metrics
        metrics: list of metric column names to include as columns
        descriptor_order: list of descriptor names in desired block order
        model_order: list of model names in desired row order (same for each block)
        add_separators: if True, insert blank rows between descriptor blocks
        cmap_name: Colormap name
        save_path: Path to save figure
        
    Returns:
        None (displays and optionally saves figure)
    """
    # Remove '_mean' suffix from column names if present
    df = df.copy()
    df.columns = df.columns.str.removesuffix('_mean')
    
    subset = df[["Descriptor", "Model"] + metrics].copy()

    all_rows = []
    row_labels = []
    block_bounds = []  # (desc, start_idx, end_idx) *before* separator row

    for i, desc in enumerate(descriptor_order):
        block = subset[subset["Descriptor"] == desc].copy()
        if block.empty:
            continue

        if model_order is not None:
            block["Model"] = pd.Categorical(block["Model"], categories=model_order, ordered=True)
            block = block.sort_values("Model")

        block = block.dropna(subset=["Model"])

        start_idx = len(all_rows)
        for _, row in block.iterrows():
            all_rows.append(row[metrics].values)
            row_labels.append(row["Model"])
        end_idx = len(all_rows)
        block_bounds.append((desc, start_idx, end_idx))

        if add_separators and i < len(descriptor_order) - 1:
            all_rows.append([np.nan] * len(metrics))
            row_labels.append("")  # blank label for separator

    mm = pd.DataFrame(all_rows, columns=metrics)

    mins = mm.min(skipna=True)
    ranges = mm.max(skipna=True) - mins
    ranges[ranges == 0] = 1.0
    mm_scaled = (mm - mins) / ranges

    data = np.ma.masked_invalid(mm_scaled.values)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.grid(False)          # turn off grid if it was enabled
    
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    display_map = {'ROC_AUC': 'AUROC'}
    metric_labels = [display_map.get(m, m) for m in metrics]
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=14)

    for desc, start, end in block_bounds:
        mid = (start + end - 1) / 2.0
        ax.text(
            -2.6,
            mid,
            desc,
            va="center",
            ha="right",
            fontsize=12,
            rotation=90,
        )
    fig.subplots_adjust(left=0.22)
    # Optional: thicker horizontal lines at block boundaries (just above each block)
    # (Use start index; separator row gives additional white space)
    for desc, start, end in block_bounds[1:]:
        ax.axhline(start - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Min-max scaled metric", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================================
# COMBINED BOXPLOT
# ============================================================================

def create_combined_boxplot(df, metric, descriptor_order=None, model_order=None, save_path=None):
    """
    Create combined boxplot showing all descriptors together.

    Args:
        df: DataFrame with per-fold results (must have 'Descriptor' column)
        metric: Metric to plot
        descriptor_order: Order of descriptors (for legend)
        model_order: Order of models on x-axis
        save_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    set_publication_style()

    display_name = {'ROC_AUC': 'AUROC'}.get(metric, metric)

    if save_path is None:
        save_path = Path(f"all_descriptors_{metric.lower()}_boxplot.png")

    plot_boxplots(
        data=df,
        x='Model',
        y=metric,
        hue='Descriptor',
        title=f"{display_name} Across All Descriptors",
        xlabel="Model",
        ylabel=display_name,
        figsize=(16, 7),
        palette="Set2",
        order=model_order,
        save_path=save_path
    )
    
    logger.info(f"Saved combined boxplot to: {save_path}")
    return save_path


# ============================================================================
# TOP-10 MODELS STACKED BOXPLOT
# ============================================================================

def plot_top10_boxplot(df, save_path):
    """
    Stacked 3-panel boxplot for the top-10 descriptor+model combos by MCC.

    Ordering uses average percent rank of MCC across CV folds (same as CD diagram).
    Each panel shows one metric: MCC, AUROC, GMean. Boxes are coloured by descriptor.

    Args:
        df: Per-fold results DataFrame with Descriptor, Model, Repeat, Fold, MCC, ROC_AUC, GMean
        save_path: Output path for the figure

    Returns:
        Path to saved figure
    """
    import matplotlib.patches as mpatches

    df = df.copy()
    df["Combo"] = df["Descriptor"] + "/" + df["Model"]
    df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)

    pivot = df.pivot_table(index="Subject", columns="Combo", values="MCC", aggfunc="mean")
    avg_rank = pivot.rank(axis=1, pct=True).mean()
    order = avg_rank.sort_values(ascending=False).head(10).index.tolist()

    plot_df = df[df["Combo"].isin(order)]

    desc_color = {
        "Mordred": "#ff7f0e", "Morgan": "#2ca02c", "RDKit": "#d62728",
        "SMILES": "#9467bd", "MACCS": "#1f77b4",
        "ChemBERTa": "#8c564b", "MolFormer": "#17becf",
    }
    label_color = {lbl: desc_color.get(lbl.split("/")[0], "#888888") for lbl in order}

    metrics = ["MCC", "ROC_AUC", "GMean"]
    metric_labels = ["MCC", "AUROC", "GMean"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, constrained_layout=True)

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        data = [plot_df.loc[plot_df["Combo"] == lbl, metric].values for lbl in order]
        bp = ax.boxplot(
            data, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, linestyle="none", alpha=0.6),
            widths=0.55,
        )
        for patch, lbl in zip(bp["boxes"], order):
            patch.set_facecolor(label_color[lbl])
            patch.set_alpha(0.80)
        ax.set_ylabel(mlabel, fontsize=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(range(1, len(order) + 1))
    axes[-1].set_xticklabels(order, rotation=40, ha="right", fontsize=9)

    descriptors_in_plot = sorted({lbl.split("/")[0] for lbl in order})
    legend_patches = [mpatches.Patch(color=desc_color.get(d, "#888"), alpha=0.8, label=d)
                      for d in descriptors_in_plot]
    fig.legend(handles=legend_patches, title="Descriptor", loc="upper right",
               fontsize=9, title_fontsize=10)

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# EXTERNAL PERFORMANCE COMPARISON (GROUPED BAR CHART)
# ============================================================================

def plot_ext_performance_comparison(csv_path="ext_performance_compare.csv", save_path=None):
    """
    Grouped bar chart comparing best single model vs ensemble sizes.

    Each metric gets its own subplot with a zoomed y-axis so that small
    differences are visible.

    Args:
        csv_path: Path to ext_performance_compare.csv
        save_path: Directory or file path to save figure. If None, saves
                   in the same directory as csv_path.

    Returns:
        Path to saved PNG figure
    """
    set_publication_style()

    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path, header=[0, 1])

    # Flatten the multi-level header into clean model names
    model_names = []
    for c0, c1 in raw.columns:
        c0, c1 = str(c0).strip(), str(c1).strip()
        if "Unnamed" in c0 and "Unnamed" in c1:
            model_names.append("Metric")
        elif "Unnamed" in c1:
            model_names.append(c0)
        elif "Unnamed" in c0:
            model_names.append(c1)
        else:
            model_names.append(f"{c0} {c1}".strip())
    raw.columns = model_names

    df = raw.set_index("Metric")
    df = df.apply(pd.to_numeric, errors="coerce")

    models = df.columns.tolist()
    metrics = df.index.tolist()

    n_metrics = len(metrics)
    n_models = len(models)
    cols = 3
    rows = (n_metrics + cols - 1) // cols

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    x = np.arange(n_models)
    bar_width = 0.6

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols, idx % cols]
        values = df.loc[metric].values.astype(float)

        bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor="black",
                      linewidth=0.5)

        # Zoomed y-axis: pad around min/max
        v_min, v_max = np.nanmin(values), np.nanmax(values)
        v_range = v_max - v_min if v_max > v_min else 0.01
        pad = max(v_range * 1.5, 0.005)
        ax.set_ylim(v_min - pad, v_max + pad)

        # Value labels
        for bar, val in zip(bars, values):
            is_best = np.isclose(val, v_max)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad * 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if is_best else "normal")

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Hide unused subplots
    for idx in range(n_metrics, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("External Test Set Performance", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path is None:
        save_path = csv_path.parent / "ext_performance_comparison.png"
    else:
        save_path = Path(save_path)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved: {save_path}")
    logger.info(f"Saved: {pdf_path}")
    return save_path
