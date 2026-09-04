"""
multirun_rmse_heatmap.py

Combine RMSE (mean + HAC standard error) across N independent training runs
for a given model/interval, and plot it as a Test-vs-Validation heatmap in
the style of Figure 4 (row-wise best validation event boxed in green).

Why "combine" rather than just average
---------------------------------------
Each run's evaluation_output_constrained.txt row already carries a
within-run HAC-corrected uncertainty (RMSE_HAC_SE) that accounts for
temporal autocorrelation in that one run's residuals. Running N independent
trainings adds a second, separate source of uncertainty: run-to-run
variability from stochastic training (init, batch order, etc.).

For each (Test, Val) cell, across n_runs independent runs:

    combined_var = between_run_var / n_runs + mean(within_run_HAC_var) / n_runs

    combined_se  = sqrt(combined_var)

This is the same decomposition discussed earlier in the conversation:
uncertainty of the mean = (between-run variance)/n + (average within-run
variance)/n. With few runs (e.g. n=3), between_run_var is a noisy estimate
of its own, so treat the resulting CI as indicative rather than exact.

Usage
-----
    from multirun_rmse_heatmap import (
        load_multi_run_results, aggregate_metric_across_runs,
        build_heatmap_matrices, plot_rmse_heatmap,
    )

    df = load_multi_run_results([
        "run1/evaluation_output_constrained.txt",
        "run2/evaluation_output_constrained.txt",
        "run3/evaluation_output_constrained.txt",
    ])

    agg = aggregate_metric_across_runs(df, model="xLSTM", interval=5, metric="RMSE")
    mean_mat, se_mat = build_heatmap_matrices(agg, metric="RMSE")
    plot_rmse_heatmap(mean_mat, se_mat, title="xLSTM, Interval: 5s")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns


# ======================================================================
# 1. Load and concatenate multiple runs
# ======================================================================


def load_multi_run_results(paths: list[str], run_labels: list[str] | None = None) -> pd.DataFrame:
    """
    Load evaluation_output_constrained.txt (or _wo_noise.txt) from N
    independent runs and concatenate them, tagged with a 'run' column.
    """
    labels = run_labels or [f"run{i + 1}" for i in range(len(paths))]
    if len(labels) != len(paths):
        raise ValueError("run_labels must be the same length as paths")

    dfs = []
    for path, label in zip(paths, labels):
        df = pd.read_csv(path)
        df["run"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ======================================================================
# 2. Aggregate a metric (e.g. RMSE) across runs, combining uncertainty
# ======================================================================


def aggregate_metric_across_runs(
    df: pd.DataFrame,
    model: str,
    interval: int,
    metric: str = "RMSE",
    se_col: str | None = None,
) -> pd.DataFrame:
    """
    For a given model and interval, group by (Test, Val) and combine the
    metric's mean and uncertainty across runs.

    Returns a DataFrame with columns:
        Test, Val, {metric}_mean, {metric}_combined_se,
        {metric}_between_run_sd, n_runs
    """
    se_col = se_col or f"{metric}_HAC_SE"
    sub = df[(df["Model"] == model) & (df["Interval"] == interval)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for Model={model!r}, Interval={interval!r}")

    def _combine(group: pd.DataFrame) -> pd.Series:
        n_runs = len(group)
        mean_val = group[metric].mean()
        between_var = group[metric].var(ddof=1) if n_runs > 1 else 0.0
        within_var_mean = (group[se_col] ** 2).mean()
        combined_var = between_var / n_runs + within_var_mean / n_runs
        return pd.Series(
            {
                f"{metric}_mean": mean_val,
                f"{metric}_combined_se": np.sqrt(combined_var),
                f"{metric}_between_run_sd": np.sqrt(between_var),
                "n_runs": n_runs,
            }
        )

    agg = sub.groupby(["Test", "Val"]).apply(_combine).reset_index()
    return agg


# ======================================================================
# 3. Reshape into Test x Val matrices (+ "avg" column) for the heatmap
# ======================================================================


def build_heatmap_matrices(agg: pd.DataFrame, metric: str = "RMSE") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivot the aggregated long-format table into square Test x Val matrices
    of the mean and combined SE, with an added 'avg' column (row-wise mean
    across validation events, diagonal excluded automatically since it's
    never present in the data).
    """
    tests = sorted(agg["Test"].unique())
    vals = sorted(agg["Val"].unique())

    mean_mat = pd.DataFrame(index=tests, columns=vals, dtype=float)
    se_mat = pd.DataFrame(index=tests, columns=vals, dtype=float)
    for _, row in agg.iterrows():
        mean_mat.loc[row["Test"], row["Val"]] = row[f"{metric}_mean"]
        se_mat.loc[row["Test"], row["Val"]] = row[f"{metric}_combined_se"]

    mean_mat["avg"] = mean_mat[vals].mean(axis=1, skipna=True)
    se_mat["avg"] = np.sqrt((se_mat[vals] ** 2).mean(axis=1, skipna=True))

    return mean_mat, se_mat


# ======================================================================
# 4. Plot: annotated heatmap with row-wise best (green box), avg column
# ======================================================================


def cache_heatmap_results(mean_mat: pd.DataFrame, se_mat: pd.DataFrame, cache_path: str) -> None:
    """
    Save one plot's aggregated mean/SE matrices (Test x Val, incl. 'avg'
    column) to a single .npz file, so repeated aggregation across runs
    doesn't need to be redone just to re-plot.

    mean_mat and se_mat must share the same index/columns (as produced by
    build_heatmap_matrices).
    """
    if not mean_mat.index.equals(se_mat.index) or not mean_mat.columns.equals(se_mat.columns):
        raise ValueError("mean_mat and se_mat must share the same index/columns")

    np.savez(
        cache_path,
        mean=mean_mat.to_numpy(dtype=float),
        se=se_mat.to_numpy(dtype=float),
        index=mean_mat.index.to_numpy(),
        columns=mean_mat.columns.to_numpy(),
    )


def plot_rmse_heatmap(
    mean_mat: pd.DataFrame,
    se_mat: pd.DataFrame,
    title: str = "",
    ax: plt.Axes | None = None,
    cmap: str = "Oranges",
    vmax: float | None = None,
    highlight_row_min: bool = True,
) -> plt.Axes:
    """
    Draw a Test-vs-Validation heatmap of mean +/- combined SE, matching the
    style of Figure 4: diagonal left blank, a black separator before the
    'avg' column, and a green box around the best (lowest-mean) validation
    event per test row (avg column excluded from that comparison).
    """
    vals = [c for c in mean_mat.columns if c != "avg"]
    display_cols = vals + ["avg"]

    data = mean_mat[display_cols].astype(float)
    annot = pd.DataFrame(index=data.index, columns=data.columns, dtype=object)
    for r in data.index:
        for c in display_cols:
            m, s = mean_mat.loc[r, c], se_mat.loc[r, c]
            annot.loc[r, c] = "" if pd.isna(m) else f"{m:.2f}\n\u00b1\n{s:.2f}"

    mask = data.isna()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    if vmax is None:
        vmax = np.nanmax(data.values)

    sns.heatmap(
        data,
        mask=mask,
        annot=annot.values,
        fmt="",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Root Mean Squared Error"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 6},
    )
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
    ax.set_xlabel("Validation (Julday)")
    ax.set_ylabel("Test (Julday)")

    # Separator line before the 'avg' column
    ax.axvline(len(vals), color="black", linewidth=2)

    # Green box around the row-wise minimum (excluding 'avg')
    if highlight_row_min:
        for i, r in enumerate(data.index):
            row_vals = mean_mat.loc[r, vals].dropna()
            if row_vals.empty:
                continue
            min_col = row_vals.idxmin()
            j = vals.index(min_col)
            ax.add_patch(patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="green", linewidth=3))

    return ax


# ======================================================================
# 5. Convenience: full pipeline for one model/interval
# ======================================================================


def compile_and_plot(
    paths: list[str],
    model: str,
    interval: int,
    metric: str = "RMSE",
    run_labels: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cache_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Axes]:
    """One-call pipeline: load runs -> aggregate -> pivot -> plot."""
    df = load_multi_run_results(paths, run_labels)
    agg = aggregate_metric_across_runs(df, model=model, interval=interval, metric=metric)
    mean_mat, se_mat = build_heatmap_matrices(agg, metric=metric)
    cache_heatmap_results(mean_mat, se_mat, cache_path)
    ax = plot_rmse_heatmap(
        mean_mat,
        se_mat,
        title=title or f"{model}, Interval: {interval}s",
        ax=ax,
    )
    return mean_mat, se_mat, ax


# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    run_paths = [
        "/mnt/user-data/uploads/evaluation_output_constrained.txt",
        # "/path/to/run2/evaluation_output_constrained.txt",
        # "/path/to/run3/evaluation_output_constrained.txt",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    compile_and_plot(run_paths, model="xLSTM", interval=5, ax=axes[0])
    compile_and_plot(run_paths, model="LSTM", interval=5, ax=axes[1])
    fig.tight_layout()
    # fig.savefig("/mnt/user-data/outputs/rmse_multirun_heatmap.png", dpi=200)
    print("Saved heatmap.")
