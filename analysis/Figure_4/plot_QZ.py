import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.patches import Rectangle


def highlight_row_mins(pivot, axis, cols):
    # Exclude 'avg' unconditionally: it's a summary column, never a real
    # validation event, and should never be the one boxed as the "best".
    cols_present = [c for c in cols if c in pivot.columns and c != "avg"]
    for row in pivot.index:
        row_vals = pivot.loc[row, cols_present]
        if row_vals.isna().all():
            continue
        # find the column with minimum value for this row (among cols_present)
        try:
            min_col = row_vals.astype(float).idxmin()
        except Exception as e:
            print(f"Exception: {e}")
            continue
        if pd.isna(min_col):
            continue
        # compute 0-based positions for rectangle
        x_pos = list(pivot.columns).index(min_col)
        y_pos = list(pivot.index).index(row)
        # add rectangle (non-filled red box)
        rect = Rectangle((x_pos, y_pos), 1, 1, fill=False, edgecolor="green", lw=2, zorder=50)
        axis.add_patch(rect)


# helper to format annotations "mean ± se" (empty string for NaNs)
def make_annot(mean_df, se_df, fmt_mean="{:.2f}", fmt_se="{:.2f}"):
    annot = mean_df.copy().astype(object)
    for i in mean_df.index:
        for j in mean_df.columns:
            m = mean_df.at[i, j]
            s = se_df.at[i, j]
            if pd.isna(m):
                annot.at[i, j] = ""
            else:
                annot.at[i, j] = f"{fmt_mean.format(m)}\n±\n{fmt_se.format(s if not pd.isna(s) else 0.0)}"
    return annot


def load_cached_heatmap_results(cache_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load mean/SE matrices previously saved with cache_heatmap_results()."""
    data = np.load(cache_path, allow_pickle=True)
    index = data["index"]
    columns = data["columns"]
    mean_mat = pd.DataFrame(data["mean"], index=index, columns=columns)
    se_mat = pd.DataFrame(data["se"], index=index, columns=columns)
    return mean_mat, se_mat


interval = 5
mapping = {161: 1, 172: 2, 182: 3, 196: 4, 207: 5, 223: 6, 232: 7}
mapping2 = {161: 1, 172: 2, 182: 3, 196: 4, 207: 5, 223: 6, 232: 7, "avg": 8}


plt.rcParams.update({"font.size": 7, "axes.formatter.limits": (-3, 6), "axes.formatter.use_mathtext": True})


from matplotlib import gridspec

fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[10, 10, 0.5])

subplot_idx = ["(a)", "(b)", "(c)", "(d)"]
cmap = "Oranges"
ax_list = {
    0: plt.subplot(gs[0, 0]),
    1: plt.subplot(gs[0, 1]),
    2: plt.subplot(gs[1, 0]),
    3: plt.subplot(gs[1, 1]),
    4: plt.subplot(gs[0, 2]),
    5: plt.subplot(gs[1, 2]),
}


idx = 0
metric = "RMSE"
for model in ["xLSTM", "LSTM"]:
    for time_shift in ["0", "1"]:  # ["0", "average", "1"]:
        ax = ax_list.get(idx)

        pivot_mean, pivot_se = load_cached_heatmap_results(f"./{model}-{time_shift}-5-{metric}.npz")
        # Create annotation with "mean ± se" format
        annot = make_annot(pivot_mean, pivot_se)

        # Create heatmap
        if metric == "MSE":
            vmin, vmax = 0, 7
            metricl = "Mean Squared Error"
            fs = 6
        elif metric == "RMSE":
            vmin, vmax = 0, 20
            metricl = "Root Mean Squared Error"
            fs = 6
        elif metric == "Peak_Diff":
            vmin, vmax = 0, 5
            metricl = "Peak Absolute Error"
            fs = 6
        elif metric == "ref_MSE":
            vmin, vmax = 0, 7
            metricl = "ref Mean Squared Error"
            fs = 6

        if idx in [1]:
            sns.heatmap(
                pivot_mean,
                annot=annot,
                fmt="",
                cmap=cmap,
                ax=ax,
                annot_kws={"fontsize": fs},
                vmin=vmin,
                vmax=vmax,
                cbar_ax=ax_list.get(4),
            )
            ax.collections[0].colorbar.ax.tick_params(labelsize=5, length=3)
            ax.collections[0].colorbar.set_label(label=metricl, size=7)
        elif idx in [3]:
            sns.heatmap(
                pivot_mean,
                annot=annot,
                fmt="",
                cmap=cmap,
                ax=ax,
                annot_kws={"fontsize": fs},
                vmin=vmin,
                vmax=vmax,
                cbar_ax=ax_list.get(5),
            )
            ax.collections[0].colorbar.ax.tick_params(labelsize=5, length=3)
            ax.collections[0].colorbar.set_label(label=metricl, size=7)
        else:
            sns.heatmap(
                pivot_mean,
                annot=annot,
                fmt="",
                cmap=cmap,
                ax=ax,
                annot_kws={"fontsize": fs},
                vmin=vmin,
                vmax=vmax,
                cbar=False,
            )

        highlight_row_mins(pivot_mean, ax, pivot_mean.columns.tolist())
        # Labels and ticks
        order = list(range(1, 9))
        inv_map = {v: k for k, v in mapping2.items()}
        labelx = [inv_map[i] for i in order]

        # for axis in ax:
        ax.tick_params(axis="both", which="major", labelsize=7, length=3)
        ax.set_xticklabels(labelx, rotation=0, ha="center", fontsize=7)
        ax.set_yticklabels(labelx[:7], rotation=0, fontsize=7)

        if idx in [0, 2]:
            ax.set_ylabel("Test (Julday)", fontsize=7, fontweight="bold")
        else:
            ax.set_ylabel("", fontsize=7, fontweight="bold")

        if idx in [2, 3]:
            ax.set_xlabel("Validation (Julday)", fontsize=7, fontweight="bold")
        else:
            ax.set_xlabel("", fontsize=7, fontweight="bold")

        ax.set_ylim(7, 0)
        ax.set_xlim(0, 8)
        ax.axvline(x=7, color="k", linewidth=1.5, zorder=20)
        ax.set_title(
            f"{subplot_idx[idx]} {model}, Time shift: {time_shift}-min", fontsize=7, fontweight="bold", loc="left"
        )
        # highlight_row_mins(pivot_mean, ax, pivot_mean.columns.tolist())
        idx = idx + 1

plt.tight_layout()
# plt.subplots_adjust(wspace=0.1)
# Make output dirs and save figures
png_path = "heatmap.png"
fig.savefig(png_path, dpi=600)
# plt.show()
plt.close(fig=fig)
