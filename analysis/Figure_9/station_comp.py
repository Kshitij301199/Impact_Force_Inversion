import os
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter, MultipleLocator
import seaborn as sns

plt.rcParams.update({
    'font.size': 7,
    'font.family': 'Arial',
    'legend.fontsize': 6,
    'axes.formatter.limits': (-3, 6),
    'axes.formatter.use_mathtext': True,
})

from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_plot2(ax0, ax1, startt, endt, cache_name, idx, x_gap):
    data = np.load(f"./fig9-cache/{cache_name}.npz")


    # --- Panel 0: seismic amplitude traces ---
    x0a, y0a = data["ax0_ILL11_x"], data["ax0_ILL11_y"]
    ax0.plot(x0a, y0a * 1e3, color='black', linewidth=1, label="ILL11", alpha=0.7, zorder=1)

    x0b, y0b = data["ax0_ILL12_x"], data["ax0_ILL12_y"]
    ax0.plot(x0b, y0b * 1e3, color='C0', linewidth=1, label="ILL12", alpha=0.7, zorder=3)

    x0c, y0c = data["ax0_ILL12_rescaled_x"], data["ax0_ILL12_rescaled_y"]
    ax0.plot(x0c, y0c * 1e3, color='C3', linewidth=1, label="ILL12-rescaled", alpha=0.7, zorder=2)

    ax0.set_ylim(-1.2, 1.2)
    ax0.set_ylabel("", fontsize=7, fontweight='bold')
    ax0.yaxis.get_offset_text().set_fontsize(7)
    ax0.yaxis.get_offset_text().set_fontweight('bold')

    ax0.grid(axis="both", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=0)



    # --- Panel 1: predicted impact force curves ---
    x1a, y1a = data["ax1_m1_ILL11_x"], data["ax1_m1_ILL11_y"]
    ax1.plot(x1a, y1a, color='black', linewidth=3, label="ILL11 xLSTM", alpha=0.7, zorder=3)
    
    x1b, y1b = data["ax1_m1_ILL12_x"], data["ax1_m1_ILL12_y"]
    ax1.plot(x1b, y1b, color='C0', linewidth=3, label="ILL12 xLSTM", alpha=0.7, zorder=3)

    x1c, y1c = data["ax1_m1_ILL12_rescaled_x"], data["ax1_m1_ILL12_rescaled_y"]
    ax1.plot(x1c, y1c, color='C3', linewidth=3, label="ILL12-rescaled xLSTM", alpha=0.7, zorder=3)



    x1d, y1d = data["ax1_m2_ILL11_x"], data["ax1_m2_ILL11_y"]
    ax1.plot(x1d, y1d, color='black', linewidth=1, label="ILL11 LSTM", alpha=0.7, ls="--", zorder=4)

    x1e, y1e = data["ax1_m2_ILL12_x"], data["ax1_m2_ILL12_y"]
    ax1.plot(x1e, y1e, color='C0', linewidth=1, label="ILL12 LSTM", alpha=0.7, ls="--", zorder=4)

    x1f, y1f = data["ax1_m2_ILL12_rescaled_x"], data["ax1_m2_ILL12_rescaled_y"]
    ax1.plot(x1f, y1f, color='C3', linewidth=1, label="ILL12-rescaled LSTM", alpha=0.7, ls="--", zorder=4)



    ax1.set_ylabel("", fontsize=7, fontweight='bold')
    ax1.set_xlabel("", fontsize=7, fontweight='bold')
    ax1.set_ylim(0, 45)
    ax1.grid(axis="both", color="grey", which="major", linestyle="--", lw=0.5, alpha=0.5, zorder=0)


    
    for idy, axis in enumerate([ax0, ax1]):
        
        if idx == 0:
            if idy == 0:
                axis.legend(loc="upper right", fontsize=6)
            else:
                axis.legend(loc="upper right", fontsize=6, ncol=1)
                
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axis.set_xlim(UTCDateTime(f"{startt}").matplotlib_date, UTCDateTime(f"{endt}").matplotlib_date)
        
        start = UTCDateTime(f"{startt}").matplotlib_date
        end = UTCDateTime(f"{endt}").matplotlib_date
        
        axis.set_xticks(np.arange(start, end + 1e-9, x_gap/24))
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axis.tick_params(axis='both', which='major', labelsize=7, length=3)
        axis.tick_params(axis='y', which='minor', labelsize=7, length=3)

        if idy == 0:
            axis.tick_params(axis='x', which='both', labelbottom=False)



subplot_idx_l = ["(a)", "(b)", "(c)", "(d)"]
date_l = ["2019-06-21", "2020-08-30", "2021-06-24", "2022-07-04"]

startt_l = ["2019-06-21T19:30:00", "2020-08-30T05:00:00", "2021-06-24T15:00:00", "2022-07-04T21:30:00"]
endt_l = ["2019-06-21T21:00:00", "2020-08-30T10:30:00", "2021-06-24T18:00:00", "2022-07-04T23:30:00"]
cache_name_l = ["subplot_a", "subplot_b", "subplot_c", "subplot_d"]
x_gap_l = [0.5, 1, 1, 0.5]

import matplotlib.gridspec as gridspec
# region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# endregion

plot_mapping = {0: (0, 2), 
                1: (1, 3),
                2: (4, 6),
                3: (5, 7)}

# fig = plt.figure(figsize=(8, 7))
# gs = gridspec.GridSpec(4, 2)

fig = plt.figure(figsize=(7, 7))

outer = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)


for idx in range(4):

    row = idx // 2
    col = idx % 2

    inner = outer[row, col].subgridspec(2, 1, hspace=0.1)

    ax0 = fig.add_subplot(inner[0])
    ax1 = fig.add_subplot(inner[1], sharex=ax0)
    
    make_plot2(ax0, ax1, startt_l[idx], endt_l[idx], cache_name_l[idx], idx, x_gap_l[idx])
    
    ax0.set_title(f"{subplot_idx_l[idx]} {date_l[idx]}", fontsize=8, fontweight="bold", loc="left")

    if idx in [0, 2]:
        ax0.set_ylabel("Seismic Amplitude [mm/s]", fontsize=7, fontweight='bold')
        ax1.set_ylabel("Impact Force [kN]", fontsize=7, fontweight='bold')
    
    if idx in [2, 3]:
        ax1.set_xlabel("Time [UTC+0]", fontsize=7, fontweight='bold')

plt.tight_layout()
png_path = Path(current_dir) / f"Figure-9.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
# plt.show()
plt.close(fig)