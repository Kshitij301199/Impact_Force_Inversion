import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import UTCDateTime

from matplotlib.ticker import ScalarFormatter, MultipleLocator


from matplotlib import gridspec

# region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# endregion

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "Arial",
        "legend.fontsize": 6,
        "axes.formatter.limits": (-3, 6),
        "axes.formatter.use_mathtext": True,
    }
)

data1 = np.load("cache1.npz")
data2 = np.load("cache2.npz")

fig = plt.figure(figsize=(5.5, 3.5))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

ax = plt.subplot(gs[0, 0])
ax.set_title(label="(a)", loc="left", fontsize=8, fontweight="bold")

# ax.set_ylabel("Amplitude [mm/s]", fontsize=7, weight="bold")
ax.text(
    -0.17,
    0.5,
    "Amplitude [mm/s]",
    transform=ax.transAxes,
    rotation="vertical",
    va="center",
    ha="left",
    fontweight="bold",
    fontsize=7,
)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax.set_ylim(-1.6, 1.6)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.tick_params(axis="y", which="major", labelsize=7, length=3)
ax.tick_params(axis="y", which="minor", labelsize=7, length=1)
# ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=0)

ax.plot(data1["seis_time1"], data1["data1"], color="grey", linewidth=2, zorder=1, label="ILL11")
ax.plot(data1["seis_time2"], data1["data2"], color="k", linewidth=2, zorder=2, label="Filtered+Env")

start = UTCDateTime("2019-06-21T19:40:00").matplotlib_date
end = UTCDateTime("2019-06-21T20:45:00").matplotlib_date
ax.set_xticks(np.arange(start, end + 1e-9, 15 / (24 * 60)))
ax.set_xlim(UTCDateTime("2019-06-21T19:40:00").matplotlib_date, UTCDateTime("2019-06-21T20:45:00").matplotlib_date)
ax.legend(loc="upper right", fontsize=6)

ax = plt.subplot(gs[0, 1])
ax.set_title(label="(b)", loc="left", fontsize=8, fontweight="bold")

# ax.set_ylabel("Amplitude [mm/s]", fontsize=7, weight="bold")
ax.text(
    -0.17,
    0.5,
    "Amplitude [mm/s]",
    transform=ax.transAxes,
    rotation="vertical",
    va="center",
    ha="left",
    fontweight="bold",
    fontsize=7,
)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax.set_ylim(-1.6, 1.6)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.tick_params(axis="y", which="major", labelsize=7, length=3)
ax.tick_params(axis="y", which="minor", labelsize=7, length=1)
# ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=0)

ax.plot(data2["seis_time1"], data2["data1"], color="grey", linewidth=2, zorder=1, label="ILL11")
ax.plot(data2["seis_time2"], data2["data2"], color="k", linewidth=2, zorder=2, label="Filtered+Env")

start = UTCDateTime("2019-07-15T04:15:00").matplotlib_date
end = UTCDateTime("2019-07-15T05:10:00").matplotlib_date
ax.set_xticks(np.arange(start, end + 1e-9, 15 / (24 * 60)))
ax.set_xlim(UTCDateTime("2019-07-15T04:15:00").matplotlib_date, UTCDateTime("2019-07-15T05:10:00").matplotlib_date)
ax.legend(loc="upper right", fontsize=6)

ax = plt.subplot(gs[1, 0])
ax.set_title(label="(c)", loc="left", fontsize=8, fontweight="bold")

# ax.set_ylabel("Amplitude [mm/s]", fontsize=7, weight="bold")
ax.text(
    -0.17,
    0.5,
    "Impact Force [kN]",
    transform=ax.transAxes,
    rotation="vertical",
    va="center",
    ha="left",
    fontweight="bold",
    fontsize=7,
)
ax.set_xlabel("Time [UTC+0]", fontweight="bold", fontsize=7)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax.set_ylim(0, 320)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_locator(MultipleLocator(5 * 8))
ax.yaxis.set_minor_locator(MultipleLocator(2.5 * 8))
ax.tick_params(axis="y", which="major", labelsize=7, length=3)
ax.tick_params(axis="y", which="minor", labelsize=7, length=1)
# ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=0)

ax.plot(
    [UTCDateTime(i).matplotlib_date for i in data1["times1"]],
    data1["force1"],
    label="Raw IF Data",
    color="blue",
    linewidth=1,
)
ax.plot(
    [UTCDateTime(i).matplotlib_date for i in data1["times2"]],
    data1["force2"],
    label="Shifted+\nSynthetic IF Data",
    color="k",
    linewidth=1,
    alpha=0.8,
)
ax.plot(
    [UTCDateTime(i).matplotlib_date for i in data1["times3"]],
    data1["force3"],
    label="Trimed+\nSmoothened IF Data",
    color="r",
    linewidth=1.5,
    alpha=0.8,
)

start = UTCDateTime("2019-06-21T19:40:00").matplotlib_date
end = UTCDateTime("2019-06-21T20:45:00").matplotlib_date
ax.set_xticks(np.arange(start, end + 1e-9, 15 / (24 * 60)))
ax.set_xlim(UTCDateTime("2019-06-21T19:40:00").matplotlib_date, UTCDateTime("2019-06-21T20:45:00").matplotlib_date)
ax.legend(loc="upper right", fontsize=6)

ax = plt.subplot(gs[1, 1])
ax.set_title(label="(d)", loc="left", fontsize=8, fontweight="bold")

# ax.set_ylabel("Amplitude [mm/s]", fontsize=7, weight="bold")
ax.text(
    -0.17,
    0.5,
    "Impact Force [kN]",
    transform=ax.transAxes,
    rotation="vertical",
    va="center",
    ha="left",
    fontweight="bold",
    fontsize=7,
)
ax.set_xlabel("Time [UTC+0]", fontweight="bold", fontsize=7)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
ax.set_ylim(0, 320)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_locator(MultipleLocator(5 * 8))
ax.yaxis.set_minor_locator(MultipleLocator(2.5 * 8))
ax.tick_params(axis="y", which="major", labelsize=7, length=3)
ax.tick_params(axis="y", which="minor", labelsize=7, length=1)
# ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=0)

ax.plot(
    [UTCDateTime(i).matplotlib_date for i in data2["times1"]],
    data2["force1"],
    label="Raw IF Data",
    color="blue",
    linewidth=1,
)
ax.plot(
    [UTCDateTime(i).matplotlib_date for i in data2["times2"]],
    data2["force2"],
    label="Shifted+\nSynthetic IF Data",
    color="k",
    linewidth=1,
    alpha=0.8,
)
ax.plot(
    [UTCDateTime(i).matplotlib_date for i in data2["times3"]],
    data2["force3"],
    label="Trimed+\nSmoothened IF Data",
    color="r",
    linewidth=1.5,
    alpha=0.8,
)

start = UTCDateTime("2019-07-15T04:15:00").matplotlib_date
end = UTCDateTime("2019-07-15T05:10:00").matplotlib_date
ax.set_xticks(np.arange(start, end + 1e-9, 15 / (24 * 60)))
ax.set_xlim(UTCDateTime("2019-07-15T04:15:00").matplotlib_date, UTCDateTime("2019-07-15T05:10:00").matplotlib_date)
ax.legend(loc="upper right", fontsize=6)


plt.tight_layout()
png_path = Path(current_dir) / "Figure-2.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
# plt.show()
plt.close(fig)
