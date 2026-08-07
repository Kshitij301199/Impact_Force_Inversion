import os
import obspy
import obspy.signal.filter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from obspy import read, Trace, Stream, UTCDateTime

import matplotlib.ticker as ticker

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


import matplotlib.gridspec as gridspec
# region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# endregion




plt.rcParams.update({
    "font.size": 7,
    "font.family": "Arial",
    "legend.fontsize": 6,
    "axes.formatter.limits": (-3, 6),
    "axes.formatter.use_mathtext": True,
})

# MAKE PCA PLOT
def plot_embedding_projection(projection: np.ndarray, labels: np.ndarray,
                              title: str, save_path: str = None, label_order=[0, 1, 2]):

    fig, ax = plt.subplots(figsize=(3, 3))
    colors = np.array(["black", "blue", "green"])
    label_names = ["Background", "Flow Transition", "Flow Front"]
    id_to_name = {cluster_id: name for cluster_id, name in zip(label_order, label_names)}
    cmap = ListedColormap(colors)

    scatter = ax.scatter(projection[:, 0], projection[:, 1],
                         c=colors[labels], cmap=cmap, s=20, alpha=0.75)

    ax.set_xlabel("PCA component 1", fontsize=8, weight="bold")
    ax.set_ylabel("PCA component 2", fontsize=8, weight="bold")
    ax.tick_params(axis='both', which='major', labelsize=7, length=3)
    ax.grid(True, linestyle="--", alpha=0.25)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
               markeredgecolor="k", markersize=8, label=id_to_name[i])
        for i in label_order
    ]
    legend1 = ax.legend(handles=handles, title="Cluster", loc="best", fontsize=7)
    plt.setp(legend1.get_title(), fontsize=7)
    ax.add_artist(legend1)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=600, transparent=False)
    return fig, ax

# PSD PLOT CODE
def convert_st2tr(st):

    if isinstance(st, Stream):
        st = st[0]
    elif isinstance(st, Trace):
        pass
    else:
        print(f"!!! Error\n"
              f"Make sure the input for <convert_st2tr> is Obspy 'Trace' or 'Stream'.")

    return st

def rewrite_x_ticks(ax, data_start, data_end, data_sps, x_interval=1):
    '''
    Re write the x/time ticks

    Args:
        ax:
        data_start:
        data_end:
        data_sps:
        x_interval: for PSD, levea it as 1, for waveform or other, set as SPS

    Returns:

    '''
    start = UTCDateTime(data_start).timestamp
    end = UTCDateTime(data_end).timestamp

    x_location = np.arange(start, end + 1, 3600 * x_interval)
    x_ticks = []
    for j, k in enumerate(x_location):
        if j == 0:
            fmt = "%Y-%m-%d\n%H:%M:%S"
        else:
            fmt = "%H:%M"
        x_ticks.append(datetime.utcfromtimestamp(int(k)).strftime(fmt))

    x_location = (x_location - start) * data_sps

    ax.set_xticks(x_location, x_ticks)

def psd_plot(fig, ax, cbar_ax, st, fix_colorbar=True, per_lap=0.5, wlen=60, x_interval=1):

    st = convert_st2tr(st)
    st.spectrogram(per_lap=per_lap, wlen=wlen, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    data_sps = 1 / (per_lap * wlen)

    if fix_colorbar is True:
        ax.images[0].set_clim(-180, -100) # from experiences

    ax.set_ylim(1, 30)
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30], [1, 5, 10, 15, 20, 25, 30])
    ax.tick_params(axis='y', which='major', labelsize=7, length=3)
    ax.tick_params(axis='x', which='major', labelsize=5, length=2)
    ax.set_ylabel('Frequency [Hz]', fontsize=7, weight='bold')

    if cbar_ax is not None:
        cbar = fig.colorbar(ax.images[0], cax=cbar_ax, orientation="vertical")
        cbar.set_label("Power Spectral Density (dB)", fontsize=7)

    rewrite_x_ticks(ax,
                    data_start=st.stats.starttime,
                    data_end=st.stats.endtime,
                    data_sps=1, # fixed for psd
                    x_interval=x_interval)


    return ax, data_sps

def waveform_plot(ax, st, x_interval=1):

    st = convert_st2tr(st)
    data_source = f"{st.stats.network}-{st.stats.station}-{st.stats.channel}-SPS={int(st.stats.sampling_rate)}"

    ax.plot(st.data, color="black", label=data_source)
    ax.set_xlim(0, st.data.size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(st.stats.sampling_rate * 3600 * x_interval))  # unit is saecond
    ax.legend(loc="upper left", fontsize=6)
    ax.set_ylabel('Ampitude\n[m/s]', weight='bold')

    rewrite_x_ticks(ax,
                    data_start=st.stats.starttime,
                    data_end=st.stats.endtime,
                    data_sps=st.stats.sampling_rate,
                    x_interval=x_interval)

    return ax

###################################################################
print("APPLY PCA CASE 1")
st = read("9S.ILL11.EHZ.2019.172.mseed")
starttime = UTCDateTime("2019-06-21T19:30:00")
endtime = UTCDateTime("2019-06-21T22:30:00")
st.trim(starttime, endtime)
stt = st.copy()
stt[0].data = stt[0].data
timestamps = np.array([t for t in st[0].times("timestamp")])[5*60*100::5*100]
cache_path = Path(current_dir) / f"fig7-cache/row1.npz"
data = np.load(cache_path)





fig = plt.figure(figsize=(5.5, 6))
gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1], width_ratios=[10, 10, 0.5])


# ------------------------------------------------------------------
# Panel 0: K-means clusters on embedding PCA projection
# ------------------------------------------------------------------
ax = plt.subplot(gs[0, 0])
ax.set_title(label=f"(a)", loc="left", fontsize=8, fontweight='bold')

labels_pca = data['labels']
label_order = [0, 2, 1]
colors = ["black", "blue", "green"]
label_names = ["Background", "Flow Transition", "Flow Front"]
id_to_name = {cluster_id: name for cluster_id, name in zip(label_order, label_names)}
cmap = ListedColormap(colors)
# cache["embedding_2d"] = embedding_2d
# cache["labels"] = labels
scatter = ax.scatter(data["embedding_2d"][:, 0], data["embedding_2d"][:, 1],
                         c=data["labels"], cmap=cmap, s=20, alpha=0.75)

ax.set_ylabel("PCA component 2", fontsize=7, weight="bold")
ax.set_xlabel("PCA component 1", fontsize=7, weight="bold")
ax.tick_params(axis='both', which='major', labelsize=7, length=3)
ax.set_ylim(-0.5, 0.5)
# ax.set_xlim(-3, 10)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], [-0.5, -0.25, 0, 0.25, 0.5]) # type: ignore
ax.set_xticks([-5, 0, 5, 10, 15], [-5, 0, 5, 10, 15]) # type: ignore

ax.grid(True, linestyle="--", alpha=0.25)

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
           markeredgecolor=colors[i], markersize=7, label=id_to_name[i])
    for i in label_order
]
# legend0 = ax.legend(handles=handles, loc="best", fontsize=7)
# plt.setp(legend0.get_title(), fontsize=7)




# ------------------------------------------------------------------
# Panel 1 + 2: PSD spectrogram + amplitude envelope with cluster overlay
# ------------------------------------------------------------------
ax = plt.subplot(gs[1, :2])
ax.set_title(label=f"(c)", loc="left", fontsize=8, fontweight='bold')
cbar_ax = plt.subplot(gs[1, 2])

psd_plot(fig, ax, cbar_ax , stt, True, per_lap=0.9, wlen=5, x_interval=0.5)
ax.set_xlabel("Time [UTC+0]", fontsize=7, weight="bold")
ax.tick_params(axis='y', which='major', labelsize=7, length=3)
ax.tick_params(axis='x', which='major', labelsize=5, length=3)
ax_t = ax.twinx()
label_seen = set()
# cache["total_timestamps"] = total_timestamps
# cache["inp_seq"] = inp_seq
total_timestamps = data['total_timestamps']
inp_seq = data['inp_seq']
seq_len = inp_seq.shape[1]
n_sequences = inp_seq.shape[0]
total_timestamps = np.array(st[0].times("matplotlib")[1:])
sample_rate = st[0].stats.sampling_rate
sample_dt = 1.0 / sample_rate
if total_timestamps.size == n_sequences * seq_len:
    ts_matrix = total_timestamps.reshape(n_sequences, seq_len)
else:
    end_offset = (seq_len - 1) * sample_dt / 86400.0
    start_times = UTCDateTime(timestamps[0]).matplotlib_date - end_offset
    ts_matrix = np.tile(start_times + np.arange(seq_len) * sample_dt / 86400.0, (n_sequences, 1))
for i in range(n_sequences):
    ts = total_timestamps[i * seq_len:(i + 1) * seq_len]
    cluster = int(labels_pca[i])
    color = ["black", "blue", "green"][cluster]
    label = ["Background", "Flow Front", "Flow Transition"][cluster] if cluster not in label_seen else None
    label_seen.add(cluster)
    ax_t.scatter(np.arange(300 + (i * 5), 300 + (i * 5) + 5, 0.01), obspy.signal.filter.envelope(inp_seq[i, :]),
                 color=color, label=label, alpha=0.25, marker=".", s=1)

ax_t.set_ylim(0, 1.6)
ax.set_ylabel("Frequency [Hz]", fontsize=7, weight="bold")
ax_t.set_ylabel("Amplitude Envelope [mm/s]", fontsize=7, weight="bold")

# legend1 = ax_t.legend(loc="best", fontsize=7)
# for handle in legend1.legend_handles:
#     handle.set_sizes([50])
#     handle.set_alpha(1.0)



###################################################################
print("APPLY PCA CASE 2")
st = read("9S.ILL11.EHZ.2019.196.mseed")
starttime = UTCDateTime("2019-07-15T04:00:00")
endtime = UTCDateTime("2019-07-15T05:30:00")
st.trim(starttime, endtime)
stt = st.copy()
stt[0].data = stt[0].data
timestamps = np.array([t for t in st[0].times("timestamp")])[5*60*100::5*100]
cache_path = Path(current_dir) / f"fig7-cache/row2.npz"
data = np.load(cache_path)



# ------------------------------------------------------------------
# Panel 0: K-means clusters on embedding PCA projection
# ------------------------------------------------------------------
ax = plt.subplot(gs[0, 1])
ax.set_title(label=f"(b)", loc="left", fontsize=8, fontweight='bold')

labels_pca = data['labels']
label_order = [0, 2, 1]
colors = ["black", "blue", "green"]
label_names = ["Background", "Flow Transition", "Flow Front"]
id_to_name = {cluster_id: name for cluster_id, name in zip(label_order, label_names)}
cmap = ListedColormap(colors)
# cache["embedding_2d"] = embedding_2d
# cache["labels"] = labels
scatter = ax.scatter(data["embedding_2d"][:, 0], data["embedding_2d"][:, 1],
                         c=data["labels"], cmap=cmap, s=20, alpha=0.75)

ax.set_ylabel("PCA component 2", fontsize=7, weight="bold")
ax.set_xlabel("PCA component 1", fontsize=7, weight="bold")
ax.tick_params(axis='both', which='major', labelsize=7, length=3)
ax.set_ylim(-0.5, 0.5)
# ax.set_xlim(-3, 10)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], [-0.5, -0.25, 0, 0.25, 0.5]) # type: ignore
ax.set_xticks([-2, 0, 2, 4], [-2, 0, 2, 4]) # type: ignore


ax.grid(True, linestyle="--", alpha=0.25)

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
           markeredgecolor=colors[i], markersize=7, label=id_to_name[i])
    for i in label_order
]
legend0 = ax.legend(handles=handles, loc="best", fontsize=7)
plt.setp(legend0.get_title(), fontsize=7)

# ------------------------------------------------------------------
# Panel 1 + 2: PSD spectrogram + amplitude envelope with cluster overlay
# ------------------------------------------------------------------
ax = plt.subplot(gs[2, :2])
ax.set_title(label=f"(d)", loc="left", fontsize=8, fontweight='bold')
cbar_ax = plt.subplot(gs[2, 2])


psd_plot(fig, ax, cbar_ax , stt, True, per_lap=0.9, wlen=5, x_interval=0.5)
ax.set_xlabel("Time [UTC+0]", fontsize=7, weight="bold")
ax.tick_params(axis='y', which='major', labelsize=7, length=3)
ax.tick_params(axis='x', which='major', labelsize=5, length=3)
ax_t = ax.twinx()
label_seen = set()
# cache["total_timestamps"] = total_timestamps
# cache["inp_seq"] = inp_seq
total_timestamps = data['total_timestamps']
inp_seq = data['inp_seq']
seq_len = inp_seq.shape[1]
n_sequences = inp_seq.shape[0]
total_timestamps = np.array(st[0].times("matplotlib")[1:])
sample_rate = st[0].stats.sampling_rate
sample_dt = 1.0 / sample_rate
if total_timestamps.size == n_sequences * seq_len:
    ts_matrix = total_timestamps.reshape(n_sequences, seq_len)
else:
    end_offset = (seq_len - 1) * sample_dt / 86400.0
    start_times = UTCDateTime(timestamps[0]).matplotlib_date - end_offset
    ts_matrix = np.tile(start_times + np.arange(seq_len) * sample_dt / 86400.0, (n_sequences, 1))
for i in range(n_sequences):
    ts = total_timestamps[i * seq_len:(i + 1) * seq_len]
    cluster = int(labels_pca[i])
    color = ["black", "blue", "green"][cluster]
    label = ["Background", "Flow Front", "Flow Transition"][cluster] if cluster not in label_seen else None
    label_seen.add(cluster)
    ax_t.scatter(np.arange(300 + (i * 5), 300 + (i * 5) + 5, 0.01), obspy.signal.filter.envelope(inp_seq[i, :]),
                 color=color, label=label, alpha=0.25, marker=".", s=1)

ax_t.set_ylim(0, 1.6)
ax.set_ylabel("Frequency [Hz]", fontsize=7, weight="bold")
ax_t.set_ylabel("Amplitude Envelope [mm/s]", fontsize=7, weight="bold")

# legend1 = ax_t.legend(loc="best", fontsize=7)
# for handle in legend1.legend_handles:
#     handle.set_sizes([50])
#     handle.set_alpha(1.0)




plt.tight_layout()
png_path = Path(current_dir) / f"Figure-7.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)
# plt.show()
plt.close(fig)