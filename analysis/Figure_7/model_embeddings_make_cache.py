import os
import json
import sys

import torch
import obspy
import obspy.signal.filter
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from obspy import read, Trace, Stream, UTCDateTime

from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

with open("../../config/paths.json", "r") as file:
    paths = json.load(file)
with open("../../config/data_parameters.json", "r") as file:
    data_params = json.load(file)

sys.path.append(paths["LOCAL_BASE_DIR"])
from models.xLSTM_model import xLSTMRegressor_v2
from models.LSTM_model import LSTMRegressor_v2
from functions.data_processing.dataloader import SequenceDatasetTest, DataLoader

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "Arial",
        "legend.fontsize": 7,
        "figure.figsize": (6.0, 4.0),
        "axes.formatter.limits": (-3, 6),
        "axes.formatter.use_mathtext": True,
    }
)

with open("../../config/event_id_map.json", "r") as file:
    event_map = json.load(file)


# LOAD THE MODEL
def load_model(sub_interval: int, model: str = "xLSTM") -> xLSTMRegressor_v2 | LSTMRegressor_v2:
    config_dir = "../../config/task1"
    saved_model_dir = f"{paths['LOCAL_BASE_DIR']}/{paths['SAVED_MODEL_DIR']}/v4/2"
    if model == "xLSTM":
        config_path = os.path.join(config_dir, f"xlstm_v4_{sub_interval}sec_config.json")
        with open(config_path, "r") as file:
            config = json.load(file)

        model = xLSTMRegressor_v2(**config)
        model_path = os.path.join(saved_model_dir, f"{sub_interval}_xLSTM.pt")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
    elif model == "LSTM":
        config_path = os.path.join(config_dir, f"lstm_v4_{sub_interval}sec_config.json")
        with open(config_path, "r") as file:
            config = json.load(file)

        model = LSTMRegressor_v2(**config)
        model_path = os.path.join(saved_model_dir, f"{sub_interval}_LSTM.pt")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

    return model


# LOAD SEISMIC DATA
def load_seismic_data_test(
    julday: int | str | list,
    station: str,
    year: int = 2019,
    component: str = "EHZ",
    network: str = "9S",
    freq=None,
    rescale: bool = True,
) -> Stream:
    st = Stream()
    juldays = [julday] if isinstance(julday, (int, str)) else julday
    data_freq = data_params["fmax"] if freq is None else freq
    seis_dir = f"{paths['LOCAL_BASE_DIR']}/data_srr_30/Illgraben/2019/ILL11/EHZ"
    for j in juldays:
        filename = f"{seis_dir}/{network}.{station}.{component}.{year}.{str(j).zfill(3)}.mseed"
        st += read(filename)

    st.merge(method=1, fill_value="latest", interpolation_samples=0)
    for tr in st:
        tr.data = tr.data * 1e3

    return st


def build_case(julday: int, station: str, year: int, component: str, network: str, interval_seconds: int, freq: int):
    if julday in [182, 183]:
        st = Stream()
        st += load_seismic_data_test(
            julday=julday, station=station, year=year, component=component, network=network, freq=freq
        )
        st += load_seismic_data_test(
            julday=julday + 1, station=station, year=year, component=component, network=network, freq=freq
        )
        st.merge(method=1, fill_value="latest", interpolation_samples=0)
    else:
        st = load_seismic_data_test(
            julday=julday, station=station, year=year, component=component, network=network, freq=freq
        )
    if julday in [182, 183]:
        event_id = [k for k in ["4", "5"] if event_map[k]["julday"][0] == julday][0]
    else:
        event_id = [k for k in event_map if event_map[k]["julday"] == julday][0]
    starttime = UTCDateTime(event_map[event_id]["start_time"])
    endtime = UTCDateTime(event_map[event_id]["end_time"])
    st.trim(starttime, endtime)
    # data, _ = load_data_test(julday_list=[julday], station=station, year=year, abs=True)
    total_timestamps = st[0].times("matplotlib")[1:]
    timestamps = np.array([t for t in st[0].times("timestamp")])[5 * 60 * 100 :: 5 * 100]
    data_envelope = obspy.signal.filter.envelope(st[0].data)
    data = data_envelope
    data = data[1:]

    return st, data, timestamps


def build_dataloader(
    data: np.ndarray, timestamps: np.ndarray, interval_count: int = 60, sequence_length: int = 5 * 100
):
    dataset = SequenceDatasetTest(
        input_data=data, target_time=timestamps, interval_count=interval_count, sequence_length=sequence_length
    )
    dataloader = DataLoader(dataset=dataset, batch_size=len(timestamps), shuffle=False)
    return dataloader


def extract_embeddings(model: xLSTMRegressor_v2 | LSTMRegressor_v2, dataloader):
    outputs = []
    embeddings = []
    m_embeddings = []
    with torch.no_grad():
        for input_sequences, _ in dataloader:
            output = model(input_sequences)
            outputs.append(output.squeeze().numpy())
            layer_embeddings = model.get_embeddings(input_sequences)
            if isinstance(layer_embeddings, torch.Tensor):
                arr = layer_embeddings.squeeze().cpu().numpy()
            else:
                arr = np.asarray(layer_embeddings).squeeze()
            embeddings.append(arr)
            try:
                model_embeddings = model.get_xlstm_embeddings(input_sequences)
            except:
                model_embeddings = model.get_lstm_embeddings(input_sequences)
            if isinstance(model_embeddings, torch.Tensor):
                arr = model_embeddings.squeeze().cpu().numpy()
            else:
                arr = np.asarray(model_embeddings).squeeze()

            m_embeddings.append(arr)

    return np.array(outputs), np.array(embeddings), np.array(m_embeddings)


# KMEANS AND PCA FUNCTIONS
def fit_pca(embeddings: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components, random_state=0)
    projection = pca.fit_transform(embeddings)
    return pca, projection


def use_pca(pca: PCA, embeddings: np.ndarray, n_components: int = 2):
    projection = pca.transform(embeddings)
    return projection


def run_kmeans(projection: np.ndarray, n_clusters: int = 3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++")
    labels = kmeans.fit_predict(projection)
    return kmeans, labels


def use_kmeans(kmeans: KMeans, projection: np.ndarray, n_clusters: int = 3):
    labels = kmeans.predict(projection)
    return labels


# MAKE PCA PLOT
def plot_embedding_projection(
    projection: np.ndarray, labels: np.ndarray, title: str, save_path: str = None, label_order=[0, 1, 2]
):

    fig, ax = plt.subplots(figsize=(3, 3))
    colors = np.array(["black", "blue", "green"])
    label_names = ["Background", "Flow Transition", "Flow Front"]
    id_to_name = {cluster_id: name for cluster_id, name in zip(label_order, label_names)}
    cmap = ListedColormap(colors)

    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=colors[labels], cmap=cmap, s=20, alpha=0.75)

    ax.set_xlabel("PCA component 1", fontsize=8, weight="bold")
    ax.set_ylabel("PCA component 2", fontsize=8, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=7, length=3)
    ax.grid(True, linestyle="--", alpha=0.25)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markeredgecolor="k",
            markersize=8,
            label=id_to_name[i],
        )
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
        print("!!! Error\nMake sure the input for <convert_st2tr> is Obspy 'Trace' or 'Stream'.")

    return st


def rewrite_x_ticks(ax, data_start, data_end, data_sps, x_interval=1):
    """
    Re write the x/time ticks

    Args:
        ax:
        data_start:
        data_end:
        data_sps:
        x_interval: for PSD, levea it as 1, for waveform or other, set as SPS

    Returns:

    """
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
    st.spectrogram(per_lap=per_lap, wlen=wlen, log=False, dbscale=True, mult=True, title="", axes=ax, cmap="inferno")
    data_sps = 1 / (per_lap * wlen)

    if fix_colorbar is True:
        ax.images[0].set_clim(-180, -100)  # from experiences

    ax.set_ylim(1, 30)
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30], [1, 5, 10, 15, 20, 25, 30])
    ax.tick_params(axis="y", which="major", labelsize=7, length=3)
    ax.tick_params(axis="x", which="major", labelsize=5, length=2)
    ax.set_ylabel("Frequency [Hz]", fontsize=7, weight="bold")

    if cbar_ax is not None:
        cbar = fig.colorbar(ax.images[0], cax=cbar_ax, orientation="vertical")
        cbar.set_label("Power Spectral Density (dB)", fontsize=7)

    rewrite_x_ticks(
        ax,
        data_start=st.stats.starttime,
        data_end=st.stats.endtime,
        data_sps=1,  # fixed for psd
        x_interval=x_interval,
    )

    return ax, data_sps


def waveform_plot(ax, st, x_interval=1):

    st = convert_st2tr(st)
    data_source = f"{st.stats.network}-{st.stats.station}-{st.stats.channel}-SPS={int(st.stats.sampling_rate)}"

    ax.plot(st.data, color="black", label=data_source)
    ax.set_xlim(0, st.data.size)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(st.stats.sampling_rate * 3600 * x_interval))  # unit is saecond
    ax.legend(loc="upper left", fontsize=6)
    ax.set_ylabel("Ampitude\n[m/s]", weight="bold")

    rewrite_x_ticks(
        ax,
        data_start=st.stats.starttime,
        data_end=st.stats.endtime,
        data_sps=st.stats.sampling_rate,
        x_interval=x_interval,
    )

    return ax


print("FIT PCA")
case = {
    "julday": 207,
    "station": "ILL11",
    "year": 2019,
    "component": "EHZ",
    "network": "9S",
    "interval_seconds": 5,
    "freq": 30,
}

st, data, timestamps = build_case(**case)
dataloader = build_dataloader(data=data, timestamps=timestamps, interval_count=60, sequence_length=5 * 100)

print("Waveform sample count:", len(timestamps))
input_sequences, target_times = next(iter(dataloader))
print("Input sequences shape:", input_sequences.shape)
model = load_model(sub_interval=5, model="LSTM")
print(model)
output, embeddings, xlstm_embeddings = extract_embeddings(model, dataloader)
print("xlstm embeddings shape:", xlstm_embeddings.shape)

pca, embedding_2d = fit_pca(embeddings[0, :, -1, :], n_components=2)
kmeans, labels = run_kmeans(embedding_2d, n_clusters=3)

print("APPLY PCA CASE 1")
cache_path = "./fig7-cache/row1.npz"
cache = {}
case = {
    "julday": 172,
    "station": "ILL11",
    "year": 2019,
    "component": "EHZ",
    "network": "9S",
    "interval_seconds": 5,
    "freq": 30,
}

st, data, timestamps = build_case(**case)
dataloader = build_dataloader(data=data, timestamps=timestamps, interval_count=60, sequence_length=5 * 100)
stt = st.copy()
stt[0].data = stt[0].data / 1e3
print("Waveform sample count:", len(timestamps))
input_sequences, target_times = next(iter(dataloader))
print("Input sequences shape:", input_sequences.shape)
model = load_model(sub_interval=5, model="LSTM")
print(model)
output, embeddings, xlstm_embeddings = extract_embeddings(model, dataloader)
print("xlstm embeddings shape:", xlstm_embeddings.shape)
embedding_2d = use_pca(pca, embeddings[0, :, -1, :], n_components=2)

indices_by_cluster = {cluster: np.where(labels == cluster)[0].tolist() for cluster in np.unique(labels)}
for cluster, indices in indices_by_cluster.items():
    print(f"Cluster {cluster} sample indices: {indices[:10]}{'...' if len(indices) > 10 else ''}")
input_sequences_np = input_sequences.cpu().numpy() if isinstance(input_sequences, torch.Tensor) else input_sequences
inp_seq = input_sequences_np[:, -1, :]
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

fig, ax = plt.subplots(1, 3, figsize=(6.5, 3), width_ratios=[4, 6, 0.25])
# ------------------------------------------------------------------
# Panel 0: K-means clusters on embedding PCA projection
# ------------------------------------------------------------------
labels = use_kmeans(kmeans, embedding_2d, n_clusters=3)
labels_pca = labels

label_order = [0, 2, 1]
colors = ["black", "blue", "green"]
label_names = ["Background", "Flow Transition", "Flow Front"]
id_to_name = {cluster_id: name for cluster_id, name in zip(label_order, label_names)}
cmap = ListedColormap(colors)
cache["embedding_2d"] = embedding_2d
cache["labels"] = labels
scatter = ax[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap, s=20, alpha=0.75)

ax[0].set_ylabel("PCA component 2", fontsize=7, weight="bold")
ax[0].set_xlabel("PCA component 1", fontsize=7, weight="bold")
ax[0].tick_params(axis="both", which="major", labelsize=7, length=3)
ax[0].set_ylim(-0.5, 0.5)
ax[0].set_xlim(-3, 10)
ax[0].grid(True, linestyle="--", alpha=0.25)

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=colors[i],
        markeredgecolor=colors[i],
        markersize=7,
        label=id_to_name[i],
    )
    for i in label_order
]
legend0 = ax[0].legend(handles=handles, loc="best", fontsize=7)
plt.setp(legend0.get_title(), fontsize=7)

# ------------------------------------------------------------------
# Panel 1 + 2: PSD spectrogram + amplitude envelope with cluster overlay
# ------------------------------------------------------------------
psd_plot(fig, ax[1], ax[2], stt, True, per_lap=0.9, wlen=5, x_interval=0.5)
ax[1].set_xlabel("Time [UTC+0]", fontsize=7, weight="bold")
ax[1].tick_params(axis="y", which="major", labelsize=7, length=3)
ax[1].tick_params(axis="x", which="major", labelsize=5, length=3)
ax_t = ax[1].twinx()
label_seen = set()
cache["total_timestamps"] = total_timestamps
cache["inp_seq"] = inp_seq
for i in range(n_sequences):
    ts = total_timestamps[i * seq_len : (i + 1) * seq_len]
    cluster = int(labels_pca[i])
    color = ["black", "blue", "green"][cluster]
    label = ["Background", "Flow Front", "Flow Transition"][cluster] if cluster not in label_seen else None
    label_seen.add(cluster)
    ax_t.scatter(
        np.arange(300 + (i * 5), 300 + (i * 5) + 5, 0.01),
        obspy.signal.filter.envelope(inp_seq[i, :]),
        color=color,
        label=label,
        alpha=0.25,
        marker=".",
        s=1,
    )

ax_t.set_ylim(0, 1.6)
ax[1].set_ylabel("Frequency [Hz]", fontsize=7, weight="bold")
ax_t.set_ylabel("Amplitude Envelope [mm/s]", fontsize=7, weight="bold")

legend1 = ax_t.legend(loc="best", fontsize=7)
for handle in legend1.legend_handles:
    handle.set_sizes([50])
    handle.set_alpha(1.0)

fig.tight_layout()
fig.savefig(f"embedding_combined_{case['julday']}.png", dpi=600)
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
np.savez(cache_path, **cache)

print("APPLY PCA CASE 2")
cache_path = "./fig7-cache/row2.npz"
cache = {}
case = {
    "julday": 196,
    "station": "ILL11",
    "year": 2019,
    "component": "EHZ",
    "network": "9S",
    "interval_seconds": 5,
    "freq": 30,
}

st, data, timestamps = build_case(**case)
dataloader = build_dataloader(data=data, timestamps=timestamps, interval_count=60, sequence_length=5 * 100)
stt = st.copy()
stt[0].data = stt[0].data / 1e3
print("Waveform sample count:", len(timestamps))
input_sequences, target_times = next(iter(dataloader))
print("Input sequences shape:", input_sequences.shape)
model = load_model(sub_interval=5, model="LSTM")
print(model)
output, embeddings, xlstm_embeddings = extract_embeddings(model, dataloader)
print("xlstm embeddings shape:", xlstm_embeddings.shape)
embedding_2d = use_pca(pca, embeddings[0, :, -1, :], n_components=2)

indices_by_cluster = {cluster: np.where(labels == cluster)[0].tolist() for cluster in np.unique(labels)}
for cluster, indices in indices_by_cluster.items():
    print(f"Cluster {cluster} sample indices: {indices[:10]}{'...' if len(indices) > 10 else ''}")
input_sequences_np = input_sequences.cpu().numpy() if isinstance(input_sequences, torch.Tensor) else input_sequences
inp_seq = input_sequences_np[:, -1, :]
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

fig, ax = plt.subplots(1, 3, figsize=(6.5, 3), width_ratios=[4, 6, 0.25])
# ------------------------------------------------------------------
# Panel 0: K-means clusters on embedding PCA projection
# ------------------------------------------------------------------
labels = use_kmeans(kmeans, embedding_2d, n_clusters=3)
labels_pca = labels

label_order = [0, 2, 1]
colors = ["black", "blue", "green"]
label_names = ["Background", "Flow Transition", "Flow Front"]
id_to_name = {cluster_id: name for cluster_id, name in zip(label_order, label_names)}
cmap = ListedColormap(colors)
cache["embedding_2d"] = embedding_2d
cache["labels"] = labels
scatter = ax[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap=cmap, s=20, alpha=0.75)

ax[0].set_ylabel("PCA component 2", fontsize=7, weight="bold")
ax[0].set_xlabel("PCA component 1", fontsize=7, weight="bold")
ax[0].tick_params(axis="both", which="major", labelsize=7, length=3)
ax[0].set_ylim(-0.5, 0.5)
ax[0].set_xlim(-3, 10)
ax[0].grid(True, linestyle="--", alpha=0.25)

handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=colors[i],
        markeredgecolor=colors[i],
        markersize=7,
        label=id_to_name[i],
    )
    for i in label_order
]
legend0 = ax[0].legend(handles=handles, loc="best", fontsize=7)
plt.setp(legend0.get_title(), fontsize=7)

# ------------------------------------------------------------------
# Panel 1 + 2: PSD spectrogram + amplitude envelope with cluster overlay
# ------------------------------------------------------------------
psd_plot(fig, ax[1], ax[2], stt, True, per_lap=0.9, wlen=5, x_interval=0.5)
ax[1].set_xlabel("Time [UTC+0]", fontsize=7, weight="bold")
ax[1].tick_params(axis="y", which="major", labelsize=7, length=3)
ax[1].tick_params(axis="x", which="major", labelsize=5, length=3)
ax_t = ax[1].twinx()
label_seen = set()
cache["total_timestamps"] = total_timestamps
cache["inp_seq"] = inp_seq
for i in range(n_sequences):
    ts = total_timestamps[i * seq_len : (i + 1) * seq_len]
    cluster = int(labels_pca[i])
    color = ["black", "blue", "green"][cluster]
    label = ["Background", "Flow Front", "Flow Transition"][cluster] if cluster not in label_seen else None
    label_seen.add(cluster)
    ax_t.scatter(
        np.arange(300 + (i * 5), 300 + (i * 5) + 5, 0.01),
        obspy.signal.filter.envelope(inp_seq[i, :]),
        color=color,
        label=label,
        alpha=0.25,
        marker=".",
        s=1,
    )

ax_t.set_ylim(0, 1.6)
ax[1].set_ylabel("Frequency [Hz]", fontsize=7, weight="bold")
ax_t.set_ylabel("Amplitude Envelope [mm/s]", fontsize=7, weight="bold")

legend1 = ax_t.legend(loc="best", fontsize=7)
for handle in legend1.legend_handles:
    handle.set_sizes([50])
    handle.set_alpha(1.0)

fig.tight_layout()
fig.savefig(f"embedding_combined_{case['julday']}.png", dpi=600)
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
np.savez(cache_path, **cache)
