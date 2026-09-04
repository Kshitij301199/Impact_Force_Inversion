import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

from obspy import UTCDateTime, read

with open("../../config/paths.json", "r") as file:
    paths = json.load(file)
with open("../../config/data_parameters.json", "r") as file:
    data_params = json.load(file)

sys.path.append(paths["LOCAL_BASE_DIR"])

plt.rcParams.update(
    {
        "font.size": 7,
        "font.family": "Arial",
        "legend.fontsize": 6,
        "figure.figsize": (5.5, 3.5),
        "axes.formatter.limits": (-3, 6),
        "axes.formatter.use_mathtext": True,
    }
)


def rescale_amplitude_by_distance(st, r0, r, n, Q, c, freq_band=None):
    """
    Rescale amplitudes in an ObsPy Stream from an original station distance (r0)
    to an equivalent target distance (r), using geometric spreading + anelastic
    attenuation:

        S(f) = (r0/r)^n * exp(-pi * f * (r - r0) / (Q * c))

    Args:
        st (obspy.Stream): input stream of traces to rescale.
        r0 (float): original station distance [m].
        r (float): target station distance [m].
        n (float): geometric spreading exponent (0.5 for surface waves).
        Q (float): quality factor.
        c (float): wave velocity [m/s].
        freq_band (tuple or None): (fmin, fmax) in Hz. If given, the scaling
            factor is only applied within this band; frequencies outside are
            left unscaled (factor = 1). If None, scaling is applied across
            the full spectrum.

    Returns:
        obspy.Stream: new stream with rescaled traces (input is not modified).
    """
    st_out = st.copy()

    geom_term = (r0 / r) ** n

    for tr in st_out:
        fs = tr.stats.sampling_rate
        data = tr.data.astype(np.float64)
        N = len(data)

        freqs = np.fft.rfftfreq(N, d=1 / fs)
        spectrum = np.fft.rfft(data)

        atten_term = np.exp(-np.pi * freqs * (r - r0) / (Q * c))
        S = geom_term * atten_term

        if freq_band is not None:
            fmin, fmax = freq_band
            mask = (freqs >= fmin) & (freqs <= fmax)
            S_full = np.ones_like(S)
            S_full[mask] = S[mask]
            S = S_full

        scaled_spectrum = spectrum * S
        rescaled_data = np.fft.irfft(scaled_spectrum, n=N)

        tr.data = rescaled_data.astype(tr.data.dtype, copy=False)

        # keep a record of what was done, in case you need it later
        tr.stats.processing = tr.stats.get("processing", [])
        tr.stats.processing.append(
            f"rescale_amplitude_by_distance(r0={r0}, r={r}, n={n}, Q={Q}, c={c}, freq_band={freq_band})"
        )

    return st_out


def make_plot(
    st_ILL11, st_ILL12, st_ILL12_org, m1o11, m1o12, m1o12r, m2o11, m2o12, m2o12r, date, startt, endt, cache_name
):
    cache_path = f"./fig9-cache/{cache_name}.npz"
    cache = {}

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 3.5))

    # --- Panel 0: seismic amplitude traces ---
    x0a, y0a = st_ILL11[0].times("matplotlib"), st_ILL11[0].data
    cache["ax0_ILL11_x"], cache["ax0_ILL11_y"] = np.array(x0a), np.array(y0a)
    ax[0].plot(x0a, y0a, color="green", linewidth=1, label="ILL11", alpha=0.7, zorder=1)

    x0b, y0b = st_ILL12_org[0].times("matplotlib"), st_ILL12_org[0].data
    cache["ax0_ILL12_x"], cache["ax0_ILL12_y"] = np.array(x0b), np.array(y0b)
    ax[0].plot(x0b, y0b, color="blue", linewidth=1, label="ILL12", alpha=0.7, zorder=1)

    x0c, y0c = st_ILL12[0].times("matplotlib"), st_ILL12[0].data
    cache["ax0_ILL12_rescaled_x"], cache["ax0_ILL12_rescaled_y"] = np.array(x0c), np.array(y0c)
    ax[0].plot(x0c, y0c, color="red", linewidth=1, label="ILL12-rescaled", alpha=0.7, zorder=0)

    ax[0].set_ylim(-1.1e-3, 1.1e-3)
    ax[0].text(
        -0.1,
        0.5,
        "Seismic Amplitude [m/s]",
        rotation="vertical",
        ha="center",
        va="center",
        transform=ax[0].transAxes,
        fontsize=7,
        fontweight="bold",
    )
    ax[0].yaxis.get_offset_text().set_fontsize(7)
    ax[0].yaxis.get_offset_text().set_fontweight("bold")

    # --- Panel 1: predicted impact force curves ---
    x1a = m1o11.Timestamps.apply(lambda x: UTCDateTime(x).matplotlib_date)
    y1a = m1o11.Predicted_Output_Mean
    cache["ax1_m1_ILL11_x"], cache["ax1_m1_ILL11_y"] = np.array(x1a), np.array(y1a)
    ax[1].plot(x1a, y1a, color="green", linewidth=1, label="ILL11", alpha=0.7)

    x1b = m1o12r.Timestamps.apply(lambda x: UTCDateTime(x).matplotlib_date)
    y1b = m1o12r.Predicted_Output_Mean
    cache["ax1_m1_ILL12_x"], cache["ax1_m1_ILL12_y"] = np.array(x1b), np.array(y1b)
    ax[1].plot(x1b, y1b, color="blue", linewidth=1, label="ILL12", alpha=0.7)

    x1c = m1o12.Timestamps.apply(lambda x: UTCDateTime(x).matplotlib_date)
    y1c = m1o12.Predicted_Output_Mean
    cache["ax1_m1_ILL12_rescaled_x"], cache["ax1_m1_ILL12_rescaled_y"] = np.array(x1c), np.array(y1c)
    ax[1].plot(x1c, y1c, color="red", linewidth=1, label="ILL12-rescaled", alpha=0.7)

    x1d = m2o11.Timestamps.apply(lambda x: UTCDateTime(x).matplotlib_date)
    y1d = m2o11.Predicted_Output_Mean
    cache["ax1_m2_ILL11_x"], cache["ax1_m2_ILL11_y"] = np.array(x1d), np.array(y1d)
    ax[1].plot(x1d, y1d, color="green", linewidth=1, alpha=0.7, linestyle="--")

    x1e = m2o12r.Timestamps.apply(lambda x: UTCDateTime(x).matplotlib_date)
    y1e = m2o12r.Predicted_Output_Mean
    cache["ax1_m2_ILL12_x"], cache["ax1_m2_ILL12_y"] = np.array(x1e), np.array(y1e)
    ax[1].plot(x1e, y1e, color="blue", linewidth=1, alpha=0.7, linestyle="--")

    x1f = m2o12.Timestamps.apply(lambda x: UTCDateTime(x).matplotlib_date)
    y1f = m2o12.Predicted_Output_Mean
    cache["ax1_m2_ILL12_rescaled_x"], cache["ax1_m2_ILL12_rescaled_y"] = np.array(x1f), np.array(y1f)
    ax[1].plot(x1f, y1f, color="red", linewidth=1, alpha=0.7, linestyle="--")

    ax[1].text(
        -0.1,
        0.5,
        "Impact Force [kN]",
        rotation="vertical",
        ha="center",
        va="center",
        transform=ax[1].transAxes,
        fontsize=7,
        fontweight="bold",
    )
    ax[1].set_xlabel("Time [UTC+0]", fontsize=7, fontweight="bold")
    ax[1].set_ylim(0, 320)

    ax[0].set_title(f"{date}", fontsize=8, fontweight="bold")
    for axis in ax:
        axis.legend(loc="upper right", fontsize=7)
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axis.set_xlim(UTCDateTime(f"{startt}").matplotlib_date, UTCDateTime(f"{endt}").matplotlib_date)
        start = UTCDateTime(f"{startt}").matplotlib_date
        end = UTCDateTime(f"{endt}").matplotlib_date
        axis.set_xticks(np.arange(start, end + 1e-9, 20 / (24 * 60)))
        axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axis.tick_params(axis="both", which="major", labelsize=7, length=3)
        axis.tick_params(axis="y", which="minor", labelsize=7, length=3)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    fig.savefig(f"./{date}.png", dpi=600)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **cache)


Q, v, n = 25, 800, 0.5
model_output_dir = "../../output_final3/model_test"
model_output_dir2 = "../../output_final3/2model_test"
model_output_dir3 = "../../output_final3/model_test"
seis_dir = f"{paths['LOCAL_BASE_DIR']}/data_srr_30/Illgraben"


# 2019-06-21
st_ILL11 = read(f"{seis_dir}/2019/ILL11/EHZ/9S.ILL11.EHZ.2019.172.mseed")
st_ILL12 = read(f"{seis_dir}/2019/ILL12/EHZ/9S.ILL12.EHZ.2019.172.mseed")
st_ILL11.trim(starttime=UTCDateTime("2019-06-21T18:00:00"), endtime=UTCDateTime("2019-06-21T22:00:00"))
st_ILL12.trim(starttime=UTCDateTime("2019-06-21T18:00:00"), endtime=UTCDateTime("2019-06-21T22:00:00"))
st_ILL12_org = st_ILL12.copy()

model1_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/xLSTM_5/2019/df/172.csv")
model2_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/LSTM_5/2019/df/172.csv")

model1_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/xLSTM_5/2019/df/172.csv")
model1_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/xLSTM_5/2019/df/172.csv")
model2_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/LSTM_5/2019/df/172.csv")
model2_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/LSTM_5/2019/df/172.csv")

st_ILL12 = rescale_amplitude_by_distance(st_ILL12, 90, 15, n, Q, v, (1, 30))

make_plot(
    st_ILL11,
    st_ILL12,
    st_ILL12_org,
    model1_output_ILL11,
    model1_output_ILL12,
    model1_output_ILL12_r,
    model2_output_ILL11,
    model2_output_ILL12,
    model2_output_ILL12_r,
    "2019-06-21",
    "2019-06-21T19:30:00",
    "2019-06-21T21:00:00",
    "subplot_a",
)

# 2020-08-30
st_ILL11 = read(f"{seis_dir}/2020/ILL11/EHZ/9S.ILL11.EHZ.2020.243.mseed")
st_ILL12 = read(f"{seis_dir}/2020/ILL12/EHZ/9S.ILL12.EHZ.2020.243.mseed")
st_ILL11.trim(starttime=UTCDateTime("2020-08-30T05:00:00"), endtime=UTCDateTime("2020-08-30T10:30:00"))
st_ILL12.trim(starttime=UTCDateTime("2020-08-30T05:00:00"), endtime=UTCDateTime("2020-08-30T10:30:00"))
st_ILL12_org = st_ILL12.copy()

model1_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/xLSTM_5/2020/df/243.csv")
model2_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/LSTM_5/2020/df/243.csv")

model1_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/xLSTM_5/2020/df/243.csv")
model1_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/xLSTM_5/2020/df/243.csv")
model2_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/LSTM_5/2020/df/243.csv")
model2_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/LSTM_5/2020/df/243.csv")

st_ILL12 = rescale_amplitude_by_distance(st_ILL12, 90, 15, n, Q, v, (1, 30))

make_plot(
    st_ILL11,
    st_ILL12,
    st_ILL12_org,
    model1_output_ILL11,
    model1_output_ILL12,
    model1_output_ILL12_r,
    model2_output_ILL11,
    model2_output_ILL12,
    model2_output_ILL12_r,
    "2020-08-30",
    "2020-08-30T05:00:00",
    "2020-08-30T10:30:00",
    "subplot_b",
)

# 2021-06-24
st_ILL11 = read(f"{seis_dir}/2021/ILL11/EHZ/9S.ILL11.EHZ.2021.175.mseed")
st_ILL12 = read(f"{seis_dir}/2021/ILL12/EHZ/9S.ILL12.EHZ.2021.175.mseed")
st_ILL11.trim(starttime=UTCDateTime("2021-06-24T15:00:00"), endtime=UTCDateTime("2021-06-24T18:00:00"))
st_ILL12.trim(starttime=UTCDateTime("2021-06-24T15:00:00"), endtime=UTCDateTime("2021-06-24T18:00:00"))
st_ILL12_org = st_ILL12.copy()

model1_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/xLSTM_5/2021/df/175.csv")
model2_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/LSTM_5/2021/df/175.csv")

model1_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/xLSTM_5/2021/df/175.csv")
model1_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/xLSTM_5/2021/df/175.csv")
model2_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/LSTM_5/2021/df/175.csv")
model2_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/LSTM_5/2021/df/175.csv")

st_ILL12 = rescale_amplitude_by_distance(st_ILL12, 90, 15, n, Q, v, (1, 30))

make_plot(
    st_ILL11,
    st_ILL12,
    st_ILL12_org,
    model1_output_ILL11,
    model1_output_ILL12,
    model1_output_ILL12_r,
    model2_output_ILL11,
    model2_output_ILL12,
    model2_output_ILL12_r,
    "2021-06-24",
    "2021-06-24T15:00:00",
    "2021-06-24T18:00:00",
    "subplot_c",
)

# 2022-07-04
st_ILL11 = read(f"{seis_dir}/2022/ILL11/EHZ/9S.ILL11.EHZ.2022.185.mseed")
st_ILL12 = read(f"{seis_dir}/2022/ILL12/EHZ/9S.ILL12.EHZ.2022.185.mseed")
st_ILL11.trim(starttime=UTCDateTime("2022-07-04T21:00:00"), endtime=UTCDateTime("2022-07-04T23:30:00"))
st_ILL12.trim(starttime=UTCDateTime("2022-07-04T21:00:00"), endtime=UTCDateTime("2022-07-04T23:30:00"))
st_ILL12_org = st_ILL12.copy()

model1_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/xLSTM_5/2022/df/185.csv")
model2_output_ILL11 = pd.read_csv(f"{model_output_dir}/ILL11/LSTM_5/2022/df/185.csv")

model1_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/xLSTM_5/2022/df/185.csv")
model1_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/xLSTM_5/2022/df/185.csv")
model2_output_ILL12 = pd.read_csv(f"{model_output_dir3}/ILL12/LSTM_5/2022/df/185.csv")
model2_output_ILL12_r = pd.read_csv(f"{model_output_dir2}/ILL12/LSTM_5/2022/df/185.csv")

st_ILL12 = rescale_amplitude_by_distance(st_ILL12, 90, 15, n, Q, v, (1, 30))

make_plot(
    st_ILL11,
    st_ILL12,
    st_ILL12_org,
    model1_output_ILL11,
    model1_output_ILL12,
    model1_output_ILL12_r,
    model2_output_ILL11,
    model2_output_ILL12,
    model2_output_ILL12_r,
    "2022-07-04",
    "2022-07-04T21:30:00",
    "2022-07-04T23:30:00",
    "subplot_d",
)
