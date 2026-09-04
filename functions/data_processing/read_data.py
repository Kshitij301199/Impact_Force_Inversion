#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-01
# __author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
# __find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import obspy
import obspy.signal.filter
from obspy import read, Stream
from obspy.core import UTCDateTime

# Path and Config Handling
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent


def load_json_config(name):
    try:
        path = project_root / "config" / name
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # HPC Fallback
        hpc_path = Path("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config") / name
        with open(hpc_path, "r") as f:
            return json.load(f)


paths = load_json_config("paths.json")
data_params = load_json_config("data_parameters.json")
time_config = load_json_config("event_id_map.json")

sys.path.append(paths["BASE_DIR"])


def _resolve_path(sub_path):
    """Helper to check both HPC and Local base directories."""
    for base in [paths.get("BASE_DIR"), paths.get("LOCAL_BASE_DIR")]:
        if base:
            full_path = os.path.join(base, sub_path)
            if os.path.exists(full_path):
                return full_path
    return None


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


def load_seismic_data(
    event_id: str | int,
    station: str,
    year: int = 2019,
    component: str = "EHZ",
    network: str = "9S",
    trim: bool = True,
    val: bool = False,
) -> Stream:
    event_info = time_config[str(event_id)]

    # Delegate loading to the standardized test loader
    st = load_seismic_data_test(
        julday=event_info["julday"],
        station=station,
        year=year,
        component=component,
        network=network,
        freq=data_params["fmax"],
        rescale=False,  # Rescaling is handled separately if needed
    )

    if trim:
        if val:
            time_window = data_params["time_window"]
            st.trim(
                starttime=UTCDateTime(event_info["start_time"]) - (time_window * 60 * 15),
                endtime=UTCDateTime(event_info["end_time"]) + (time_window * 60 * 15),
            )
        else:
            time_window = data_params["time_window"]
            st.trim(
                starttime=UTCDateTime(event_info["start_time"]) - (time_window * 60 * 2),
                endtime=UTCDateTime(event_info["end_time"]) + (time_window * 60 * 2),
            )
    return st


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

    for j in juldays:
        filename = f"{network}.{station}.{component}.{year}.{str(j).zfill(3)}.mseed"
        sub_path = os.path.join(
            f"{paths['DATA_DIR']}_{data_freq}", "Illgraben", str(year), station, component, filename
        )
        full_path = _resolve_path(sub_path)

        if full_path:
            st += read(full_path)
        else:
            raise FileNotFoundError(f"Seismic data file not found: {sub_path}")

    st.merge(method=1, fill_value="latest", interpolation_samples=0)
    print(f"Are we rescaling data: {rescale}")
    if rescale:
        if station == "ILL12":
            f, Q, v, n = 1, 25, 800, 0.5
            st = rescale_amplitude_by_distance(st, 90, 15, n, Q, v, (1, 30))
        elif station == "ILL13":
            f, Q, v, n = 1, 25, 800, 0.5
            st = rescale_amplitude_by_distance(st, 600, 15, n, Q, v, (1, 30))
    else:
        print("No rescaling applied to the data.")

    # Apply scaling to all traces in the stream
    for tr in st:
        tr.data = tr.data * 1e3

    return st


def load_data(
    event_id_list: list,
    station: str,
    year: int = 2019,
    trim: bool = True,
    abs: bool = True,
    env: bool = True,
    val: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function loads and concatenates seismic data for multiple events.
    """
    data_list = []
    time_list = []

    for event_id in event_id_list:
        st = load_seismic_data(event_id=str(event_id), station=station, year=year, trim=trim, val=val)

        # Extract data (envelope or raw) and times, skipping first sample to ensure alignment
        data = obspy.signal.filter.envelope(st[0].data)[1:] if env else st[0].data[1:]  # type: ignore
        times = st[0].times("matplotlib")[1:]  # type: ignore

        data_list.append(data)
        time_list.append(times)

    total_data = np.concatenate(data_list)
    total_times = np.concatenate(time_list)

    if abs:
        total_data = np.abs(total_data)
    return total_data, total_times


def load_data_test(
    julday_list: list, station: str, year: int = 2019, abs: bool = True, env: bool = True, rescale: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function loads and concatenates seismic data for multiple julian days for application.
    """
    data_list = []
    time_list = []

    for julday in julday_list:
        st = load_seismic_data_test(julday=julday, station=station, year=year, rescale=rescale)

        data = obspy.signal.filter.envelope(st[0].data)[1:] if env else st[0].data[1:]  # type: ignore
        times = st[0].times("matplotlib")[1:]  # type: ignore

        data_list.append(data)
        time_list.append(times)

    total_data = np.concatenate(data_list)
    total_times = np.concatenate(time_list)

    if abs:
        total_data = np.abs(total_data)
    return total_data, total_times


def load_label(
    event_id: str,
    station: str,
    interval_seconds: int,
    time_shift_minutes,
    trim: bool = True,
    smoothing: int | None = 30,
    divide_by: int | None = 45,
    val: bool = False,
) -> pd.DataFrame:
    """
    Loads and processes label data for a list of events or dates.

    This function can accept a list containing:
    - Event IDs (strings/integers that are keys in `time_config`), for which it will
      look up associated dates and time boundaries.
    - Raw date strings (e.g., "YYYY-MM-DD"), for which it will treat each string
      as a single date without `time_config` lookup.

    Args:
        event_id_list (list): A list of event IDs (from time_config) or date strings.
        station (str): Seismic station identifier.
        interval_seconds (int): Time interval in seconds for downsampling.
        time_shift_minutes (Union[int, str]): Time shift applied to the labels in minutes,
                                              or "raw" for UTC+0, or "average"/"dynamic".
        trim (bool): Whether to trim the data to event duration.
        smoothing (int | None): Smoothing window size for the target output. If 0 or None,
                                uses raw "Fv [kN]".
        divide_by (int | None): Value to divide the final labels by for normalization.

    Returns:
        pd.DataFrame: Concatenated and processed label data from all events/dates.
                      Contains 'Timestamp', 'Fv [kN]', and 'Fv std' columns.
    """
    time_window = data_params["time_window"]
    data_col = "Fv [kN]" if (not smoothing) else f"moving_avg_{smoothing}"
    time_col = "Time UTC+0" if time_shift_minutes == "raw" else "Time"

    # Resolve dates and time boundaries from event metadata or raw string
    info = time_config[event_id]
    dates = [info["date"]] if isinstance(info["date"], str) else info["date"]
    start_time = UTCDateTime(info["start_time"])
    end_time = UTCDateTime(info["end_time"])

    # Load available files for the given dates
    item_dfs = []
    for d in dates:
        sub_p = os.path.join(
            paths["UTC0_LABEL_DIR"] if time_shift_minutes == "raw" else f"{paths['LABEL_DIR']}_{time_shift_minutes}",
            station if time_shift_minutes != "raw" else "",
            f"{d}.csv",
        )
        fpath = _resolve_path(sub_p)
        if fpath:
            item_dfs.append(pd.read_csv(fpath))

    target = pd.concat(item_dfs).reset_index(drop=True)

    # Temporal trimming
    if trim:
        if val:
            st_limit = start_time - (time_window * 60 * 14)
            target = target[target[time_col].apply(UTCDateTime).between(st_limit, end_time + (time_window * 60 * 15))]
        else:
            st_limit = start_time - (time_window * 60 * 1)
            target = target[target[time_col].apply(UTCDateTime).between(st_limit, end_time + (time_window * 60 * 2))]

    # Feature engineering and scaling
    target["Timestamp"] = target[time_col].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)
    # target[data_col] /= 8 # Area conversion from Pressure to Force

    if not smoothing and "Fv max" in target.columns:
        mask = (target["Fv max"] - target[data_col]) / target[data_col] > 1
        target.loc[mask, "Fv max"] = target.loc[mask, data_col] * 2

    # Window-based downsampling
    num_w = len(target) // interval_seconds
    target = target.iloc[: num_w * interval_seconds]
    agg_data = {
        "Timestamp": target["Timestamp"].values[::interval_seconds],
        "Fv [kN]": np.mean(target[data_col].values.reshape(num_w, interval_seconds), axis=1),
        "Fv std": np.std(target[data_col].values.reshape(num_w, interval_seconds), axis=1),
    }
    if not smoothing:
        for col in ["Fv min", "Fv max"]:
            if col in target.columns:
                agg_data[col] = np.mean(target[col].values.reshape(num_w, interval_seconds), axis=1)
    target = pd.DataFrame(agg_data)
    if divide_by:
        target["Fv [kN]"] /= divide_by
    return target


# def load_label(event_id: str, station: str, interval_seconds: int, time_shift_minutes,
#                trim: bool = True, smoothing: int | None = 30, divide_by: int | None = 45, val:bool = False) -> pd.DataFrame:
#     """
#     Loads and processes label data for a list of events or dates.

#     This function can accept a list containing:
#     - Event IDs (strings/integers that are keys in `time_config`), for which it will
#       look up associated dates and time boundaries.
#     - Raw date strings (e.g., "YYYY-MM-DD"), for which it will treat each string
#       as a single date without `time_config` lookup.

#     Args:
#         event_id_list (list): A list of event IDs (from time_config) or date strings.
#         station (str): Seismic station identifier.
#         interval_seconds (int): Time interval in seconds for downsampling.
#         time_shift_minutes (Union[int, str]): Time shift applied to the labels in minutes,
#                                               or "raw" for UTC+0, or "average"/"dynamic".
#         trim (bool): Whether to trim the data to event duration.
#         smoothing (int | None): Smoothing window size for the target output. If 0 or None,
#                                 uses raw "Fv [kN]".
#         divide_by (int | None): Value to divide the final labels by for normalization.

#     Returns:
#         pd.DataFrame: Concatenated and processed label data from all events/dates.
#                       Contains 'Timestamp', 'Fv [kN]', and 'Fv std' columns.
#     """
#     time_window = 5
#     data_col = "Fv [kN]"
#     time_col = "Time UTC+0" if time_shift_minutes == "raw" else "Time"

#     # Resolve dates and time boundaries from event metadata or raw string
#     info = time_config[event_id]
#     dates = [info['date']] if isinstance(info['date'], str) else info['date']
#     start_time = UTCDateTime(info['start_time'])
#     end_time = UTCDateTime(info['end_time'])

#     # Load available files for the given dates
#     item_dfs = []
#     for d in dates:
#         sub_p = os.path.join(paths['UTC0_LABEL_DIR'] if time_shift_minutes == "raw" else f"{paths['LABEL_DIR']}_{time_shift_minutes}",
#                                 station if time_shift_minutes != "raw" else "", f"{d}.csv")
#         fpath = _resolve_path(sub_p)
#         if fpath: item_dfs.append(pd.read_csv(fpath))

#     target = pd.concat(item_dfs).reset_index(drop=True)

#     # Temporal trimming
#     if trim:
#         if val:
#             st_limit = start_time - (time_window * 60 * 14)
#             target = target[target[time_col].apply(UTCDateTime).between(st_limit, end_time + (time_window * 60 * 15))]
#         else:
#             st_limit = start_time - (time_window * 60 * 1)
#             target = target[target[time_col].apply(UTCDateTime).between(st_limit, end_time + (time_window * 60 * 2))]

#     # Feature engineering and scaling
#     target['Timestamp'] = target[time_col].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)
#     target = compute_delta_fmv(target, time_col='Timestamp', value_col=data_col, fs=1, window_seconds=interval_seconds, quantile=0.25)
#     if smoothing:
#         target[data_col] = smooth_series(target[data_col], **{"window": smoothing, "center": True})

#     # Window-based downsampling
#     num_w = len(target) // interval_seconds
#     target = target.iloc[:num_w * interval_seconds]
#     agg_data = {
#         'Timestamp': target['Timestamp'].values[::interval_seconds],
#         'Fv [kN]': np.nanquantile(target[data_col].values.reshape(num_w, interval_seconds), q=0.5, axis=1),
#         'Fv std': np.nanquantile(target[data_col].values.reshape(num_w, interval_seconds), q=0.5, axis=1)
#     }
#     target = pd.DataFrame(agg_data)
#     if divide_by:
#         target['Fv [kN]'] /= divide_by
#     return target
