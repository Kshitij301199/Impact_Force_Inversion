#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2026-02-01
#__author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
#__find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
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

def _resolve_path(sub_path):
    """Helper to check both HPC and Local base directories."""
    for base in [paths.get('BASE_DIR'), paths.get('LOCAL_BASE_DIR')]:
        if base:
            full_path = os.path.join(base, sub_path)
            if os.path.exists(full_path):
                return full_path
    return None


def load_seismic_data(event_id:str|int, station:str, 
                      year:int=2019, component:str='EHZ', network:str="9S", 
                      trim:bool = True) -> Stream:
    event_info = time_config[str(event_id)]
    
    # Delegate loading to the standardized test loader
    st = load_seismic_data_test(
        julday=event_info['julday'], 
        station=station, 
        year=year, 
        component=component, 
        network=network,
        freq=data_params['fmax']
    )

    if trim:
        time_window = data_params['time_window']
        st.trim(starttime=UTCDateTime(event_info['start_time']) - (time_window * 60 * 2), 
                endtime=UTCDateTime(event_info['end_time']) + (time_window * 60 * 2))
    return st

def load_seismic_data_test(julday:int|str|list, station:str, 
                      year:int=2019, component:str='EHZ', network:str="9S", freq = None,
                      ) -> Stream:
    st = Stream()
    juldays = [julday] if isinstance(julday, (int, str)) else julday
    data_freq = data_params['fmax'] if freq is None else freq
    
    for j in juldays:
        filename = f"{network}.{station}.{component}.{year}.{str(j).zfill(3)}.mseed"
        sub_path = os.path.join(f"{paths['DATA_DIR']}_{data_freq}", "Illgraben", str(year), station, component, filename)
        full_path = _resolve_path(sub_path)
        
        if full_path:
            st += read(full_path)
        else:
            raise FileNotFoundError(f"Seismic data file not found: {sub_path}")

    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    
    # Apply scaling to all traces in the stream
    for tr in st:
        tr.data = tr.data * 1e3
        
    return st

def load_data(event_id_list:list, station:str, year:int=2019, trim:bool=True, abs:bool=True, env:bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    This function loads and concatenates seismic data for multiple events.
    """
    data_list = []
    time_list = []
    
    for event_id in event_id_list:
        st = load_seismic_data(event_id = str(event_id), station= station, year=year, trim= trim)
        
        # Extract data (envelope or raw) and times, skipping first sample to ensure alignment
        data = obspy.signal.filter.envelope(st[0].data)[1:] if env else st[0].data[1:]
        times = st[0].times("matplotlib")[1:]
        
        data_list.append(data)
        time_list.append(times)

    total_data = np.concatenate(data_list)
    total_times = np.concatenate(time_list)
    
    if abs:
        total_data = np.abs(total_data)
    return total_data, total_times

def load_data_test(julday_list:list, station:str, year:int=2019, abs:bool=True, env:bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    This function loads and concatenates seismic data for multiple julian days for application.
    """
    data_list = []
    time_list = []
    
    for julday in julday_list:
        st = load_seismic_data_test(julday = julday, station= station, year=year)
        
        data = obspy.signal.filter.envelope(st[0].data)[1:] if env else st[0].data[1:]
        times = st[0].times("matplotlib")[1:]
        
        data_list.append(data)
        time_list.append(times)

    total_data = np.concatenate(data_list)
    total_times = np.concatenate(time_list)
    
    if abs:
        total_data = np.abs(total_data)
    return total_data, total_times

def load_label(event_id_list: list, station: str, interval_seconds: int, time_shift_minutes, 
               trim: bool = True, smoothing: int | None = 30, divide_by: int | None = 45) -> pd.DataFrame:
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
    time_window = data_params['time_window']
    data_col = "Fv [kN]" if (not smoothing) else f"moving_avg_{smoothing}"
    time_col = "Time UTC+0" if time_shift_minutes == "raw" else "Time"
    
    label_dfs = []
    for i, item in enumerate(event_id_list):
        item_str = str(item)
        
        # Resolve dates and time boundaries from event metadata or raw string
        if item_str in time_config:
            info = time_config[item_str]
            dates = [info['date']] if isinstance(info['date'], str) else info['date']
            start_time = UTCDateTime(info['start_time'])
            end_time = UTCDateTime(info['end_time'])
            is_event = True
        else:
            dates = [item_str]
            start_time = UTCDateTime(item_str) + (time_window * 60)
            end_time = None
            is_event = False

        # Load available files for the given dates
        item_dfs = []
        for d in dates:
            sub_p = os.path.join(paths['UTC0_LABEL_DIR'] if time_shift_minutes == "raw" else f"{paths['LABEL_DIR']}_{time_shift_minutes}", 
                                 station if time_shift_minutes != "raw" else "", f"{d}.csv")
            fpath = _resolve_path(sub_p)
            if fpath: item_dfs.append(pd.read_csv(fpath))

        if not item_dfs: continue
        target = pd.concat(item_dfs).reset_index(drop=True)

        # Temporal trimming
        if trim:
            if is_event:
                st_limit = start_time - (time_window * 60 * (1 if i == 0 else 2))
                target = target[target[time_col].apply(UTCDateTime).between(st_limit, end_time + (time_window * 60 * 2))]
            elif i == 0:
                target = target[target[time_col].apply(UTCDateTime) >= start_time]

        # Feature engineering and scaling
        target['Timestamp'] = target[time_col].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)
        target[data_col] /= 8 # Area conversion from Pressure to Force
        
        if not smoothing:
            for col in ["Fv min", "Fv max"]:
                if col in target.columns: target[col] /= 8
            if "Fv max" in target.columns:
                mask = (target["Fv max"] - target[data_col]) / target[data_col] > 1
                target.loc[mask, "Fv max"] = target.loc[mask, data_col] * 2

        # Window-based downsampling
        if interval_seconds != 1:
            num_w = len(target) // interval_seconds
            target = target.iloc[:num_w * interval_seconds]
            agg_data = {
                'Timestamp': target['Timestamp'].values[::interval_seconds],
                'Fv [kN]': np.mean(target[data_col].values.reshape(num_w, interval_seconds), axis=1),
                'Fv std': np.std(target[data_col].values.reshape(num_w, interval_seconds), axis=1)
            }
            if not smoothing:
                for col in ["Fv min", "Fv max"]:
                    if col in target.columns:
                        agg_data[col] = np.mean(target[col].values.reshape(num_w, interval_seconds), axis=1)
            target = pd.DataFrame(agg_data)
        else:
            target = pd.DataFrame({
                'Timestamp': target['Timestamp'].values,
                'Fv [kN]': target[data_col].values,
                'Fv std': target.get('Fv std', 0)
            })
        label_dfs.append(target)

    if not label_dfs: return pd.DataFrame()
    total_target = pd.concat(label_dfs).reset_index(drop=True)
    if divide_by: total_target['Fv [kN]'] /= divide_by
    return total_target

def load_label2(date_list: list, station: str, interval_seconds: int, time_shift_minutes, 
                smoothing: int | None = 30, divide_by: int | None = 350) -> pd.DataFrame:
    """
    Independent implementation of load_label2 optimized for processing date lists.
    """
    time_window = data_params['time_window']
    data_col = "Fv [kN]" if (not smoothing) else f"moving_avg_{smoothing}"
    time_col = "Time UTC+0" if time_shift_minutes == "raw" else "Time"
    
    label_dfs = []
    for i, date in enumerate(date_list):
        dates = [date] if isinstance(date, str) else date
        
        day_dfs = []
        for d in dates:
            sub_p = os.path.join(
                paths['UTC0_LABEL_DIR'] if time_shift_minutes == "raw" else f"{paths['LABEL_DIR']}_{time_shift_minutes}",
                station if time_shift_minutes != "raw" else "", 
                f"{d}.csv"
            )
            fpath = _resolve_path(sub_p)
            if fpath: day_dfs.append(pd.read_csv(fpath))
        
        if not day_dfs: continue
        target = pd.concat(day_dfs).reset_index(drop=True)

        # Original logic: Trim the start of the first date provided in the list
        if i == 0:
            start_ref = dates[0]
            target = target[target[time_col].apply(UTCDateTime) >= UTCDateTime(start_ref) + (time_window * 60)]

        # Vectorized feature engineering and scaling
        target['Timestamp'] = target[time_col].apply(UTCDateTime).apply(lambda x: x.timestamp)
        target[data_col] /= 8 # Area conversion
        
        # Optimized window-based downsampling
        if interval_seconds != 1:
            num_w = len(target) // interval_seconds
            vals = target[data_col].values[:num_w * interval_seconds].reshape(num_w, interval_seconds)
            target = pd.DataFrame({
                'Timestamp': target['Timestamp'].values[::interval_seconds][:num_w],
                'Fv [kN]': np.mean(vals, axis=1),
                'Fv std': np.std(vals, axis=1)
            })
        else:
            target = pd.DataFrame({
                'Timestamp': target['Timestamp'].values,
                'Fv [kN]': target[data_col].values,
                'Fv std': target['Fv std'].values if 'Fv std' in target.columns else np.zeros(len(target))
            })
        label_dfs.append(target)

    if not label_dfs: return pd.DataFrame()
    total_target = pd.concat(label_dfs).reset_index(drop=True)
    if divide_by: total_target['Fv [kN]'] /= divide_by
    return total_target
