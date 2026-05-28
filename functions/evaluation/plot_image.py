#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2026-02-01
#__author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
#__find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from obspy import UTCDateTime

# Dynamic path resolution
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

def _load_config(filename):
    path = project_root / "config" / filename
    with open(path, "r") as f:
        return json.load(f)

try:
    paths = _load_config("paths.json")
    time_config = _load_config("event_id_map.json")
except FileNotFoundError:
    # Fallback for HPC or different environments
    paths = {"BASE_DIR": str(project_root)}
    time_config = {}

sys.path.append(paths.get('BASE_DIR', str(project_root)))

import matplotlib.font_manager as fm
font_dirs = [os.path.join(paths.get('BASE_DIR', ''), 'fonts/arial')]
if os.path.exists(font_dirs[0]):
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    plt.rcParams['font.family'] = 'Arial'
else:
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'legend.fontsize': 6,        # Set legend font size
    'figure.figsize': (7, 3.5) # Set figure size in inches
})

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # Increase the chunk size limit
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5  # Adjust this value if needed

from functions.data_processing.read_data import load_label

def plot_image(st, predicted_output, target_output, timestamps,
                image_dir:str, test_id, val_id, interval, trim=True, smoothing=30):    
    """Plot the seismogram along with target and predicted impact forces.
     Args:
        st (obspy.Stream): Seismogram data.
        predicted_output (list): List of predicted impact forces.
        target_output (list): List of target impact forces.
        timestamps (list): List of timestamps corresponding to the data points.
        image_dir (str): Directory to save the plot.
        test_id (int): Test event ID.
        val_id (int): Validation event ID.
        interval (str): Time interval for the data.
        trim (bool): Whether to trim the data to the event duration.
        smoothing (int): Smoothing window size for the target output.
    Returns:
        None
    """
    print(f"{'Plotting Image':-^30}")
    
    # Flatten inputs once
    flat_timestamps = np.concatenate(timestamps)
    times = np.array([UTCDateTime(t).matplotlib_date for t in flat_timestamps])
    target_output = np.concatenate(target_output)
    predicted_output = np.concatenate(predicted_output)
    
    test_id_str, val_id_str = str(test_id), str(val_id)
    test_info = time_config[test_id_str]
    val_info = time_config[val_id_str]
    
    test_julday = test_info['julday'] if isinstance(test_info['julday'], int) else test_info['julday'][0]
    val_julday = val_info['julday'] if isinstance(val_info['julday'], int) else val_info['julday'][0]
    
    zero_label = load_label([test_id], "ILL11", interval, "average", trim, smoothing=None, divide_by=None)
    
    if trim:
        start_time, end_time = UTCDateTime(test_info['start_time']), UTCDateTime(test_info['end_time'])
        mat_start_time = start_time.matplotlib_date
        mat_end_time = end_time.matplotlib_date
        
        # Using searchsorted for O(log N) search and safety against floating point mismatch
        idx_start = np.searchsorted(times, mat_start_time)
        idx_end = np.searchsorted(times, mat_end_time)
        
        times = times[idx_start: idx_end]
        target_output = target_output[idx_start: idx_end]
        predicted_output = predicted_output[idx_start: idx_end]

    fig, ax1 = plt.subplots(1,1)
    ax1.plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5, linewidth=1)
    ax1.set_ylabel(r"Amplitude (mm/s)");
    ax1.set_ylim(-2, 2);
    ax = ax1.twinx()
    ax.plot(times, target_output, label="Impact Force Target [kN]", alpha=0.8, color='r',linewidth=1)
    ax.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
    times2 = [UTCDateTime(t).matplotlib_date for t in zero_label['Timestamp'].to_numpy()]
    ax.plot(times2, zero_label['Fv [kN]'], label="Without Smoothing", alpha=0.5, color="green", linewidth=0.5)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("Normal Force [kN]");
    ax.set_ylim(0,45);
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f"{image_dir}/{test_julday}_{val_julday}_{interval}.png", dpi=300)
    plt.close()
    return None

def plot_image_test(st, predicted_output, timestamps,
                        image_dir:str, julday:str):
    """Plot the seismogram along with predicted impact forces for test data.
     Args:
        st (obspy.Stream): Seismogram data.
        predicted_output (list): List of predicted impact forces.
        timestamps (list): List of timestamps corresponding to the data points.
        image_dir (str): Directory to save the plot.
        julday (str): Julian day identifier for the event.
    Returns:
        None
    """
    print(f"{'Plotting Image':-^30}")
    
    flat_preds = np.concatenate(predicted_output)
    flat_ts = np.concatenate(timestamps)
    times = [UTCDateTime(t).matplotlib_date for t in flat_ts]
    
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
    ax1.set_ylabel(r"Amplitude (mm/s)");
    ax1.set_ylim(-1.7, 1.7);
    ax = ax1.twinx()
    ax.plot(times, flat_preds, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("Normal Force [kN]");
    ax.set_ylim(0,50);
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f"{image_dir}/{julday}.png", dpi=300)
    plt.close()
    return None