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
from obspy.core import UTCDateTime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Dynamic path resolution for configuration files
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
    # Fallback to current directory root if config is missing
    paths = {"BASE_DIR": str(project_root)}
    time_config = {}

sys.path.append(paths.get('BASE_DIR', str(project_root)))

plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 8,        # Set legend font size
    'figure.figsize': (8, 5) # Set figure size in inches
})

from functions.data_processing.read_data import load_label

def _get_metrics_string(model_type, time_to_train, test_julday, val_julday, interval_seconds, y_true, y_pred):
    """Calculates evaluation metrics and returns a CSV-formatted string."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation and Lag analysis
    p_corr, _ = pearsonr(y_true, y_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    # Use cross-correlation to find temporal lag
    corr_full = np.correlate(y_true_centered, y_pred_centered, mode='full')
    lag = np.argmax(corr_full) - (len(y_true) - 1)

    # Histogram Similarity Metric: Weighted by bin center
    bins = np.arange(3, 45, 2)
    h1, edges = np.histogram(y_true, bins=bins, density=True)
    h2, _ = np.histogram(y_pred, bins=bins, density=True)
    centers = edges[:-1] + np.diff(edges) / 2
    hist_wmse = np.dot(centers, np.sqrt((h2 - h1) ** 2)) / len(bins)

    return f"{model_type},{time_to_train},{test_julday},{val_julday},{interval_seconds},{mse:.4f},{rmse:.4f},{mae:.4f},{r2:.4f},{lag:.4f},{p_corr:.4f},{hist_wmse:.4f}\n"

def evaluate_model(model_type:str, test_id:str, val_id:str, interval_seconds:int, y_true, y_pred, smoothing:int, out_dir:str, time_to_train:str):
    """Evaluate the model performance and save the results.
    Args:
        model_type (str): Type of the model (e.g., 'LSTM', 'xLSTM').
        test_id (str): Test event ID.
        val_id (str): Validation event ID.
        interval_seconds (int): Time interval in seconds.
        y_true (list): List of true impact forces.
        y_pred (list): List of predicted impact forces.
        smoothing (int): Smoothing window size for the target output.
        out_dir (str): Output directory to save the evaluation results.
        time_to_train (str): Time taken to train the model.
    Returns:
        None
    """
    print(f"{'Evaluating Model':-^50}")
    test_info = time_config[test_id]
    val_info = time_config[val_id]

    test_julday = test_info['julday'] if isinstance(test_info['julday'], int) else test_info['julday'][0]
    val_julday = val_info['julday'] if isinstance(val_info['julday'], int) else val_info['julday'][0]

    output_dir = f"{out_dir}/model_evaluation"
    dist_dir = f"{out_dir}/dist_plots/test/{interval_seconds}/{test_julday}"
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/evaluation_output_wo_noise.txt"
    filename2 = f"{output_dir}/evaluation_output_constrained.txt"

    header = "Model,Config,Time_To_Train,Test,Val,Interval,MSE,RMSE,MAE,R2,Corr,PearsonR,Hist_WMSE\n"
    for fname in [filename, filename2]:
        if not os.path.exists(fname):
            with open(fname, "w") as file:
                file.write(header)

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    # 1. Constrained Evaluation (Full Data)
    metrics_constrained = _get_metrics_string(model_type, time_to_train, test_julday, val_julday, interval_seconds, y_true, y_pred)
    with open(filename2, "a") as f:
        f.write(metrics_constrained)

    # Save distribution plot for constrained case
    fig, ax = plt.subplots()
    bins = np.arange(3, 45, 2)
    ax.hist(y_true, bins=bins, color='red', alpha=0.8, label="Impact Force [kN]", density=True)
    ax.hist(y_pred, bins=bins, color='blue', alpha=0.6, label="Model Prediction", density=True)
    ax.set_xlabel("Normal Force [kN]")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{model_type} {interval_seconds} test {test_julday} val {val_julday}")
    ax.legend(loc='best')
    fig.tight_layout()
    # Clean up model name for plotting if it contains config parameters
    plot_model_name = model_type.split(',')[0]
    fig.savefig(f"{dist_dir}/{plot_model_name}_{val_julday}.png", dpi=300)
    plt.close(fig)

    # 2. Evaluation without Noise (True Impact Force >= 3 kN)
    mask = y_true >= 3
    y_true_no_noise = y_true[mask]
    y_pred_no_noise = y_pred[mask]

    if len(y_true_no_noise) > 0:
        metrics_wo_noise = _get_metrics_string(model_type, time_to_train, test_julday, val_julday, interval_seconds, y_true_no_noise, y_pred_no_noise)
        with open(filename, "a") as f:
            f.write(metrics_wo_noise)

    return None

def sanity_check_train(target, pred, model_type, interval_seconds, test_julday, val_julday, out_dir):
    dist_dir = f"{out_dir}/dist_plots/train/{interval_seconds}/{test_julday}"
    os.makedirs(dist_dir, exist_ok=True)
    fig, ax = plt.subplots()
    bins = np.arange(1, 51, 5)
    ax.hist(target, bins=bins, color='red', alpha=0.8, label="Impact Force [kN]")
    ax.hist(pred, bins=bins, color='blue', alpha=0.6, label="Model Prediction")
    ax.set_xlabel("Normal Force [kN]")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_type} {interval_seconds} test {test_julday} val {val_julday}")
    ax.legend(loc='best')
    fig.tight_layout()
    plot_model_name = model_type.split(',')[0]
    fig.savefig(f"{dist_dir}/{plot_model_name}_{val_julday}.png", dpi=300)
    plt.close(fig)
    return None