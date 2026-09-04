#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-01
# __author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
# __find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

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

sys.path.append(paths.get("BASE_DIR", str(project_root)))

plt.rcParams.update(
    {
        "font.size": 7,  # Set global font size
        "font.family": "Arial",  # Set global font family
        "legend.fontsize": 8,  # Set legend font size
        "figure.figsize": (8, 5),  # Set figure size in inches
    }
)


# ----------------------------------------------------------------------
# Autocorrelation-aware evaluation helpers
#
# Debris-flow impact-force series are strongly autocorrelated (a single
# rise/decay event, not iid noise), which means the *count* of timesteps
# hugely overstates the number of independent observations behind a
# metric like MSE. These helpers quantify that and correct the
# uncertainty on MSE accordingly, without changing the point estimate
# itself.
# ----------------------------------------------------------------------


def _default_hac_maxlag(n):
    """Newey-West rule-of-thumb bandwidth."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def _hac_se_of_mean(x, maxlag=None):
    """Newey-West (HAC) standard error of the mean of x, Bartlett kernel."""
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return np.nan, 0
    if maxlag is None:
        maxlag = _default_hac_maxlag(n)
    maxlag = min(maxlag, n - 1)

    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    var = gamma0
    for lag in range(1, maxlag + 1):
        w = 1 - lag / (maxlag + 1)
        gamma = np.dot(xc[lag:], xc[:-lag]) / n
        var += 2 * w * gamma

    se = np.sqrt(max(var, 0.0) / n)
    return se, maxlag


def _autocorr_diagnostics(resid, lb_lag=10):
    """
    Lightweight autocorrelation diagnostics on a residual series:
    lag-1 autocorrelation, AR(1)-approx effective sample size,
    ADF stationarity p-value, and Ljung-Box p-value.
    Returns NaNs gracefully if the sample is too small for a given test.
    """
    resid = np.asarray(resid)
    n = len(resid)
    out = {"lag1_autocorr": np.nan, "n_eff_ar1": np.nan, "adf_pvalue": np.nan, "lb_pvalue": np.nan}
    if n < 10:
        return out

    lag1 = pd.Series(resid).autocorr(1)
    out["lag1_autocorr"] = lag1
    if lag1 is not None and -1 < lag1 < 1:
        out["n_eff_ar1"] = n * (1 - lag1) / (1 + lag1)

    try:
        out["adf_pvalue"] = adfuller(resid)[1]  # type: ignore
    except Exception:
        pass

    try:
        this_lag = max(1, min(lb_lag, n // 5))
        lb = acorr_ljungbox(resid, lags=[this_lag], return_df=True)
        out["lb_pvalue"] = lb["lb_pvalue"].iloc[0]
    except Exception:
        pass

    return out


def _get_metrics_string(model_type, time_to_train, test_julday, val_julday, interval_seconds, y_true, y_pred):
    """Calculates evaluation metrics and returns a CSV-formatted string."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    resid = y_true - y_pred
    sq_err = resid**2
    abs_err = np.abs(resid)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # --- Autocorrelation-aware uncertainty for MSE, RMSE, and MAE ---
    # Point estimates above are correct regardless of autocorrelation.
    # What autocorrelation breaks is the *uncertainty* on them: naive
    # iid-based SEs badly understate it. We report the HAC (Newey-West)
    # standard error and 95% CI for each metric instead, plus diagnostics
    # on the residual series itself.
    diag = _autocorr_diagnostics(resid)

    mse_hac_se, hac_maxlag = _hac_se_of_mean(sq_err)
    mse_ci_low = max(mse - 1.96 * mse_hac_se, 0)
    mse_ci_high = mse + 1.96 * mse_hac_se

    # RMSE CI via sqrt-transform of the MSE CI bounds; RMSE HAC SE via the
    # delta method (propagating the MSE HAC SE through sqrt).
    rmse_ci_low = np.sqrt(mse_ci_low)
    rmse_ci_high = np.sqrt(mse_ci_high)
    rmse_hac_se = mse_hac_se / (2 * rmse) if rmse > 0 else np.nan

    mae_hac_se, mae_maxlag = _hac_se_of_mean(abs_err)
    mae_ci_low = mae - 1.96 * mae_hac_se
    mae_ci_high = mae + 1.96 * mae_hac_se

    return (
        f"{model_type},{time_to_train},{test_julday},{val_julday},{interval_seconds},"
        f"{mse:.4f},{rmse:.4f},{mae:.4f},"
        f"{diag['lag1_autocorr']:.4f},{diag['n_eff_ar1']:.1f},{diag['adf_pvalue']:.4g},{diag['lb_pvalue']:.4g},"
        f"{mse_hac_se:.4f},{mse_ci_low:.4f},{mse_ci_high:.4f},"
        f"{rmse_hac_se:.4f},{rmse_ci_low:.4f},{rmse_ci_high:.4f},"
        f"{mae_hac_se:.4f},{mae_ci_low:.4f},{mae_ci_high:.4f}\n"
    )


def evaluate_model(
    model_type: str,
    test_id: str,
    val_id: str,
    interval_seconds: int,
    y_true,
    y_pred,
    smoothing: int,
    out_dir: str,
    time_to_train: str,
):
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

    test_julday = test_info["julday"] if isinstance(test_info["julday"], int) else test_info["julday"][0]
    val_julday = val_info["julday"] if isinstance(val_info["julday"], int) else val_info["julday"][0]

    output_dir = f"{out_dir}/model_evaluation"
    dist_dir = f"{out_dir}/dist_plots/test/{interval_seconds}/{test_julday}"
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/evaluation_output_wo_noise.txt"
    filename2 = f"{output_dir}/evaluation_output_constrained.txt"

    header = (
        "Model,Config,Time_To_Train,Test,Val,Interval,MSE,RMSE,MAE,"
        "Lag1_Autocorr,N_eff_AR1,ADF_pvalue,LjungBox_pvalue,"
        "MSE_HAC_SE,MSE_CI_HAC_low,MSE_CI_HAC_high,"
        "RMSE_HAC_SE,RMSE_CI_HAC_low,RMSE_CI_HAC_high,"
        "MAE_HAC_SE,MAE_CI_HAC_low,MAE_CI_HAC_high\n"
    )
    for fname in [filename, filename2]:
        if not os.path.exists(fname):
            with open(fname, "w") as file:
                file.write(header)

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    # 1. Constrained Evaluation (Full Data)
    metrics_constrained = _get_metrics_string(
        model_type, time_to_train, test_julday, val_julday, interval_seconds, y_true, y_pred
    )
    with open(filename2, "a") as f:
        f.write(metrics_constrained)

    # 2. Evaluation without Noise (True Impact Force >= 2 kN)
    mask = y_true >= 2
    y_true_no_noise = y_true[mask]
    y_pred_no_noise = y_pred[mask]

    if len(y_true_no_noise) > 0:
        metrics_wo_noise = _get_metrics_string(
            model_type, time_to_train, test_julday, val_julday, interval_seconds, y_true_no_noise, y_pred_no_noise
        )
        with open(filename, "a") as f:
            f.write(metrics_wo_noise)
