"""
autocorr_metrics.py

Modular tools for computing MSE, RMSE, and MAE on autocorrelated time-series
predictions (e.g. debris-flow impact-force forecasts), while correctly
accounting for the fact that autocorrelated data does NOT contain as much
independent information as its raw sample size suggests.

Key idea
--------
Autocorrelation does not bias the point estimates (MSE/RMSE/MAE are still
correct averages). It biases any naive (iid-assumed) standard error or
confidence interval built on top of them, making them look far more
precise than they really are. This module reports HAC (Newey-West) and
block-bootstrap corrected uncertainty alongside the usual point estimates.

Usage
-----
    from autocorr_metrics import analyze_predictions

    result = analyze_predictions("LSTM_t172_v207.csv")
    print(result["metrics"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


# ======================================================================
# 1. Data loading
# ======================================================================

def load_predictions(
    path: str,
    time_col: str = "Timestamps",
    target_col: str = "Output",
    pred_col: str = "Predicted_Output",
) -> pd.DataFrame:
    """Load a (Target, Predicted) CSV into a sorted, time-indexed DataFrame."""
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    df = df[[target_col, pred_col]].rename(
        columns={target_col: "Target", pred_col: "Predicted"}
    )
    return df.dropna()


# ======================================================================
# 2. Autocorrelation diagnostics
# ======================================================================

def autocorr_diagnostics(x: np.ndarray, lb_lags=(5, 10, 20)) -> dict:
    """
    Diagnose how autocorrelated a series is, and what that implies for
    effective sample size.

    Returns: n, lag1_autocorr, n_eff_ar1, adf_pvalue, is_stationary,
             ljung_box (DataFrame of stat/p-value per requested lag)
    """
    x = np.asarray(x)
    n = len(x)

    lag1 = pd.Series(x).autocorr(1)
    n_eff = n * (1 - lag1) / (1 + lag1) if lag1 is not None and -1 < lag1 < 1 else np.nan

    try:
        adf_p = adfuller(x)[1]
    except Exception:
        adf_p = np.nan

    valid_lags = [l for l in lb_lags if l < n // 2]
    lb = acorr_ljungbox(x, lags=valid_lags, return_df=True) if valid_lags else None

    return {
        "n": n,
        "lag1_autocorr": lag1,
        "n_eff_ar1": n_eff,
        "adf_pvalue": adf_p,
        "is_stationary": (adf_p < 0.05) if not np.isnan(adf_p) else None,
        "ljung_box": lb,
    }


# ======================================================================
# 3. HAC (Newey-West) standard error of a mean
# ======================================================================

def _default_hac_maxlag(n: int) -> int:
    """Newey-West rule-of-thumb bandwidth."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def hac_se_of_mean(x: np.ndarray, maxlag: int | None = None) -> tuple[float, int]:
    """
    Newey-West (HAC) standard error of mean(x) for an autocorrelated series,
    using a Bartlett kernel. Returns (se, maxlag_used).
    """
    x = np.asarray(x)
    n = len(x)
    if maxlag is None:
        maxlag = _default_hac_maxlag(n)
    maxlag = min(maxlag, n - 1)

    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    var = gamma0
    for lag in range(1, maxlag + 1):
        w = 1 - lag / (maxlag + 1)  # Bartlett weight
        gamma = np.dot(xc[lag:], xc[:-lag]) / n
        var += 2 * w * gamma

    se = np.sqrt(max(var, 0.0) / n)
    return se, maxlag


# ======================================================================
# 4. Block bootstrap CI (nonparametric alternative to HAC)
# ======================================================================

def block_bootstrap_ci(
    x: np.ndarray,
    block_len: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, int]:
    """
    Moving block bootstrap confidence interval for mean(x).
    block_len defaults to the HAC maxlag (i.e. the estimated decorrelation
    timescale), which keeps blocks long enough to preserve local dependence.
    """
    x = np.asarray(x)
    n = len(x)
    if block_len is None:
        block_len = _default_hac_maxlag(n)
    block_len = max(1, min(block_len, n - 1))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_len, size=n_blocks)
        sample = np.concatenate([x[s:s + block_len] for s in starts])[:n]
        boot_means[b] = sample.mean()

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, block_len


# ======================================================================
# 5. Metric-specific autocorrelation-corrected uncertainty
# ======================================================================

def _metric_with_ci(values: np.ndarray, transform=lambda m: m) -> dict:
    """
    Generic helper: given a per-timestep loss array (e.g. squared errors
    for MSE, absolute errors for MAE), compute the point estimate plus
    naive, HAC, and block-bootstrap 95% CIs. `transform` is applied to
    convert the mean (and its CI bounds) to the final reported scale,
    e.g. sqrt() for turning MSE into RMSE.
    """
    n = len(values)
    point = values.mean()

    se_naive = values.std(ddof=1) / np.sqrt(n)
    se_hac, maxlag = hac_se_of_mean(values)
    boot_lo, boot_hi, block_len = block_bootstrap_ci(values)

    return {
        "point_estimate": transform(point),
        "ci_naive": (transform(point - 1.96 * se_naive), transform(point + 1.96 * se_naive)),
        "ci_hac": (transform(max(point - 1.96 * se_hac, 0)), transform(point + 1.96 * se_hac)),
        "ci_block_bootstrap": (transform(max(boot_lo, 0)), transform(boot_hi)),
        "hac_se": se_hac,
        "naive_se": se_naive,
        "hac_widening_factor": se_hac / se_naive if se_naive > 0 else np.nan,
        "hac_maxlag": maxlag,
        "block_len": block_len,
    }


def compute_metrics_with_ci(target: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute MSE, RMSE, and MAE with autocorrelation-corrected 95% CIs.

    RMSE's CI is derived by taking sqrt() of the MSE mean's CI bounds
    (the delta method would work too, but for a one-sided monotonic
    transform like sqrt, directly transforming the CI bounds is simpler
    and accurate here).
    """
    target = np.asarray(target)
    pred = np.asarray(pred)
    resid = target - pred
    sq_err = resid ** 2
    abs_err = np.abs(resid)

    mse_result = _metric_with_ci(sq_err)
    rmse_result = _metric_with_ci(sq_err, transform=np.sqrt)  # same underlying series, sqrt-transformed
    mae_result = _metric_with_ci(abs_err)

    return {
        "n": len(target),
        "MSE": mse_result,
        "RMSE": rmse_result,
        "MAE": mae_result,
    }


# ======================================================================
# 6. Full pipeline
# ======================================================================

def analyze_predictions(
    path: str,
    time_col: str = "Timestamps",
    target_col: str = "Output",
    pred_col: str = "Predicted_Output",
    label: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: load a (Target, Predicted) CSV, run autocorrelation
    diagnostics on the residuals, and compute MSE/RMSE/MAE with
    autocorrelation-corrected confidence intervals.
    """
    df = load_predictions(path, time_col, target_col, pred_col)
    target = df["Target"].values
    pred = df["Predicted"].values
    resid = target - pred

    diagnostics = autocorr_diagnostics(resid)
    metrics = compute_metrics_with_ci(target, pred)

    result = {
        "label": label or path,
        "diagnostics": diagnostics,
        "metrics": metrics,
    }

    if verbose:
        _print_summary(result)

    return result


def _print_summary(result: dict) -> None:
    diag = result["diagnostics"]
    m = result["metrics"]

    print(f"\n===== Autocorrelation-corrected metrics: {result['label']} =====")
    print(f"n = {diag['n']}")
    print(f"Residual lag-1 autocorrelation: {diag['lag1_autocorr']:.4f}")
    print(f"Effective sample size (AR1 approx): {diag['n_eff_ar1']:.1f}")
    print(f"ADF p-value (residual stationarity): {diag['adf_pvalue']:.4g}"
          f"  -> {'stationary' if diag['is_stationary'] else 'non-stationary'}")
    if diag["ljung_box"] is not None:
        print("Ljung-Box test on residuals:")
        print(diag["ljung_box"].to_string())

    print()
    for name in ("MSE", "RMSE", "MAE"):
        r = m[name]
        lo_n, hi_n = r["ci_naive"]
        lo_h, hi_h = r["ci_hac"]
        lo_b, hi_b = r["ci_block_bootstrap"]
        print(f"{name}: {r['point_estimate']:.5f}")
        print(f"   naive 95% CI          : [{lo_n:.5f}, {hi_n:.5f}]")
        print(f"   HAC 95% CI            : [{lo_h:.5f}, {hi_h:.5f}]  "
              f"({r['hac_widening_factor']:.1f}x wider than naive)")
        print(f"   block-bootstrap 95% CI: [{lo_b:.5f}, {hi_b:.5f}]")


# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    result = analyze_predictions(
        "/mnt/user-data/uploads/LSTM_t172_v207.csv",
        label="LSTM t172_v207",
    )
