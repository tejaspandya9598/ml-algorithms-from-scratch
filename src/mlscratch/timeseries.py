"""
Time-series tools from scratch (HW1): multiplicative decomposition, an Augmented
Dickey-Fuller stationarity test, and a lagged feature-matrix builder.
"""
from __future__ import annotations

import numpy as np


def decompose_multiplicative(series_values, period=4):
    n = len(series_values)
    trend = np.full(n, np.nan)
    
    # Calculate Trend Component (Centered Moving Average for s=4)
    # Weights: [0.5, 1, 1, 1, 0.5] / 4
    for i in range(2, n - 2):
        trend[i] = (0.5*series_values[i-2] + 
                    1.0*series_values[i-1] + 
                    1.0*series_values[i] + 
                    1.0*series_values[i+1] + 
                    0.5*series_values[i+2]) / period
        
    # Isolating Seasonality + Residuals (Multiplicative: Data / Trend)
    detrended = series_values / trend
    
    # Calculating Seasonal Component
    seasonal_indices = np.zeros(period)
    for i in range(period):
        # Extract all available values for this specific quarter (ignoring NaNs)
        quarter_vals = detrended[i::period]
        seasonal_indices[i] = np.nanmean(quarter_vals)
        
    # Normalizing seasonal indices so their average is exactly 1.0
    seasonal_indices = seasonal_indices / np.mean(seasonal_indices)
    
    # Mapping the 4 seasonal indices across the entire length of the dataset
    seasonal = np.array([seasonal_indices[i % period] for i in range(n)])
    
    # Calculating Residual Component (Multiplicative: Data / (Trend * Seasonal))
    residual = series_values / (trend * seasonal)
    
    return trend, seasonal, residual


def adf_test(series, alpha=0.05, verbose=True):
    """Augmented Dickey-Fuller test. Returns (statistic, p_value, is_stationary).

    `adfuller` was never imported, so every call to this raised NameError. It
    also only printed, which meant nothing downstream could act on the result.

    statsmodels supplies the Dickey-Fuller critical-value tables, which are
    simulated rather than derived - the one part of this test not worth writing
    out by hand. Imported here so the rest of the module loads without it.
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, result_object=True)
    stat, p_value = float(result.statistic), float(result.pvalue)
    is_stationary = p_value <= alpha
    if verbose:
        print(f"ADF Statistic: {stat}")
        print(f"p-value: {p_value}")
        verdict = "Stationary" if is_stationary else "Non-Stationary"
        sign = "<=" if is_stationary else ">"
        print(f"Result: {verdict} (since p-value is {sign} {alpha})\n")
    return stat, p_value, is_stationary


def feature_lag_matrix(time_series, p):
    X, y = [], []
    for i in range(len(time_series) - p):
        X.append(time_series[i : i + p])
        y.append(time_series[i + p])
    return np.array(X), np.array(y)
