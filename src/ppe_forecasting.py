"""Time-series tools for an educational nursing-equivalent-workforce forecast.

All examples use only the historical monthly PPE series. The model is SARIMA,
implemented through statsmodels' SARIMAX class without external predictors.
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATE_COL = "DATANOM"
TARGET_COL = "PPE"
TEST_MONTHS = 10
CANDIDATES = [
    ((1, 1, 0), (1, 0, 1, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 1), (1, 0, 0, 12)),
]


def prepare_monthly_data(path: str | Path) -> pd.DataFrame:
    """Accept only dates and PPE, rejecting missing or duplicated months."""
    data = pd.read_csv(path)
    if not {DATE_COL, TARGET_COL}.issubset(data.columns):
        raise ValueError("Source CSV must contain DATANOM and PPE columns")
    # Even if a source CSV has other columns, never use them for modeling.
    data = data[[DATE_COL, TARGET_COL]].copy()
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="raise") + pd.offsets.MonthEnd(0)
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="raise")
    if data[DATE_COL].duplicated().any():
        raise ValueError("Duplicate monthly dates")
    if data[TARGET_COL].isna().any() or not np.isfinite(data[TARGET_COL]).all() or (data[TARGET_COL] < 0).any():
        raise ValueError("Monthly PPE must be non-negative and finite")
    prepared = data.sort_values(DATE_COL).set_index(DATE_COL)
    expected = pd.date_range(prepared.index.min(), prepared.index.max(), freq="ME")
    missing = expected.difference(prepared.index)
    if len(missing):
        raise ValueError(f"Missing monthly observations: {missing.strftime('%Y-%m').tolist()}")
    return prepared.asfreq("ME")


def load_monthly_series(path: str | Path) -> pd.Series:
    return prepare_monthly_data(path)[TARGET_COL]


def chronological_split(y: pd.Series, test_months: int = TEST_MONTHS):
    if len(y) < 36 + test_months:
        raise ValueError("Require 36 or more training months plus the test period")
    return y.iloc[:-test_months], y.iloc[-test_months:]


def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    """Assume every future month equals the last observed month."""
    return np.repeat(float(train.iloc[-1]), steps)


def seasonal_naive_forecast(train: pd.Series, steps: int, period: int = 12) -> np.ndarray:
    """Repeat the last observed year's month-by-month pattern."""
    if len(train) < period:
        raise ValueError("Insufficient history for seasonal naive forecast")
    return np.resize(train.iloc[-period:].to_numpy(dtype=float), steps)


def moving_average_forecast(train: pd.Series, steps: int, window: int = 3) -> np.ndarray:
    """Keep the last three observed months' mean constant over the horizon.

    No held-out actual values enter later multi-step predictions.
    """
    if len(train) < window:
        raise ValueError("Insufficient history for moving average forecast")
    return np.repeat(float(train.iloc[-window:].mean()), steps)


def score_forecast(actual, predicted) -> dict:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Actual and forecasts must have the same finite values shape")
    errors = a - p
    nonzero = a != 0
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE_pct": float(np.mean(np.abs(errors[nonzero] / a[nonzero])) * 100) if nonzero.any() else np.nan,
    }


def fit_sarima(y: pd.Series, order: tuple, seasonal_order: tuple):
    """Fit a univariate SARIMA; only the historical PPE values are supplied."""
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=150)


def select_by_train_aic(train: pd.Series, candidates=CANDIDATES):
    """Choose SARIMA orders on training AIC alone, before test evaluation."""
    comparison, valid = [], []
    for order, seasonal_order in candidates:
        try:
            result = fit_sarima(train, order, seasonal_order)
            converged = bool(result.mle_retvals.get("converged", True))
            usable = converged and np.isfinite(result.aic)
            comparison.append({"order": str(order), "seasonal_order": str(seasonal_order),
                               "AIC_train": round(float(result.aic), 3) if usable else np.nan,
                               "converged": converged})
            if usable:
                valid.append((result.aic, order, seasonal_order, result))
        except Exception as exc:
            comparison.append({"order": str(order), "seasonal_order": str(seasonal_order),
                               "AIC_train": np.nan, "converged": False, "note": str(exc)[:120]})
    if not valid:
        raise RuntimeError("No candidate SARIMA model converged")
    best = min(valid, key=lambda item: item[0])
    return best[1], best[2], best[3], pd.DataFrame(comparison)


def build_prediction_frame(model, last_observed: pd.Timestamp, steps: int) -> pd.DataFrame:
    if steps <= 0:
        raise ValueError("Forecast horizon must be positive")
    future = model.get_forecast(steps=steps)
    interval = future.conf_int(alpha=0.05)
    dates = pd.date_range(pd.Timestamp(last_observed) + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
    return pd.DataFrame({
        "month": dates,
        "ppe_forecast": np.asarray(future.predicted_mean, dtype=float),
        "lower_95": np.asarray(interval.iloc[:, 0], dtype=float),
        "upper_95": np.asarray(interval.iloc[:, 1], dtype=float),
    })
