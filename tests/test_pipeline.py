"""Basic tests: run `python -m pytest -q` from the repository root."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.ppe_forecasting import (
    prepare_monthly_data, load_monthly_series, chronological_split,
    naive_forecast, seasonal_naive_forecast, moving_average_forecast,
    score_forecast,
)

RAW = Path(__file__).resolve().parents[1] / "data/raw/nursing_workforce_synthetic.csv"


def test_dataset_and_temporal_split(tmp_path):
    prepared = prepare_monthly_data(RAW)
    assert list(prepared.columns) == ["PPE"]
    assert len(prepared) == 130
    assert prepared.index[0] == pd.Timestamp("2016-01-31")
    assert prepared.index[-1] == pd.Timestamp("2026-10-31")
    exported = tmp_path / "processed.csv"
    prepared.to_csv(exported, index_label="DATANOM")
    series = load_monthly_series(exported)
    assert series.equals(prepared.PPE)
    train, test = chronological_split(series)
    assert len(test) == 10 and train.index.max() < test.index.min()
    assert len(naive_forecast(train, 10)) == 10
    assert len(seasonal_naive_forecast(train, 10)) == 10
    assert len(moving_average_forecast(train, 10)) == 10


def test_metric_calculation():
    result = score_forecast([10, 15, 20], [10, 16, 18])
    assert result["MAE"] == pytest.approx(1)
    assert result["RMSE"] == pytest.approx(np.sqrt(5 / 3))


def test_missing_month_rejected(tmp_path):
    data = pd.read_csv(RAW).drop(index=10)
    path = tmp_path / "missing_month.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing monthly"):
        prepare_monthly_data(path)


def test_unrelated_columns_never_used(tmp_path):
    data = pd.read_csv(RAW)
    data["unrelated"] = np.arange(len(data))
    path = tmp_path / "extra_columns.csv"
    data.to_csv(path, index=False)
    assert list(prepare_monthly_data(path).columns) == ["PPE"]
