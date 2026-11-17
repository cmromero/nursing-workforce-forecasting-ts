"""Create a purely synthetic monthly nursing PPE series (not real hospital data).

Run from the repository root: python scripts/generate_synthetic_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/nursing_workforce_synthetic.csv"
rng = np.random.default_rng(82)
dates = pd.date_range("2016-01-31", "2026-10-31", freq="ME")
t = np.arange(len(dates))
# Arbitrary patterns for an educational, reproducible example — not real observations.
trend = 1090 + 3.55 * t
seasonal_pattern = 14 * np.sin(2 * np.pi * (t % 12) / 12) + 8 * np.cos(2 * np.pi * (t % 12) / 12)
noise = rng.normal(0, 11, len(t))
for i in range(1, len(noise)):
    noise[i] += 0.42 * noise[i - 1]
temporary_change = np.where((dates >= "2020-04-01") & (dates <= "2020-07-31"), -22, 0)
level_change = np.where(dates >= "2022-01-31", 20, 0)
ppe = np.round(trend + seasonal_pattern + noise + temporary_change + level_change, 2)
frame = pd.DataFrame({"DATANOM": dates.strftime("%Y-%m-%d"), "PPE": ppe})
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(OUTPUT, index=False)
print(f"Generated {len(frame)} synthetic monthly PPE observations in {OUTPUT}")
