import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_col(df: pd.DataFrame, must_contain: list[str]) -> str:
    for c in df.columns:
        lc = c.lower()
        if all(s.lower() in lc for s in must_contain):
            return c
    raise ValueError(f"Could not find column containing: {must_contain}")


def fit_line(x: np.ndarray, y: np.ndarray):
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def bootstrap_single_prediction(x, y, x_new, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    preds = np.zeros(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        m, b = fit_line(xb, yb)
        preds[i] = m * x_new + b

    mean = preds.mean()
    low = np.percentile(preds, 2.5)
    high = np.percentile(preds, 97.5)

    return mean, low, high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--level", default="All")
    ap.add_argument("--athlete_cmj", type=float, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    jump_col = find_col(df, ["jump_height", "mean_cmj"])

    # Prefer bat_speed_mph, otherwise use hitting_max_hss
    bat_col = None
    if "bat_speed_mph" in df.columns:
        df["bat_speed_mph"] = pd.to_numeric(df["bat_speed_mph"], errors="coerce")
        if df["bat_speed_mph"].notna().sum() > 10:
            bat_col = "bat_speed_mph"

    if bat_col is None:
        if "hitting_max_hss" not in df.columns:
            raise ValueError("No usable bat speed column found.")
        bat_col = "hitting_max_hss"

    if args.level.lower() != "all" and "playing_level" in df.columns:
        df = df[df["playing_level"].str.lower() == args.level.lower()]

    df[jump_col] = pd.to_numeric(df[jump_col], errors="coerce")
    df[bat_col] = pd.to_numeric(df[bat_col], errors="coerce")

    df = df.dropna(subset=[jump_col, bat_col])

    if len(df) < 20:
        raise ValueError("Not enough usable rows.")

    x = df[jump_col].to_numpy()
    y = df[bat_col].to_numpy()

    # Fit model on ALL usable data
    m, b = fit_line(x, y)

    # Predict for new athlete
    pred_mean, ci_low, ci_high = bootstrap_single_prediction(
        x, y, args.athlete_cmj
    )

    r = np.corrcoef(x, y)[0, 1]

    print("\n=== CMJ → Bat Speed Projection ===")
    print(f"Model: bat_speed = {m:.4f} * cmj + {b:.4f}")
    print(f"Dataset r: {r:.3f}")
    print(f"\nAthlete CMJ: {args.athlete_cmj:.2f} cm")
    print(f"Predicted Bat Speed: {pred_mean:.2f} mph")
    print(f"95% CI: {ci_low:.2f} – {ci_high:.2f} mph\n")


if __name__ == "__main__":
    main()