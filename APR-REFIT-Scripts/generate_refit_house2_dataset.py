"""
Generates APR-REFIT-House2-dataset/ -- multi-day REFIT House 2 splits, same
column format as APR-new-House1-dataset/ (timestamp, aggregate, dishwasher,
fridge, microwave, washing_machine) but at 8s resolution.

REFIT House 2 appliance channels (processed REFITPowerData111215/House2.csv):
    Columns (no header): Unix, Aggregate, App1-App9
    aggregate        -> col 1
    fridge           -> col 2  (Appliance1 = Fridge-Freezer)
    washing_machine  -> col 3  (Appliance2 = Washing Machine)
    dishwasher       -> col 4  (Appliance3 = Dishwasher)
    microwave        -> col 6  (Appliance5 = Microwave; col5 is Television Site)

REFIT is sampled at 8-second intervals (vs UKDALE's 6-second).
MAX_FILL = 37  (~5 min = 37.5 x 8s steps, floored).

A 44-day chronologically contiguous window (30 train + 7 val + 7 test) is
selected automatically by scan_best_window(), which maximises the minimum
per-appliance activation count across dishwasher, microwave, and
washing_machine in the training portion (fridge cycles continuously so it is
not a selection factor).

Gap-filling policy: forward-fill gaps <= 5 min (37 x 8s steps), then zero-fill.
Appliance power is clipped to >= 0 W after filling.
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
REFIT   = os.path.join(BASE, "..", "REFITPowerData111215")
OUT_DIR = os.path.join(BASE, "..", "APR-REFIT-House2-dataset")
os.makedirs(OUT_DIR, exist_ok=True)

# REFIT House 2 column names after loading (see H2_COL_NAMES below)
# Appliance mapping from nilmtk building2.yaml:
#   Appliance1 (col2) = Fridge-Freezer  -> fridge
#   Appliance2 (col3) = Washing Machine -> washing_machine
#   Appliance3 (col4) = Dishwasher      -> dishwasher
#   Appliance4 (col5) = Television Site (skipped)
#   Appliance5 (col6) = Microwave       -> microwave
H2_COL_MAP = {
    "aggregate":       "agg",
    "fridge":          "app1",   # col2 -- Fridge-Freezer
    "washing_machine": "app2",   # col3 -- Washing Machine
    "dishwasher":      "app3",   # col4 -- Dishwasher
    "microwave":       "app5",   # col6 -- Microwave (app4=Television skipped)
}

N_TRAIN = 30
N_VAL   = 7
N_TEST  = 7

FREQ     = "8s"
MAX_FILL = 37    # ~5 min / 8s = 37.5 -> floor to 37 steps
THRESHOLD = 10.0  # W -- ON/OFF boundary used during scanning


def load_house(csv_path: str) -> pd.DataFrame:
    """
    Load a REFIT processed CSV (no header).
    Column layout: Unix(0), Aggregate(1), App1(2) ... App9(10).
    Returns a DataFrame indexed by UTC datetime with named columns.
    """
    df = pd.read_csv(
        csv_path, header=None,
        names=["unix", "agg", "app1", "app2", "app3", "app4",
               "app5", "app6", "app7", "app8", "app9"],
        dtype={"unix": np.int64},
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(df["unix"], unit="s", utc=True)
    df = (df.drop_duplicates(subset="datetime")
            .set_index("datetime")
            .sort_index())
    return df


def resample_to_grid(series: pd.Series, start: str, end: str) -> pd.Series:
    grid = pd.date_range(start=start, end=end, freq=FREQ, tz="UTC")
    reindexed = series.reindex(grid.union(series.index)).sort_index()
    reindexed = (
        reindexed
        .resample(FREQ)
        .mean()
        .reindex(grid)
        .ffill(limit=MAX_FILL)
        .fillna(0.0)
    )
    return reindexed.clip(lower=0.0)


def scan_best_window(raw_df: pd.DataFrame) -> dict:
    """
    Find the best (N_TRAIN + N_VAL + N_TEST)-day contiguous window by
    maximising the minimum per-appliance (dishwasher, microwave,
    washing_machine) activation count in the training portion.

    Returns a dict with keys: train_start, train_end, val_start, val_end,
    test_start, test_end (all 'YYYY-MM-DD' strings), and
    min_train_activation (int, worst-case activated sample count).
    """
    total_days = N_TRAIN + N_VAL + N_TEST   # 44

    # Per-day ON-counts (samples > THRESHOLD) for the three target appliances
    app_col = {
        "dishwasher":      "app3",
        "microwave":       "app5",
        "washing_machine": "app2",
    }
    daily = {}
    for app, col in app_col.items():
        s = raw_df[col].clip(lower=0.0)
        daily[app] = (s > THRESHOLD).resample("D").sum().rename(app)

    counts = pd.DataFrame(daily).fillna(0)
    all_days = counts.index.sort_values()
    n_days_total = len(all_days)

    if n_days_total < total_days:
        raise ValueError(
            f"REFIT House 2 has only {n_days_total} days of data; "
            f"need at least {total_days}."
        )

    best_score  = -1
    best_start  = 0

    for start_idx in range(n_days_total - total_days + 1):
        window_days = all_days[start_idx : start_idx + total_days]
        # Require strictly contiguous dates (no gap > 1 day)
        diffs = (window_days[1:] - window_days[:-1]).days
        if diffs.max() > 1:
            continue
        # Score = min training-portion activation count across the three apps
        train_days  = window_days[:N_TRAIN]
        train_mask  = counts.index.isin(train_days)
        score       = int(counts.loc[train_mask].sum().min())
        if score > best_score:
            best_score = score
            best_start = start_idx

    d = all_days
    i = best_start
    return {
        "train_start": d[i].strftime("%Y-%m-%d"),
        "train_end":   d[i + N_TRAIN - 1].strftime("%Y-%m-%d"),
        "val_start":   d[i + N_TRAIN].strftime("%Y-%m-%d"),
        "val_end":     d[i + N_TRAIN + N_VAL - 1].strftime("%Y-%m-%d"),
        "test_start":  d[i + N_TRAIN + N_VAL].strftime("%Y-%m-%d"),
        "test_end":    d[i + N_TRAIN + N_VAL + N_TEST - 1].strftime("%Y-%m-%d"),
        "min_train_activation": best_score,
    }


def generate_split(raw_df: pd.DataFrame, name: str,
                   start: str, end: str) -> None:
    start_ts = f"{start} 00:00:00"
    end_ts   = f"{end} 23:59:52"   # last 8s-aligned step of the day

    print(f"\n{'='*60}")
    print(f"Processing split: {name}  (House 2, {start} -> {end})")
    print(f"{'='*60}")

    cols = {}
    for label, col in H2_COL_MAP.items():
        print(f"  Resampling  {col:<6}  ->  {label}")
        cols[label] = resample_to_grid(raw_df[col], start_ts, end_ts)

    df = pd.DataFrame(cols)
    df.index.name = "timestamp"

    n_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    print(f"\n  Shape       : {df.shape}  ({n_days:.1f} days)")
    print(f"  Time range  : {df.index[0]}  ->  {df.index[-1]}")
    print(f"\n  Power stats (W):")
    print(df.describe().round(2).to_string())

    threshold = {"dishwasher": 10, "fridge": 10, "microwave": 10, "washing_machine": 10}
    print("\n  Appliance activation rate (% time > threshold):")
    for app, thr in threshold.items():
        pct = (df[app] > thr).mean() * 100
        print(f"    {app:<20s}: {pct:6.2f}%")

    out_path = os.path.join(OUT_DIR, f"REFIT_HF_{name}.csv")
    df.to_csv(out_path)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    csv_path = os.path.join(REFIT, "House2.csv")
    if not os.path.exists(csv_path):
        import sys
        print(f"Error: {csv_path} not found. "
              f"Ensure REFITPowerData111215/ is at the expected location.")
        sys.exit(1)

    print(f"Loading REFIT House 2: {csv_path}")
    raw = load_house(csv_path)
    print(f"  Total rows  : {len(raw):,}")
    print(f"  Date range  : {raw.index[0]}  to  {raw.index[-1]}")

    print("\nScanning for best 44-day window (30 train + 7 val + 7 test)...")
    w = scan_best_window(raw)
    print(f"\nSelected window:")
    print(f"  train      : {w['train_start']} to {w['train_end']}  ({N_TRAIN} days)")
    print(f"  validation : {w['val_start']}   to {w['val_end']}    ({N_VAL} days)")
    print(f"  test       : {w['test_start']}  to {w['test_end']}   ({N_TEST} days)")
    print(f"  min train activation (DW / MW / WM): {w['min_train_activation']:,} hits")

    splits = [
        {"name": "train",      "start": w["train_start"], "end": w["train_end"]},
        {"name": "validation", "start": w["val_start"],   "end": w["val_end"]},
        {"name": "test",       "start": w["test_start"],  "end": w["test_end"]},
    ]
    for split in splits:
        generate_split(raw, split["name"], split["start"], split["end"])

    print("\nAll splits complete.")
