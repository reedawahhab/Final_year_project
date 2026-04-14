import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

# =========================================================
# USER SETTINGS
# =========================================================
HOWLAND_INPUT_CSV = Path("/Users/reedaww/Desktop/final experiments/47_Seb Arithmetic/spike_data 6(in).csv")

# If None, save next to input file
HOWLAND_OUTPUT_CSV = None

# Recommended starting values for GSR
HOWLAND_WINDOW_SEC = 10.0
HOWLAND_STEP_SEC   = 0.5

# Peak detection settings
PEAK_MIN_DISTANCE_SEC = 1.5
PEAK_PROM_FACTOR = 0.5
MIN_PEAK_PROM = 1e-9


# =========================================================
# HELPERS
# =========================================================
def estimate_fs_from_t(t_s: np.ndarray) -> float:
    t_s = np.asarray(t_s, dtype=float)
    dt = np.diff(t_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Could not estimate sampling frequency from t_s.")
    return float(1.0 / np.median(dt))


def compute_positive_auc(t_s: np.ndarray, y: np.ndarray) -> float:
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(t_s) & np.isfinite(y)
    t_s = t_s[mask]
    y = y[mask]

    if len(t_s) < 2:
        return np.nan

    y_pos = np.maximum(y, 0.0)
    return float(np.trapezoid(y_pos, t_s))


def compute_peak_count(
    y: np.ndarray,
    fs: float,
    min_distance_sec: float = PEAK_MIN_DISTANCE_SEC,
    prom_factor: float = PEAK_PROM_FACTOR,
    min_prom: float = MIN_PEAK_PROM,
) -> int:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) < 3:
        return 0

    prom = max(float(prom_factor * np.std(y)), float(min_prom))
    distance = max(1, int(round(min_distance_sec * fs)))

    peaks, _ = find_peaks(y, distance=distance, prominence=prom)
    return int(len(peaks))


# =========================================================
# FEATURE FUNCTION
# =========================================================
def compute_howland_window_features(win_df: pd.DataFrame, fs: float) -> dict:
    """
    Required columns:
      - t_s
      - base_lp
      - spike_lp
    """
    t_s = win_df["t_s"].to_numpy(dtype=float)
    base_lp = win_df["base_lp"].to_numpy(dtype=float)
    spike_lp = win_df["spike_lp"].to_numpy(dtype=float)

    base_mean = float(np.nanmean(base_lp)) if len(base_lp) else np.nan
    spike_auc = compute_positive_auc(t_s, spike_lp)

    spike_peak_count = compute_peak_count(
        y=spike_lp,
        fs=fs,
        min_distance_sec=PEAK_MIN_DISTANCE_SEC,
        prom_factor=PEAK_PROM_FACTOR,
        min_prom=MIN_PEAK_PROM,
    )

    return {
        "base_mean": base_mean,
        "spike_auc": spike_auc,
        "spike_peak_count": spike_peak_count,
    }


# =========================================================
# WINDOWED PROCESSOR
# =========================================================
def process_howland_features(
    input_csv: Path,
    output_csv: Path,
    window_sec: float,
    step_sec: float,
):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = ["t_s", "base_lp", "spike_lp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv.name}: {missing}")

    df["t_s"] = pd.to_numeric(df["t_s"], errors="coerce")
    df["base_lp"] = pd.to_numeric(df["base_lp"], errors="coerce")
    df["spike_lp"] = pd.to_numeric(df["spike_lp"], errors="coerce")

    df = df[np.isfinite(df["t_s"])].copy()
    df = df.sort_values("t_s").reset_index(drop=True)

    fs = estimate_fs_from_t(df["t_s"].to_numpy())
    window_n = int(round(window_sec * fs))
    step_n = int(round(step_sec * fs))

    if window_n < 2:
        raise ValueError(f"Window too short for {input_csv.name}")
    if len(df) < window_n:
        raise ValueError(
            f"{input_csv.name} is shorter than one full window. "
            f"Need at least {window_n} samples, found {len(df)}."
        )

    rows = []
    window_index = 0

    for start in range(0, len(df) - window_n + 1, step_n):
        end = start + window_n
        win_df = df.iloc[start:end]

        # Keep original columns from the window start row
        row = df.iloc[start].to_dict()

        # Add window metadata
        row["window_index"] = window_index
        row["window_start_s"] = float(win_df["t_s"].iloc[0])
        row["window_end_s"] = float(win_df["t_s"].iloc[-1])
        row["window_center_s"] = float(0.5 * (win_df["t_s"].iloc[0] + win_df["t_s"].iloc[-1]))
        row["window_duration_s"] = float(win_df["t_s"].iloc[-1] - win_df["t_s"].iloc[0])
        row["samples_in_window"] = int(len(win_df))
        row["estimated_fs_hz"] = float(fs)

        # Add computed features
        row.update(compute_howland_window_features(win_df, fs))

        rows.append(row)
        window_index += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(f"Estimated fs: {fs:.3f} Hz")
    print(f"Window: {window_sec:.3f} s -> {window_n} samples")
    print(f"Step:   {step_sec:.3f} s -> {step_n} samples")
    print(f"Rows:   {len(out_df)}\n")


# =========================================================
# MAIN
# =========================================================
def main():
    howland_output = HOWLAND_OUTPUT_CSV
    if howland_output is None:
        howland_output = HOWLAND_INPUT_CSV.with_name("howland_features.csv")

    process_howland_features(
        input_csv=HOWLAND_INPUT_CSV,
        output_csv=howland_output,
        window_sec=HOWLAND_WINDOW_SEC,
        step_sec=HOWLAND_STEP_SEC,
    )


if __name__ == "__main__":
    main()