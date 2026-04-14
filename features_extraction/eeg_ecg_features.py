from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, welch, find_peaks

# ============================================================
# ROOT FOLDER
# ============================================================
ROOT = Path("/Users/reedaww/recordings/")

# ============================================================
# FOLDERS TO PROCESS
# ============================================================
SUBFOLDERS = [
    "47_sebastien_arithmetic",
    "45_seb_color",
    "44_sebastien_right_answers_only",
    "40_sebastien_wrong_answers_only",
    "39_oliver_arithmetic_2 ",
    "34_oliver_arithmetic_take1",
    "29_oliver_colors",
    "28_oliver_right_answers_only",
    "26_oliver_wrong asnswers only",
]

# ============================================================
# FEATURE SETTINGS
# ============================================================
FEATURE_WINDOW_SEC = 2.0
FEATURE_STEP_SEC = 1.0

# ECG features are computed on a trailing 30 s window ending at each EEG window centre
ECG_FEATURE_WINDOW_SEC = 30.0

# ECG RR constraints
ECG_MIN_RR_SEC = 0.33   # ~180 bpm
ECG_MAX_RR_SEC = 1.50   # ~40 bpm

# ============================================================
# FILTER SETTINGS
# Match the acquisition/offline-processing pipeline
# ============================================================
EEG_HP_HZ = 0.5
EEG_LP_HZ = 35.0

ECG_HP_HZ = 0.5
ECG_LP_HZ = 40.0

NOTCH_1_HZ = 50.0
NOTCH_2_HZ = 100.0
NOTCH_Q = 35.0

DISPLAY_MOVAVG_N = 3

EEG_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
}
EEG_TOTAL_POWER_RANGE = (0.5, 30.0)

EPS = 1e-12

# ============================================================
# ECG R-PEAK DETECTION SETTINGS
# ============================================================
ECG_HRV_MIN_PEAK_DISTANCE_SEC = 0.30
ECG_HRV_QRS_BAND_HZ = (5.0, 15.0)
ECG_HRV_ENVELOPE_SEC = 0.08
ECG_HRV_PEAK_PROM_FACTOR = 0.50
ECG_HRV_REFINE_RADIUS_SEC = 0.08


# ============================================================
# HELPERS
# ============================================================
def estimate_fs_from_t(t: np.ndarray, fs_fallback: float = 500.0) -> float:
    if len(t) < 5:
        return fs_fallback
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return fs_fallback
    return float(1.0 / np.median(dt))


def moving_average_centered(x: np.ndarray, n: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x.copy()
    kernel = np.ones(int(n), dtype=float) / float(n)
    return np.convolve(x, kernel, mode="same")


def apply_zero_phase_sos(sos, x: np.ndarray) -> np.ndarray:
    try:
        return sosfiltfilt(sos, x)
    except Exception:
        return x.copy()


def apply_zero_phase_ba(b, a, x: np.ndarray) -> np.ndarray:
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x.copy()


def apply_offline_filter_chain(x: np.ndarray, fs: float, kind: str) -> np.ndarray:
    """
    Match the previous offline-processing chain:
      EEG: 0.5 Hz HP -> 50 Hz notch -> 100 Hz notch -> 35 Hz LP -> MA(3)
      ECG: 0.5 Hz HP -> 50 Hz notch -> 100 Hz notch -> 40 Hz LP -> MA(3)
    """
    x = np.asarray(x, dtype=float)

    if kind == "eeg":
        hp_hz = EEG_HP_HZ
        lp_hz = EEG_LP_HZ
    elif kind == "ecg":
        hp_hz = ECG_HP_HZ
        lp_hz = ECG_LP_HZ
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # High-pass
    sos_hp = butter(2, hp_hz / (fs / 2), btype="highpass", output="sos")
    y = apply_zero_phase_sos(sos_hp, x)

    # 50 Hz notch
    b50, a50 = iirnotch(NOTCH_1_HZ, NOTCH_Q, fs)
    y = apply_zero_phase_ba(b50, a50, y)

    # 100 Hz notch
    b100, a100 = iirnotch(NOTCH_2_HZ, NOTCH_Q, fs)
    y = apply_zero_phase_ba(b100, a100, y)

    # Low-pass
    sos_lp = butter(4, lp_hz / (fs / 2), btype="lowpass", output="sos")
    y = apply_zero_phase_sos(sos_lp, y)

    # Small moving average
    y = moving_average_centered(y, n=DISPLAY_MOVAVG_N)

    return y


def bandpower_welch(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < max(8, int(fs)):
        return np.nan

    nperseg = min(len(x), max(256, int(fs * 2)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)

    m = (f >= f_lo) & (f < f_hi)
    if not np.any(m):
        return np.nan

    return float(np.trapezoid(pxx[m], f[m]))


def compute_eeg_channel_window_features(x: np.ndarray, fs: float, prefix: str) -> dict:
    """
    Final EEG features for one channel on one 2 s window:
      - relative theta power
      - relative alpha power
      - theta/alpha ratio
    """
    x = np.asarray(x, dtype=float)

    out = {
        f"{prefix}_theta_rel": np.nan,
        f"{prefix}_alpha_rel": np.nan,
        f"{prefix}_theta_alpha_ratio": np.nan,
    }

    if len(x) < max(8, int(fs)):
        return out

    x = x - np.mean(x)

    theta_power = bandpower_welch(x, fs, *EEG_BANDS["theta"])
    alpha_power = bandpower_welch(x, fs, *EEG_BANDS["alpha"])
    total_power = bandpower_welch(x, fs, *EEG_TOTAL_POWER_RANGE)

    if np.isfinite(theta_power) and np.isfinite(total_power) and total_power > 0:
        out[f"{prefix}_theta_rel"] = float(theta_power / (total_power + EPS))

    if np.isfinite(alpha_power) and np.isfinite(total_power) and total_power > 0:
        out[f"{prefix}_alpha_rel"] = float(alpha_power / (total_power + EPS))

    if np.isfinite(theta_power) and np.isfinite(alpha_power) and alpha_power > 0:
        out[f"{prefix}_theta_alpha_ratio"] = float(theta_power / (alpha_power + EPS))

    return out


def deduplicate_close_peaks(peaks: np.ndarray, score: np.ndarray, min_distance: int) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) == 0:
        return peaks

    peaks = np.sort(peaks)
    kept = [int(peaks[0])]

    for p in peaks[1:]:
        p = int(p)
        if p - kept[-1] < min_distance:
            if score[p] > score[kept[-1]]:
                kept[-1] = p
        else:
            kept.append(p)

    return np.asarray(kept, dtype=int)


def detect_ecg_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect R-peaks from the offline-filtered ECG.
    Returns sample indices of detected R-peaks.
    """
    ecg = np.asarray(ecg, dtype=float)
    if len(ecg) < max(8, int(fs * 2)):
        return np.asarray([], dtype=int)

    x = ecg - np.mean(ecg)

    qrs_lo, qrs_hi = ECG_HRV_QRS_BAND_HZ
    qrs_hi = min(qrs_hi, 0.49 * fs)
    if qrs_lo >= qrs_hi:
        return np.asarray([], dtype=int)

    sos_qrs = butter(
        2,
        [qrs_lo / (fs / 2), qrs_hi / (fs / 2)],
        btype="bandpass",
        output="sos",
    )
    qrs_band = apply_zero_phase_sos(sos_qrs, x)

    env_n = max(1, int(round(ECG_HRV_ENVELOPE_SEC * fs)))
    qrs_env = moving_average_centered(np.abs(qrs_band), n=env_n)

    min_distance = max(1, int(round(ECG_HRV_MIN_PEAK_DISTANCE_SEC * fs)))
    prominence = float(ECG_HRV_PEAK_PROM_FACTOR * np.std(qrs_env))
    if not np.isfinite(prominence) or prominence <= 0:
        prominence = 0.0

    env_peaks, _ = find_peaks(
        qrs_env,
        distance=min_distance,
        prominence=prominence,
    )

    if len(env_peaks) == 0:
        return np.asarray([], dtype=int)

    refine_radius = max(1, int(round(ECG_HRV_REFINE_RADIUS_SEC * fs)))
    score = np.abs(x)
    refined = []

    for p in env_peaks:
        left = max(0, int(p) - refine_radius)
        right = min(len(x), int(p) + refine_radius + 1)
        if right <= left:
            continue
        refined_idx = left + int(np.argmax(score[left:right]))
        refined.append(refined_idx)

    refined = deduplicate_close_peaks(np.asarray(refined, dtype=int), score, min_distance)
    return refined


def get_valid_rr_intervals_in_window(
    peak_times_s: np.ndarray,
    t_end_s: float,
    window_sec: float,
) -> np.ndarray:
    """
    Return valid RR intervals in the trailing window [t_end_s - window_sec, t_end_s].
    """
    peak_times_s = np.asarray(peak_times_s, dtype=float)
    if len(peak_times_s) < 2:
        return np.asarray([], dtype=float)

    t_start_s = t_end_s - window_sec
    win_peaks = peak_times_s[(peak_times_s >= t_start_s) & (peak_times_s <= t_end_s)]

    if len(win_peaks) < 2:
        return np.asarray([], dtype=float)

    rr = np.diff(win_peaks)
    valid = (
        np.isfinite(rr)
        & (rr >= ECG_MIN_RR_SEC)
        & (rr <= ECG_MAX_RR_SEC)
    )
    return rr[valid]


def compute_ecg_hrv_features_30s(peak_times_s: np.ndarray, t_end_s: float) -> dict:
    """
    Compute the final ECG features from a trailing 30 s RR window:
      - mean heart rate
      - RMSSD
      - SDNN
    Units:
      - mean HR: bpm
      - RMSSD: s
      - SDNN: s
    """
    rr = get_valid_rr_intervals_in_window(peak_times_s, t_end_s, ECG_FEATURE_WINDOW_SEC)

    out = {
        "ecg_mean_hr_bpm_30s": np.nan,
        "ecg_rmssd_s_30s": np.nan,
        "ecg_sdnn_s_30s": np.nan,
    }

    if len(rr) >= 1:
        mean_rr = float(np.mean(rr))
        if np.isfinite(mean_rr) and mean_rr > 0:
            out["ecg_mean_hr_bpm_30s"] = float(60.0 / mean_rr)

    if len(rr) >= 2:
        drr = np.diff(rr)
        if len(drr) >= 1:
            out["ecg_rmssd_s_30s"] = float(np.sqrt(np.mean(drr ** 2)))

    if len(rr) >= 2:
        out["ecg_sdnn_s_30s"] = float(np.std(rr, ddof=1))

    return out


# ============================================================
# PROCESS ONE FOLDER
# ============================================================
def process_file(main_csv_path: Path):
    print(f"\nProcessing: {main_csv_path}")

    df = pd.read_csv(main_csv_path)

    required_cols = [
        "t_s",
        "abs_time_iso_ms",
        "raw_ch1_V_electrode",
        "raw_ch2_V_electrode",
        "raw_ch3_V_electrode",
        "raw_ch4_V_electrode",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("Missing columns:")
        for c in missing:
            print(f"  - {c}")
        return

    t = df["t_s"].to_numpy(dtype=float)
    fs_est = estimate_fs_from_t(t)
    print(f"Estimated sampling rate: {fs_est:.3f} Hz")

    # --------------------------------------------------------
    # Offline filtering from main_recording.csv raw electrode data
    # --------------------------------------------------------
    eeg1 = apply_offline_filter_chain(df["raw_ch1_V_electrode"].to_numpy(dtype=float), fs_est, "eeg")
    eeg2 = apply_offline_filter_chain(df["raw_ch2_V_electrode"].to_numpy(dtype=float), fs_est, "eeg")
    eeg3 = apply_offline_filter_chain(df["raw_ch3_V_electrode"].to_numpy(dtype=float), fs_est, "eeg")
    ecg = apply_offline_filter_chain(df["raw_ch4_V_electrode"].to_numpy(dtype=float), fs_est, "ecg")

    # --------------------------------------------------------
    # Detect ECG R-peaks globally once
    # --------------------------------------------------------
    r_peaks = detect_ecg_r_peaks(ecg, fs_est)
    r_peak_times_s = t[r_peaks] if len(r_peaks) > 0 else np.asarray([], dtype=float)
    print(f"Detected ECG R-peaks: {len(r_peaks)}")

    # --------------------------------------------------------
    # Sliding-window output rows
    # --------------------------------------------------------
    window_n = int(round(FEATURE_WINDOW_SEC * fs_est))
    step_n = int(round(FEATURE_STEP_SEC * fs_est))

    if window_n < 8:
        raise ValueError("FEATURE_WINDOW_SEC is too short.")
    if len(t) < window_n:
        print("Recording is shorter than one full feature window. No features saved.")
        return

    out_rows = []
    window_index = 0

    for start in range(0, len(t) - window_n + 1, step_n):
        end = start + window_n
        t_start = float(t[start])
        t_end = float(t[end - 1])
        t_center = float(0.5 * (t_start + t_end))

        row = {
            "window_index": window_index,
            "abs_time_iso_ms": df.loc[start, "abs_time_iso_ms"],
            "t_window_start_s": t_start,
            "t_window_end_s": t_end,
            "t_window_center_s": t_center,
        }

        # EEG channel-specific features
        eeg1_feats = compute_eeg_channel_window_features(eeg1[start:end], fs_est, "eeg1")
        eeg2_feats = compute_eeg_channel_window_features(eeg2[start:end], fs_est, "eeg2")
        eeg3_feats = compute_eeg_channel_window_features(eeg3[start:end], fs_est, "eeg3")

        row.update(eeg1_feats)
        row.update(eeg2_feats)
        row.update(eeg3_feats)

        # EEG combined features per row = median across channels
        row["eeg_theta_rel"] = float(np.nanmedian([
            eeg1_feats["eeg1_theta_rel"],
            eeg2_feats["eeg2_theta_rel"],
            eeg3_feats["eeg3_theta_rel"],
        ]))

        row["eeg_alpha_rel"] = float(np.nanmedian([
            eeg1_feats["eeg1_alpha_rel"],
            eeg2_feats["eeg2_alpha_rel"],
            eeg3_feats["eeg3_alpha_rel"],
        ]))

        row["eeg_theta_alpha_ratio"] = float(np.nanmedian([
            eeg1_feats["eeg1_theta_alpha_ratio"],
            eeg2_feats["eeg2_theta_alpha_ratio"],
            eeg3_feats["eeg3_theta_alpha_ratio"],
        ]))

        # ECG final selected features from a trailing 30 s RR window
        row.update(compute_ecg_hrv_features_30s(r_peak_times_s, t_center))

        out_rows.append(row)
        window_index += 1

    out = pd.DataFrame(out_rows)

    # Save in same folder
    output_path = main_csv_path.parent / "eeg_ecg_features.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    for folder_name in SUBFOLDERS:
        main_csv_path = ROOT / folder_name / "main_recording.csv"

        if not main_csv_path.exists():
            print(f"\nFile not found: {main_csv_path}")
            continue

        try:
            process_file(main_csv_path)
        except Exception as e:
            print(f"\nError processing {main_csv_path}: {e}")