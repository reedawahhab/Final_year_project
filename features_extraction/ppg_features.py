
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# ============================================================
# USER SETTINGS
# ============================================================
INPUT_CSV = Path("/Users/reedaww/Desktop/final experiments/47_Seb Arithmetic/.csv")   # <-- change this

AUTO_ESTIMATE_FS = True
FS_FIXED = 100.0

# -------------------------------
# Filtering
# -------------------------------
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 8.0
FILTER_ORDER = 3

# -------------------------------
# Peak detection
# -------------------------------
MIN_PEAK_DISTANCE_SEC = 0.40
MIN_PEAK_WIDTH_SEC = 0.05
MAX_PEAK_WIDTH_SEC = 0.60

PEAK_PROMINENCE = None
PROMINENCE_STD_FACTOR = 0.35

# -------------------------------
# Physiology / feature settings
# -------------------------------
MIN_IBI_SEC = 0.33
MAX_IBI_SEC = 1.50

REJECT_CLIPPED_PEAKS = True
CLIP_MARGIN_COUNTS = 2.0

FOOT_SEARCH_MAX_SEC = 1.20
FEATURE_WINDOW_SEC = 30.0

# ============================================================
# LOAD CSV
# ============================================================
def load_ppg_csv(csv_path):
    t_s = []
    abs_hms = []
    abs_iso = []
    raw_adc = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"t_s", "abs_time_hms_ms", "abs_time_iso_ms", "raw_adc"}

        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}.\n"
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            try:
                t_s.append(float(row["t_s"]))
                abs_hms.append(row["abs_time_hms_ms"])
                abs_iso.append(row["abs_time_iso_ms"])
                raw_adc.append(float(row["raw_adc"]))
            except Exception:
                continue

    if len(t_s) < 10:
        raise ValueError("Not enough valid samples in CSV.")

    return (
        np.asarray(t_s, dtype=float),
        np.asarray(abs_hms, dtype=object),
        np.asarray(abs_iso, dtype=object),
        np.asarray(raw_adc, dtype=float),
    )

# ============================================================
# HELPERS
# ============================================================
def estimate_fs_from_t(t_s):
    dt = np.diff(t_s)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Could not estimate fs from t_s.")
    return float(1.0 / np.median(dt))

def butter_bandpass_filter(x, fs, low_hz, high_hz, order=3):
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq

    if high >= 1.0:
        high = 0.999
    if low <= 0.0:
        low = 1e-6
    if low >= high:
        raise ValueError("Invalid bandpass cutoff frequencies.")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))

# ============================================================
# PEAK DETECTION
# ============================================================
def detect_ppg_peaks(filtered_ppg, raw_adc, fs):
    min_distance = max(1, int(MIN_PEAK_DISTANCE_SEC * fs))
    min_width = max(1, int(MIN_PEAK_WIDTH_SEC * fs))
    max_width = max(min_width + 1, int(MAX_PEAK_WIDTH_SEC * fs))

    if PEAK_PROMINENCE is None:
        prominence = PROMINENCE_STD_FACTOR * np.std(filtered_ppg)
    else:
        prominence = float(PEAK_PROMINENCE)

    peaks, _ = find_peaks(
        filtered_ppg,
        distance=min_distance,
        prominence=prominence,
        width=(min_width, max_width),
    )

    if REJECT_CLIPPED_PEAKS and len(peaks) > 0:
        raw_max = np.max(raw_adc)
        keep = raw_adc[peaks] < (raw_max - CLIP_MARGIN_COUNTS)
        peaks = peaks[keep]

    return peaks, prominence

# ============================================================
# FOOT DETECTION
# ============================================================
def detect_feet(filtered_ppg, peaks, fs):
    feet = []
    max_back = max(1, int(FOOT_SEARCH_MAX_SEC * fs))

    for i, peak_idx in enumerate(peaks):
        if i == 0:
            left = max(0, peak_idx - max_back)
        else:
            left = max(peaks[i - 1] + 1, peak_idx - max_back)

        right = peak_idx + 1
        if right <= left:
            feet.append(left)
            continue

        seg = filtered_ppg[left:right]
        foot_idx = left + int(np.argmin(seg))
        feet.append(foot_idx)

    return np.asarray(feet, dtype=int)

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_features(t_s, abs_hms, abs_iso, filtered_ppg, peaks, feet):
    rows = []

    peak_times = t_s[peaks]
    peak_vals = filtered_ppg[peaks]
    foot_vals = filtered_ppg[feet]
    pwa_all = peak_vals - foot_vals

    for i in range(len(peaks)):
        peak_idx = peaks[i]
        peak_t = peak_times[i]

        # Rolling window ending at current peak
        win_start_t = peak_t - FEATURE_WINDOW_SEC
        start_i = np.searchsorted(peak_times, win_start_t, side="left")

        win_peak_times = peak_times[start_i:i + 1]
        win_pwa = pwa_all[start_i:i + 1]

        ibi_window = np.diff(win_peak_times)
        valid_ibi = ibi_window[(ibi_window >= MIN_IBI_SEC) & (ibi_window <= MAX_IBI_SEC)]

        hr_mean_bpm_window = np.nan
        if len(valid_ibi) >= 1:
            hr_mean_bpm_window = float(np.mean(60.0 / valid_ibi))

        rmssd_ms_window = np.nan
        if len(valid_ibi) >= 2:
            dibi = np.diff(valid_ibi)
            rmssd_s = np.sqrt(np.mean(dibi ** 2))
            rmssd_ms_window = float(rmssd_s * 1000.0)

        valid_pwa = win_pwa[np.isfinite(win_pwa) & (win_pwa > 0)]
        pwa_mean_window = safe_mean(valid_pwa)

        rows.append({
            "t_s": f"{t_s[peak_idx]:.6f}",
            "abs_time_hms_ms": abs_hms[peak_idx],
            "abs_time_iso_ms": abs_iso[peak_idx],
            "hr_mean_bpm_window": "" if not np.isfinite(hr_mean_bpm_window) else f"{hr_mean_bpm_window:.6f}",
            "rmssd_ms_window": "" if not np.isfinite(rmssd_ms_window) else f"{rmssd_ms_window:.6f}",
            "pwa_mean_window": "" if not np.isfinite(pwa_mean_window) else f"{pwa_mean_window:.6f}",
        })

    return rows

# ============================================================
# SAVE CSV
# ============================================================
def save_feature_csv(output_csv, rows):
    if len(rows) == 0:
        raise ValueError("No feature rows to save.")

    fieldnames = [
        "t_s",
        "abs_time_hms_ms",
        "abs_time_iso_ms",
        "hr_mean_bpm_window",
        "rmssd_ms_window",
        "pwa_mean_window",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# MAIN
# ============================================================
def main():
    input_csv = Path(INPUT_CSV)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = input_csv.with_name(f"{input_csv.stem}_features.csv")

    t_s, abs_hms, abs_iso, raw_adc = load_ppg_csv(input_csv)
    fs = estimate_fs_from_t(t_s) if AUTO_ESTIMATE_FS else FS_FIXED

    filtered_ppg = butter_bandpass_filter(
        raw_adc,
        fs=fs,
        low_hz=BANDPASS_LOW_HZ,
        high_hz=BANDPASS_HIGH_HZ,
        order=FILTER_ORDER,
    )

    peaks, used_prominence = detect_ppg_peaks(filtered_ppg, raw_adc, fs)

    if len(peaks) < 2:
        raise ValueError(
            f"Only {len(peaks)} peaks detected. Try lowering prominence "
            f"or adjusting filter settings."
        )

    feet = detect_feet(filtered_ppg, peaks, fs)
    rows = compute_features(t_s, abs_hms, abs_iso, filtered_ppg, peaks, feet)
    save_feature_csv(output_csv, rows)

    print(f"Loaded file: {input_csv}")
    print(f"Samples: {len(raw_adc)}")
    print(f"Estimated fs: {fs:.3f} Hz")
    print(f"Peaks detected: {len(peaks)}")
    print(f"Prominence used: {used_prominence:.6f}")
    print(f"Saved features to: {output_csv}")

if __name__ == "__main__":
    main()

