import asyncio
import struct
import threading
import time
import csv
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
import queue

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker

from bleak import BleakScanner, BleakClient
from scipy.signal import (
    butter,
    iirnotch,
    sosfilt,
    sosfilt_zi,
    tf2sos,
    welch,
)

# =========================================================
# USER SETTINGS
# =========================================================
SESSION_TAG = "eeg_ecg_personal_test"

# CH1-CH3 = EEG, CH4 = ECG
CHANNEL_NAMES = ["EEG1", "EEG2", "EEG3", "ECG"]
CHANNEL_TYPES = ["eeg", "eeg", "eeg", "ecg"]

# Sliding-window feature settings
FEATURE_WINDOW_SEC = 2.0
FEATURE_STEP_SEC = 1.0

# =========================================================
# BLE
# =========================================================
TARGET_NAME = "EMG_Sensor"
CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Firmware packet = 2 samples/notify (40 bytes)
BYTES_PER_NOTIFY = 40
UNPACK_FMT = "<IiiiiIiiii"  # t,ch1,ch2,ch3,ch4, t,ch1,ch2,ch3,ch4

# =========================================================
# ADC / SCALING
# =========================================================
FS_NOMINAL = 500.0
ADC_BITS = 24
ADC_COUNTS_FS = 2 ** 23
ADC_VREF = 2.442
ADC_GAIN = 1.0
AFE_GAIN = 100.0

# =========================================================
# LIVE DISPLAY FILTERS
# =========================================================
EEG_HP_HZ = 0.5
EEG_LP_HZ = 35.0

ECG_HP_HZ = 0.5
ECG_LP_HZ = 40.0

NOTCH_Q = 35.0
NOTCH_1_HZ = 50.0
NOTCH_2_HZ = 100.0

DISPLAY_MOVAVG_N = 3  # visual only; 0/1 disables

# =========================================================
# OFFLINE POST-PROCESSING
# =========================================================
MNE_POST_ENABLED = True

EEG_POST_L_FREQ = 0.5
EEG_POST_H_FREQ = 35.0

ECG_POST_L_FREQ = 0.5
ECG_POST_H_FREQ = 40.0

EEG_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

EEG_TOTAL_POWER_RANGE = (0.5, 30.0)
ECG_TOTAL_POWER_RANGE = (0.5, 40.0)

EPS = 1e-12

# =========================================================
# PLOT WINDOW
# =========================================================
PLOT_WINDOW_S = 8.0
MAX_POINTS = int(PLOT_WINDOW_S * FS_NOMINAL) + 3000
X_TICK_STEP_S = 0.5

FIXED_Y = True
EEG_YLIM_UV = (-500, 500.0)
ECG_YLIM_MV = (-0.1, 0.1)

# =========================================================
# THREADING / QUEUES
# =========================================================
rxq = queue.Queue(maxsize=50000)
stop_event = threading.Event()

# =========================================================
# BUFFERS
# =========================================================
lock = threading.Lock()

t_buf = deque(maxlen=MAX_POINTS)

# EEG in uV, ECG in mV for display
disp_bufs = [deque(maxlen=MAX_POINTS) for _ in range(4)]

# full recording rows
rec_rows = []

t0_us = None
t0_wall_dt = None

_last_rate_print = time.time()
_samp_count = 0

# =========================================================
# TIMESTAMP HELPERS
# =========================================================
def fmt_hms_ms(dt: datetime) -> str:
    return dt.strftime("%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"

def fmt_iso_ms(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

def session_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# =========================================================
# CONVERSION HELPERS
# =========================================================
def counts_to_postafe_volts(counts: float) -> float:
    return (counts * ADC_VREF) / (ADC_GAIN * ADC_COUNTS_FS)

def postafe_to_electrode_volts(v_postafe: float) -> float:
    return v_postafe / AFE_GAIN

# =========================================================
# FILTER HELPERS
# =========================================================
class LiveSosChain:
    def __init__(self, sos_list, movavg_n=1):
        self.sos_list = sos_list
        self.zis = [sosfilt_zi(sos) for sos in sos_list]
        self.primed = False
        self.movavg_n = max(1, int(movavg_n))
        self.ma = deque(maxlen=self.movavg_n)

    def prime(self, x0):
        if self.primed:
            return
        for i, sos in enumerate(self.sos_list):
            self.zis[i] *= x0
        self.ma.clear()
        self.ma.append(float(x0))
        self.primed = True

    def step(self, x):
        y = float(x)
        for i, sos in enumerate(self.sos_list):
            out, self.zis[i] = sosfilt(sos, [y], zi=self.zis[i])
            y = float(out[0])

        if self.movavg_n >= 2:
            self.ma.append(y)
            y = float(sum(self.ma) / len(self.ma))
        return y

def make_live_chain(kind: str) -> LiveSosChain:
    if kind == "eeg":
        sos_hp = butter(2, EEG_HP_HZ / (FS_NOMINAL / 2), btype="highpass", output="sos")
        b50, a50 = iirnotch(NOTCH_1_HZ, NOTCH_Q, FS_NOMINAL)
        sos_n50 = tf2sos(b50, a50)
        b100, a100 = iirnotch(NOTCH_2_HZ, NOTCH_Q, FS_NOMINAL)
        sos_n100 = tf2sos(b100, a100)
        sos_lp = butter(4, EEG_LP_HZ / (FS_NOMINAL / 2), btype="lowpass", output="sos")
        return LiveSosChain([sos_hp, sos_n50, sos_n100, sos_lp], movavg_n=DISPLAY_MOVAVG_N)

    elif kind == "ecg":
        sos_hp = butter(2, ECG_HP_HZ / (FS_NOMINAL / 2), btype="highpass", output="sos")
        b50, a50 = iirnotch(NOTCH_1_HZ, NOTCH_Q, FS_NOMINAL)
        sos_n50 = tf2sos(b50, a50)
        b100, a100 = iirnotch(NOTCH_2_HZ, NOTCH_Q, FS_NOMINAL)
        sos_n100 = tf2sos(b100, a100)
        sos_lp = butter(4, ECG_LP_HZ / (FS_NOMINAL / 2), btype="lowpass", output="sos")
        return LiveSosChain([sos_hp, sos_n50, sos_n100, sos_lp], movavg_n=DISPLAY_MOVAVG_N)

    else:
        raise ValueError(f"Unknown kind: {kind}")

live_chains = [make_live_chain(t) for t in CHANNEL_TYPES]

def butter_sos_bandpass(fs, lo, hi, order=4):
    return butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass", output="sos")

def butter_sos_lowpass(fs, hi, order=4):
    return butter(order, hi / (fs / 2), btype="lowpass", output="sos")

def butter_sos_highpass(fs, lo, order=2):
    return butter(order, lo / (fs / 2), btype="highpass", output="sos")

def apply_zero_phase_sos(sos, x):
    try:
        from scipy.signal import sosfiltfilt
        return sosfiltfilt(sos, x)
    except Exception:
        return sosfilt(sos, x)

# =========================================================
# FEATURE HELPERS
# =========================================================
def estimate_fs_from_t(t):
    if len(t) < 5:
        return FS_NOMINAL
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return FS_NOMINAL
    return float(1.0 / np.median(dt))

def bandpower_welch(x, fs, f_lo, f_hi):
    x = np.asarray(x, dtype=float)
    if len(x) < max(8, int(fs)):
        return np.nan

    nperseg = min(len(x), max(256, int(fs * 2)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)

    m = (f >= f_lo) & (f < f_hi)
    if not np.any(m):
        return np.nan

    return float(np.trapezoid(pxx[m], f[m]))

def peak_frequency_welch(x, fs, f_lo, f_hi):
    x = np.asarray(x, dtype=float)
    if len(x) < max(8, int(fs)):
        return np.nan

    nperseg = min(len(x), max(256, int(fs * 2)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)

    m = (f >= f_lo) & (f <= f_hi)
    if not np.any(m):
        return np.nan

    f_sel = f[m]
    p_sel = pxx[m]
    if len(p_sel) == 0:
        return np.nan

    return float(f_sel[np.argmax(p_sel)])

def compute_eeg_window_features(x, fs, prefix):
    x = np.asarray(x, dtype=float)
    if len(x) < max(8, int(fs)):
        return {
            f"{prefix}_power_beta": np.nan,
            f"{prefix}_alpha_beta": np.nan,
            f"{prefix}_theta_beta": np.nan,
            f"{prefix}_peak_frequency_hz": np.nan,
        }

    x = x - np.mean(x)

    theta_power = bandpower_welch(x, fs, *EEG_BANDS["theta"])
    alpha_power = bandpower_welch(x, fs, *EEG_BANDS["alpha"])
    beta_power = bandpower_welch(x, fs, *EEG_BANDS["beta"])
    total_power = bandpower_welch(x, fs, *EEG_TOTAL_POWER_RANGE)
    peak_freq = peak_frequency_welch(x, fs, *EEG_TOTAL_POWER_RANGE)

    power_beta = beta_power / (total_power + EPS) if np.isfinite(beta_power) and np.isfinite(total_power) else np.nan
    alpha_beta = alpha_power / (beta_power + EPS) if np.isfinite(alpha_power) and np.isfinite(beta_power) else np.nan
    theta_beta = theta_power / (beta_power + EPS) if np.isfinite(theta_power) and np.isfinite(beta_power) else np.nan

    return {
        f"{prefix}_power_beta": float(power_beta) if np.isfinite(power_beta) else np.nan,
        f"{prefix}_alpha_beta": float(alpha_beta) if np.isfinite(alpha_beta) else np.nan,
        f"{prefix}_theta_beta": float(theta_beta) if np.isfinite(theta_beta) else np.nan,
        f"{prefix}_peak_frequency_hz": float(peak_freq) if np.isfinite(peak_freq) else np.nan,
    }

def compute_ecg_window_features(x, fs):
    x = np.asarray(x, dtype=float)
    if len(x) < max(8, int(fs)):
        return {
            "ecg_rms": np.nan,
            "ecg_peak_frequency_hz": np.nan,
        }

    x_centered = x - np.mean(x)
    ecg_rms = float(np.sqrt(np.mean(x_centered ** 2)))
    ecg_peak_freq = peak_frequency_welch(x_centered, fs, *ECG_TOTAL_POWER_RANGE)

    return {
        "ecg_rms": ecg_rms,
        "ecg_peak_frequency_hz": float(ecg_peak_freq) if np.isfinite(ecg_peak_freq) else np.nan,
    }

# =========================================================
# BLE CALLBACK
# =========================================================
def on_notify(_, data: bytearray):
    try:
        rxq.put_nowait(bytes(data))
    except Exception:
        pass

async def ble_main():
    dev = await BleakScanner.find_device_by_filter(
        lambda d, ad: d and d.name == TARGET_NAME,
        timeout=10.0
    )
    if dev is None:
        print("Could not find EMG_Sensor. Is it advertising?")
        return

    print(f"Found: {dev.address}: {dev.name}")
    async with BleakClient(dev) as client:
        print("Connected:", client.is_connected)
        await client.start_notify(CHAR_UUID, on_notify)
        print("Notifying... close plot window to stop.")
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
        finally:
            try:
                await client.stop_notify(CHAR_UUID)
            except Exception:
                pass

def ble_thread_entry():
    try:
        asyncio.run(ble_main())
    except Exception as e:
        print("BLE thread exception:", repr(e))

# =========================================================
# PARSER: drain queue, convert, filter, save
# =========================================================
def drain_and_parse(max_packets=2000):
    global t0_us, t0_wall_dt, _last_rate_print, _samp_count

    parsed = 0
    while parsed < max_packets:
        try:
            data = rxq.get_nowait()
        except queue.Empty:
            break

        parsed += 1
        if len(data) != BYTES_PER_NOTIFY:
            continue

        try:
            vals = struct.unpack(UNPACK_FMT, data)
        except Exception:
            continue

        samples = [vals[0:5], vals[5:10]]

        for (t_us, c1, c2, c3, c4) in samples:
            if t0_us is None:
                t0_us = t_us
                t0_wall_dt = datetime.now()

            t_s = (t_us - t0_us) * 1e-6
            abs_dt = t0_wall_dt + timedelta(seconds=t_s)
            abs_hms_ms = fmt_hms_ms(abs_dt)
            abs_iso_ms = fmt_iso_ms(abs_dt)

            raw_counts = [int(c1), int(c2), int(c3), int(c4)]
            raw_v_postafe = [counts_to_postafe_volts(x) for x in raw_counts]
            raw_v_electrode = [postafe_to_electrode_volts(v) for v in raw_v_postafe]

            filt_v_electrode = []
            for ch in range(4):
                chain = live_chains[ch]
                chain.prime(raw_v_electrode[ch])
                filt_v_electrode.append(chain.step(raw_v_electrode[ch]))

            disp_vals = [
                filt_v_electrode[0] * 1e6,
                filt_v_electrode[1] * 1e6,
                filt_v_electrode[2] * 1e6,
                filt_v_electrode[3] * 1e3,
            ]

            with lock:
                t_buf.append(t_s)
                for ch in range(4):
                    disp_bufs[ch].append(disp_vals[ch])

                rec_rows.append((
                    t_s,
                    abs_hms_ms,
                    abs_iso_ms,

                    raw_counts[0], raw_counts[1], raw_counts[2], raw_counts[3],

                    raw_v_postafe[0], raw_v_postafe[1], raw_v_postafe[2], raw_v_postafe[3],
                    raw_v_electrode[0], raw_v_electrode[1], raw_v_electrode[2], raw_v_electrode[3],

                    filt_v_electrode[0], filt_v_electrode[1], filt_v_electrode[2], filt_v_electrode[3]
                ))

            _samp_count += 1

    now = time.time()
    if now - _last_rate_print >= 1.0:
        print(f"rx ~{_samp_count} samples/s")
        _samp_count = 0
        _last_rate_print = now

# =========================================================
# SAVE FILES
# =========================================================
def make_out_dir(base_ts: str) -> Path:
    out_dir = Path("recordings") / f"{base_ts}_{SESSION_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_main_csv(out_dir: Path) -> Path:
    path = out_dir / "main_recording.csv"

    with lock:
        rows = list(rec_rows)

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_s",
            "abs_time_hms_ms",
            "abs_time_iso_ms",

            "raw_ch1_counts", "raw_ch2_counts", "raw_ch3_counts", "raw_ch4_counts",

            "raw_ch1_V_postAFE", "raw_ch2_V_postAFE", "raw_ch3_V_postAFE", "raw_ch4_V_postAFE",
            "raw_ch1_V_electrode", "raw_ch2_V_electrode", "raw_ch3_V_electrode", "raw_ch4_V_electrode",

            "filt_ch1_V_electrode", "filt_ch2_V_electrode", "filt_ch3_V_electrode", "filt_ch4_V_electrode"
        ])
        w.writerows(rows)

    print(f"Saved main CSV: {path.resolve()}")
    return path

def load_main_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

# =========================================================
# OFFLINE PROCESSING
# =========================================================
def run_offline_processing(main_csv_path: Path, out_dir: Path):
    rows = load_main_csv(main_csv_path)
    if not rows:
        print("No data rows found. Skipping offline processing.")
        return

    t = np.asarray([float(r["t_s"]) for r in rows], dtype=np.float64)

    data_elec = np.vstack([
        np.asarray([float(r["raw_ch1_V_electrode"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["raw_ch2_V_electrode"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["raw_ch3_V_electrode"]) for r in rows], dtype=np.float64),
        np.asarray([float(r["raw_ch4_V_electrode"]) for r in rows], dtype=np.float64),
    ])

    fs_est = estimate_fs_from_t(t)
    print(f"Estimated sampling rate: {fs_est:.3f} Hz")

    # -------------------------
    # Post-processing
    # -------------------------
    if MNE_POST_ENABLED:
        try:
            import mne

            info = mne.create_info(
                ch_names=CHANNEL_NAMES,
                sfreq=fs_est,
                ch_types=CHANNEL_TYPES
            )
            raw = mne.io.RawArray(data_elec.copy(), info, verbose=False)

            raw_eeg = raw.copy().filter(
                l_freq=EEG_POST_L_FREQ,
                h_freq=EEG_POST_H_FREQ,
                picks=[0, 1, 2],
                verbose=False
            )

            raw_ecg = raw.copy().filter(
                l_freq=ECG_POST_L_FREQ,
                h_freq=ECG_POST_H_FREQ,
                picks=[3],
                verbose=False
            )

            eeg_post = raw_eeg.get_data(picks=[0, 1, 2])
            ecg_post = raw_ecg.get_data(picks=[3])[0]

        except Exception as e:
            print(f"MNE post-processing failed, falling back to SciPy only: {repr(e)}")

            eeg_post = np.zeros((3, len(t)), dtype=np.float64)
            for ch in range(3):
                x = data_elec[ch]
                sos_hp = butter_sos_highpass(fs_est, EEG_POST_L_FREQ, order=2)
                sos_lp = butter_sos_lowpass(fs_est, EEG_POST_H_FREQ, order=4)
                y = apply_zero_phase_sos(sos_hp, x)
                y = apply_zero_phase_sos(sos_lp, y)
                eeg_post[ch] = y

            x = data_elec[3]
            sos_hp = butter_sos_highpass(fs_est, ECG_POST_L_FREQ, order=2)
            sos_lp = butter_sos_lowpass(fs_est, ECG_POST_H_FREQ, order=4)
            ecg_post = apply_zero_phase_sos(sos_hp, x)
            ecg_post = apply_zero_phase_sos(sos_lp, ecg_post)
    else:
        eeg_post = np.zeros((3, len(t)), dtype=np.float64)
        for ch in range(3):
            x = data_elec[ch]
            sos_hp = butter_sos_highpass(fs_est, EEG_POST_L_FREQ, order=2)
            sos_lp = butter_sos_lowpass(fs_est, EEG_POST_H_FREQ, order=4)
            y = apply_zero_phase_sos(sos_hp, x)
            y = apply_zero_phase_sos(sos_lp, y)
            eeg_post[ch] = y

        x = data_elec[3]
        sos_hp = butter_sos_highpass(fs_est, ECG_POST_L_FREQ, order=2)
        sos_lp = butter_sos_lowpass(fs_est, ECG_POST_H_FREQ, order=4)
        ecg_post = apply_zero_phase_sos(sos_hp, x)
        ecg_post = apply_zero_phase_sos(sos_lp, ecg_post)

    # -------------------------
    # Save offline_postprocessed.csv
    # -------------------------
    post_csv_path = out_dir / "offline_postprocessed.csv"
    with post_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_s",
            "abs_time_iso_ms",
            "eeg1_V_electrode_post",
            "eeg2_V_electrode_post",
            "eeg3_V_electrode_post",
            "ecg_V_electrode_post",
        ])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.6f}",
                rows[i]["abs_time_iso_ms"],
                f"{eeg_post[0, i]:.9f}",
                f"{eeg_post[1, i]:.9f}",
                f"{eeg_post[2, i]:.9f}",
                f"{ecg_post[i]:.9f}",
            ])
    print(f"Saved offline post-processed CSV: {post_csv_path.resolve()}")

    # -------------------------
    # Sliding-window features
    # -------------------------
    window_n = int(round(FEATURE_WINDOW_SEC * fs_est))
    step_n = int(round(FEATURE_STEP_SEC * fs_est))

    if window_n < 8:
        raise ValueError("FEATURE_WINDOW_SEC is too short.")
    if len(t) < window_n:
        print("Recording is shorter than one full feature window. No features saved.")
        return

    feature_rows = []
    window_index = 0

    for start in range(0, len(t) - window_n + 1, step_n):
        end = start + window_n

        row = {
            "window_index": window_index,
            "abs_time_iso_ms": rows[start]["abs_time_iso_ms"],
            "t_window_start_s": float(t[start]),
            "t_window_end_s": float(t[end - 1]),
            "t_window_center_s": float(0.5 * (t[start] + t[end - 1])),
        }

        row.update(compute_eeg_window_features(eeg_post[0, start:end], fs_est, "eeg1"))
        row.update(compute_eeg_window_features(eeg_post[1, start:end], fs_est, "eeg2"))
        row.update(compute_eeg_window_features(eeg_post[2, start:end], fs_est, "eeg3"))
        row.update(compute_ecg_window_features(ecg_post[start:end], fs_est))

        feature_rows.append(row)
        window_index += 1

    features_csv_path = out_dir / "features.csv"
    fieldnames = list(feature_rows[0].keys())

    with features_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in feature_rows:
            w.writerow(row)

    print(f"Saved features CSV: {features_csv_path.resolve()}")

    # -------------------------
    # Summary plot
    # -------------------------
    try:
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 9))
        fig.suptitle("Offline post-processed signals")

        axs[0].plot(t, eeg_post[0] * 1e6, lw=0.7)
        axs[0].set_ylabel("EEG1 (uV)")
        axs[0].grid(True)

        axs[1].plot(t, eeg_post[1] * 1e6, lw=0.7)
        axs[1].set_ylabel("EEG2 (uV)")
        axs[1].grid(True)

        axs[2].plot(t, eeg_post[2] * 1e6, lw=0.7)
        axs[2].set_ylabel("EEG3 (uV)")
        axs[2].grid(True)

        axs[3].plot(t, ecg_post * 1e3, lw=0.7)
        axs[3].set_ylabel("ECG (mV)")
        axs[3].set_xlabel("Time (s)")
        axs[3].grid(True)

        axs[-1].xaxis.set_major_locator(mticker.MultipleLocator(X_TICK_STEP_S))
        plt.tight_layout()

        plot_path = out_dir / "offline_postprocessed_plot.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path.resolve()}")
        plt.show()
    except Exception as e:
        print(f"Plot save/display failed: {repr(e)}")

# =========================================================
# LIVE PLOT
# =========================================================
def run_plot():
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.18)

    lines = []
    ylabels = ["EEG1 (uV)", "EEG2 (uV)", "EEG3 (uV)", "ECG (mV)"]
    for i, ax in enumerate(axs):
        line, = ax.plot([], [], lw=1)
        lines.append(line)
        ax.set_ylabel(ylabels[i])
        ax.grid(True)

    axs[3].set_xlabel("Time (s)")
    axs[3].xaxis.set_major_locator(mticker.MultipleLocator(X_TICK_STEP_S))

    if FIXED_Y:
        axs[0].set_ylim(*EEG_YLIM_UV)
        axs[1].set_ylim(*EEG_YLIM_UV)
        axs[2].set_ylim(*EEG_YLIM_UV)
        axs[3].set_ylim(*ECG_YLIM_MV)

    def update(_):
        drain_and_parse()

        with lock:
            if len(t_buf) < 5:
                return tuple(lines)

            t = np.array(t_buf, dtype=float)
            ys = [np.array(disp_bufs[ch], dtype=float) for ch in range(4)]

        t_end = t[-1]
        t_start = max(0.0, t_end - PLOT_WINDOW_S)
        m = t >= t_start

        t_plot = t[m]
        ys_plot = [y[m] for y in ys]

        for ch in range(4):
            lines[ch].set_data(t_plot, ys_plot[ch])

        axs[3].set_xlim(t_start, t_end)
        fig.suptitle(f"Live EEG/ECG | Window {PLOT_WINDOW_S:.1f}s")

        return tuple(lines)

    ani = animation.FuncAnimation(
        fig, update, interval=33, blit=False, cache_frame_data=False
    )

    base_ts = session_timestamp()
    out_dir = make_out_dir(base_ts)

    try:
        plt.show()
    finally:
        stop_event.set()
        time.sleep(0.2)

        main_csv_path = save_main_csv(out_dir)
        run_offline_processing(main_csv_path, out_dir)

# =========================================================
# MAIN
# =========================================================
def main():
    print("Starting BLE thread...")
    print("Close the plot window to stop and save.\n")

    threading.Thread(target=ble_thread_entry, daemon=True).start()
    run_plot()

if __name__ == "__main__":
    main()