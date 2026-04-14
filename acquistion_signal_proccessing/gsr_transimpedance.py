"""
GSR Live Conductance Plotter with IQ Demodulation
==================================================
Reads two ADC channels (voltage, current) from STM32 over serial,
performs lock-in amplifier (IQ demodulation) to extract skin conductance,
and plots it live.

Sliding-window version:
- lock-in window stays long for smooth output
- new conductance value computed every STEP_SIZE samples
- example: 160-sample window, 8-sample step -> ~125 updates/s
"""

import sys
import time
import argparse
import threading
import collections
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import serial.tools.list_ports
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BAUD_RATE = 115200
ADC_BITS = 12
ADC_MAX = (1 << ADC_BITS) - 1
VREF = 3.3  # STM32 ADC reference voltage

# Excitation sine parameters (must match STM32 firmware)
SINE_TABLE_LEN = 32
DECIMATION_K = 2  # sine table advances every K ADC conversions; with 2-ch scan, 1 advance per sample-pair

# Derived
SAMPLES_PER_SINE_CYCLE = SINE_TABLE_LEN  # 32 samples per full excitation cycle
SAMPLE_RATE = 1000.0  # Hz, complete (ch0,ch1) pairs
F_EXCITATION = 31.25  # measured on oscilloscope

# Lock-in settings — TUNABLE
NUM_CYCLES_AVG = 5          # was 10 — 5 cycles still plenty for noise rejection
LOCKIN_WINDOW = SAMPLES_PER_SINE_CYCLE * NUM_CYCLES_AVG   # 160 samples
STEP_SIZE = 8               # was 32 — slide by 8 samples for ~125 outputs/s
CONDUCTANCE_RATE = SAMPLE_RATE / STEP_SIZE                 # 125 Hz

# Plot history
PLOT_HISTORY_SEC = 30

# Sense resistor
R_SENSE_OHM = 10_000.0

# Optional calibration
CALIBRATION_G_REAL_US = None

# Recording output directory
RECORDING_DIR = "IQrecordings"


# ---------------------------------------------------------------------------
# Reference sine/cosine tables
# ---------------------------------------------------------------------------
def build_reference_tables(n_samples_per_cycle, num_cycles):
    """Build sin/cos reference for the full lock-in window."""
    n = n_samples_per_cycle * num_cycles
    t = np.arange(n)
    ref_i = np.sin(2.0 * np.pi * t / n_samples_per_cycle)
    ref_q = np.cos(2.0 * np.pi * t / n_samples_per_cycle)
    return ref_i, ref_q

REF_I, REF_Q = build_reference_tables(SAMPLES_PER_SINE_CYCLE, NUM_CYCLES_AVG)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
def fmt_hms_ms(dt: datetime) -> str:
    return dt.strftime("%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"

def fmt_iso_ms(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


# ---------------------------------------------------------------------------
# IQ Demodulation / Lock-in amplifier
# ---------------------------------------------------------------------------
class LockInDemodulator:
    def __init__(self, window_size=LOCKIN_WINDOW, step_size=STEP_SIZE):
        self.window = window_size
        self.step = step_size
        self.buf_v = collections.deque(maxlen=window_size)
        self.buf_i = collections.deque(maxlen=window_size)
        self.r_sense = R_SENSE_OHM
        self.calibration_factor = 1.0
        self.samples_since_last_output = 0

        # Pre-compute reference arrays sized to this window
        self.ref_i, self.ref_q = build_reference_tables(
            SAMPLES_PER_SINE_CYCLE,
            max(1, window_size // SAMPLES_PER_SINE_CYCLE)
        )
        # Trim/extend to exact window size
        if len(self.ref_i) != window_size:
            n = window_size
            t = np.arange(n)
            self.ref_i = np.sin(2.0 * np.pi * t / SAMPLES_PER_SINE_CYCLE)
            self.ref_q = np.cos(2.0 * np.pi * t / SAMPLES_PER_SINE_CYCLE)

    def add_sample(self, v_raw, i_raw):
        """
        Feed one sample pair (raw ADC values).
        Returns conductance in microsiemens when enough new samples
        have arrived, else None.
        """
        v = v_raw - ADC_MAX / 2.0
        i = i_raw - ADC_MAX / 2.0

        self.buf_v.append(v)
        self.buf_i.append(i)

        if len(self.buf_v) < self.window:
            return None

        self.samples_since_last_output += 1

        if self.samples_since_last_output < self.step:
            return None

        self.samples_since_last_output = 0
        return self._compute_conductance()

    def _compute_conductance(self):
        buf_v = np.array(self.buf_v, dtype=float)
        buf_i = np.array(self.buf_i, dtype=float)

        Eo_re = np.dot(buf_v, self.ref_i)
        Eo_im = np.dot(buf_v, self.ref_q)

        Io_re = np.dot(buf_i, self.ref_i)
        Io_im = np.dot(buf_i, self.ref_q)

        denom = Eo_re**2 + Eo_im**2
        if denom < 1e-20:
            return 0.0

        G_raw = (Io_re * Eo_re + Io_im * Eo_im) / denom
        G_us = (G_raw / self.r_sense) * 1e6

        # TIA inversion correction
        G_us = -G_us

        return G_us * self.calibration_factor

    def set_calibration(self, g_measured, g_real):
        if g_measured != 0:
            self.calibration_factor = g_real / g_measured
            print(f"[CAL] factor = {self.calibration_factor:.6f}  "
                  f"(measured {g_measured:.4f} µS, real {g_real:.4f} µS)")


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------
class SerialReader:
    def __init__(self, port, baud=BAUD_RATE):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.running = True
        self.demod = LockInDemodulator()
        self.lock = threading.Lock()

        max_points = int(PLOT_HISTORY_SEC * CONDUCTANCE_RATE) + 100
        self.g_history = collections.deque(maxlen=max_points)
        self.t_history = collections.deque(maxlen=max_points)

        self.t0 = time.time()
        self.sample_count = 0
        self.error_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Recording state
        self.t0_wall_dt = None
        self.t0_wall_epoch = None
        self.csv_rows = []
        self.conductance_count = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)
        self.ser.close()

    def _run(self):
        buf = b""
        while self.running:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._process_line(line)
            except serial.SerialException:
                print("[ERR] Serial connection lost.")
                self.running = False
            except Exception as e:
                self.error_count += 1
                if self.error_count < 5:
                    print(f"[ERR] {e}")

    def _process_line(self, line):
        try:
            parts = line.decode("ascii", errors="ignore").split(",")
            if len(parts) != 2:
                return
            ch0 = int(parts[0])
            ch1 = int(parts[1])
        except (ValueError, UnicodeDecodeError):
            return

        if self.t0_wall_dt is None:
            self.t0_wall_dt = datetime.now()
            self.t0_wall_epoch = time.time()
            self._save_first_timestamp()

        self.sample_count += 1
        g = self.demod.add_sample(ch0, ch1)

        if g is not None:
            t_now = time.time() - self.t0
            with self.lock:
                self.t_history.append(t_now)
                self.g_history.append(g)

            self.conductance_count += 1

            t_rel = self.sample_count / SAMPLE_RATE
            abs_dt = self.t0_wall_dt + timedelta(seconds=t_rel)

            self.csv_rows.append([
                f"{t_rel:.6f}",
                fmt_hms_ms(abs_dt),
                fmt_iso_ms(abs_dt),
                f"{g:.6f}",
            ])

    def _save_first_timestamp(self):
        out_dir = Path(RECORDING_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "first_timestamp.txt"

        with path.open("w") as f:
            f.write(f"HH-MM-SS-MS: {fmt_hms_ms(self.t0_wall_dt)}\n")
            f.write(f"ISO: {fmt_iso_ms(self.t0_wall_dt)}\n")
            f.write(f"Epoch_s: {self.t0_wall_epoch:.6f}\n")

        print(f"[REC] First-sample timestamp saved to {path}")

    def save_csv(self):
        out_dir = Path(RECORDING_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "conductance_data.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_s",
                "abs_time_hms_ms",
                "abs_time_iso_ms",
                "conductance_uS",
            ])
            writer.writerows(self.csv_rows)

        print(f"[REC] {len(self.csv_rows)} conductance values saved to {csv_path}")

    def get_data(self):
        with self.lock:
            return (
                np.array(self.t_history, dtype=float),
                np.array(self.g_history, dtype=float)
            )


# ---------------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------------
def run_live_plot(reader, plot_history=PLOT_HISTORY_SEC):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.canvas.manager.set_window_title("GSR Live — Skin Conductance")

    line_raw, = ax.plot([], [], color="#90CAF9", linewidth=0.8, alpha=0.5, label="Raw")
    line_smooth, = ax.plot([], [], color="#E65100", linewidth=2.2, label="Smoothed")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conductance (µS)")
    ax.set_title("GSR - Skin Conductance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Smoothing window scaled up for higher output rate
    SMOOTH_WINDOW = 32  # was 8 — covers same ~250ms at 125 Hz output

    #stats_text = ax.text(
    #    0.01, 0.97, "", transform=ax.transAxes,
    #    fontsize=9, verticalalignment="top", fontfamily="monospace",
    #    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    #)

    def moving_average(arr, w):
        if len(arr) < w:
            w = len(arr)
        if w < 1:
            return arr
        kernel = np.ones(w) / w
        smoothed = np.convolve(arr, kernel, mode="same")
        for k in range(min(w // 2, len(arr))):
            smoothed[k] = np.mean(arr[:k + w // 2 + 1])
        for k in range(max(0, len(arr) - w // 2), len(arr)):
            smoothed[k] = np.mean(arr[k - w // 2:])
        return smoothed

    def init():
        line_raw.set_data([], [])
        line_smooth.set_data([], [])
        return line_raw, line_smooth#, stats_text

    def update(frame):
        t, g = reader.get_data()
        if len(t) < 2:
            return line_raw, line_smooth#, stats_text

        line_raw.set_data(t, g)
        g_smooth = moving_average(g, SMOOTH_WINDOW)
        line_smooth.set_data(t, g_smooth)

        t_max = t[-1]
        t_min = max(0, t_max - plot_history)
        ax.set_xlim(t_min, t_max + 0.5)

        mask = t >= t_min
        g_visible = g[mask]
        g_smooth_visible = g_smooth[mask]

        if len(g_visible) > 0:
            g_lo = min(g_visible.min(), g_smooth_visible.min())
            g_hi = max(g_visible.max(), g_smooth_visible.max())
            margin = max(0.5, (g_hi - g_lo) * 0.1)
            #ax.set_ylim(g_lo - margin, g_hi + margin)
            ax.set_ylim(10,30)

        # stats_text.set_text(
        #     f"Raw:      {g[-1]:8.3f} µS\n"
        #     f"Smoothed: {g_smooth[-1]:8.3f} µS\n"
        #     f"Mean:     {g_visible.mean():8.3f} µS\n"
        #     f"Noise:    {g_visible.std():8.4f} µS\n"
        #     f"Samples:  {reader.sample_count}\n"
        #     f"G rate:   {reader.conductance_count / max(1e-9, time.time() - reader.t0):.1f} Hz"
        # )

        return line_raw, line_smooth#, stats_text

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        interval=50, blit=False, cache_frame_data=False   # 50ms refresh for smoother plot
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Calibration routine
# ---------------------------------------------------------------------------
def run_calibration(reader, g_real_us, duration=5.0):
    print(f"[CAL] Calibrating with {g_real_us} µS reference...")
    print(f"[CAL] Collecting for {duration} seconds...")
    time.sleep(duration)

    _, g = reader.get_data()
    if len(g) < 3:
        print("[CAL] Not enough data — skipping calibration.")
        return

    g_mean = np.mean(g)
    print(f"[CAL] Mean raw conductance: {g_mean:.4f} µS")
    reader.demod.set_calibration(g_mean, g_real_us)

    with reader.lock:
        reader.g_history.clear()
        reader.t_history.clear()
    reader.t0 = time.time()


# ---------------------------------------------------------------------------
# Port selection helper
# ---------------------------------------------------------------------------
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [(p.device, p.description) for p in ports]

def choose_port(requested=None):
    if requested:
        return requested

    ports = list_serial_ports()
    if not ports:
        print("No serial ports found. Connect your STM32 and try again.")
        sys.exit(1)

    if len(ports) == 1:
        print(f"Auto-selected: {ports[0][0]} ({ports[0][1]})")
        return ports[0][0]

    print("Available serial ports:")
    for i, (dev, desc) in enumerate(ports):
        print(f"  [{i}] {dev}  —  {desc}")

    while True:
        try:
            choice = int(input("Select port number: "))
            return ports[choice][0]
        except (ValueError, IndexError):
            print("Invalid selection.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GSR live conductance plotter with IQ demodulation"
    )
    parser.add_argument("-p", "--port", type=str, default=None,
                        help="Serial port (auto-detect if omitted)")
    parser.add_argument("-b", "--baud", type=int, default=BAUD_RATE,
                        help=f"Baud rate (default: {BAUD_RATE})")
    parser.add_argument("-r", "--r-sense", type=float, default=R_SENSE_OHM,
                        help=f"Transimpedance resistor in ohms (default: {R_SENSE_OHM})")
    parser.add_argument("-c", "--calibrate", type=float, default=None,
                        help="Calibrate with known conductance in µS (e.g. 100 for 10K)")
    parser.add_argument("--history", type=float, default=PLOT_HISTORY_SEC,
                        help=f"Plot history in seconds (default: {PLOT_HISTORY_SEC})")
    parser.add_argument("--step", type=int, default=STEP_SIZE,
                        help=f"Sliding window step size (default: {STEP_SIZE})")
    parser.add_argument("--cycles", type=int, default=NUM_CYCLES_AVG,
                        help=f"Number of excitation cycles to average (default: {NUM_CYCLES_AVG})")
    args = parser.parse_args()

    # Allow runtime override of step and window
    step = args.step
    n_cycles = args.cycles
    window = SAMPLES_PER_SINE_CYCLE * n_cycles
    cond_rate = SAMPLE_RATE / step

    port = choose_port(args.port)

    print(f"\nConnecting to {port} @ {args.baud} baud...")
    print(f"Excitation freq:  {F_EXCITATION:.3f} Hz")
    print(f"Sample rate:      {SAMPLE_RATE:.0f} Hz (pairs)")
    print(f"Lock-in window:   {window} samples ({window / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"Step size:        {step} samples")
    print(f"Output rate:      ~{cond_rate:.2f} Hz")
    print(f"Recording to:     {RECORDING_DIR}/\n")

    reader = SerialReader(port, args.baud)
    reader.demod = LockInDemodulator(window_size=window, step_size=step)
    reader.demod.r_sense = args.r_sense

    # Resize history buffer for the actual output rate
    max_points = int(PLOT_HISTORY_SEC * cond_rate) + 100
    reader.g_history = collections.deque(maxlen=max_points)
    reader.t_history = collections.deque(maxlen=max_points)

    reader.start()

    try:
        if args.calibrate is not None:
            run_calibration(reader, args.calibrate)

        run_live_plot(reader, args.history)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        reader.stop()
        reader.save_csv()
        print(f"Total ADC sample pairs received: {reader.sample_count}")
        print(f"Total conductance values:        {reader.conductance_count}")


if __name__ == "__main__":
    main()