import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import math
import csv
import time

# =========================
# Settings
# =========================
PORT = "COM3"
BAUDRATE = 115200

FS = 500              # Hz
N = 64                # RMS window (samples) -> 64/500 = 128 ms

MIN_VALID = 0
MAX_VALID = 4095      # 12-bit ADC max

DISPLAY_SEC = 20
MAX_POINTS = int(DISPLAY_SEC * FS)

# --- DC removal (slow) ---
off = 0.0
alpha = 1 / 500

# --- Envelope smoothing (FAST) ---
FC_ENV = 0.5
k_env = 1 - math.exp(-2 * math.pi * FC_ENV / FS)
env_lp = 0.0

# --- Baseline for spike display (VERY SLOW) ---
FC_BASE = 0.05
k_base = 1 - math.exp(-2 * math.pi * FC_BASE / FS)
base_lp = 0.0

# --- Optional extra smoothing of spike channel ---
spike_lp = 0.0
FC_SPIKE = 2.0
k_spike = 1 - math.exp(-2 * math.pi * FC_SPIKE / FS)

# --- RMS window state ---
sq_window = deque(maxlen=N)
sq_sum = 0.0

# =========================
# Timestamp state
# =========================
t0_wall_dt = None
t0_wall_epoch = None

def fmt_hms_ms(dt: datetime) -> str:
    return dt.strftime("%H-%M-%S-") + f"{dt.microsecond // 1000:03d}"

def fmt_iso_ms(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

def save_first_timestamp_txt():
    out_dir = Path("HOW_recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "first_timestamp.txt"

    with path.open("w") as f:
        f.write(f"HH-MM-SS-MS: {fmt_hms_ms(t0_wall_dt)}\n")
        f.write(f"ISO: {fmt_iso_ms(t0_wall_dt)}\n")
        f.write(f"Epoch_s: {t0_wall_epoch:.6f}\n")

    return path

# =========================
# Serial
# =========================
ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
ser.reset_input_buffer()

# =========================
# Plot
# =========================
fig, (ax_raw, ax_env) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax_raw.set_title("a) Raw GSR Signal")
ax_raw.set_ylabel("ADC Counts (0..4095)")
ax_raw.set_ylim(0, 4095)
ax_raw.grid(True, alpha=0.3)

ax_env.set_title("b) GSR Phasic Component")
ax_env.set_xlabel("Time (s)")
ax_env.set_ylabel("Phasic Amplitude (ADC Counts)")
ax_env.set_ylim(-50, 50)   # fixed y-axis
ax_env.grid(True, alpha=0.3)

x_data = deque(maxlen=MAX_POINTS)
raw_data = deque(maxlen=MAX_POINTS)
env_data = deque(maxlen=MAX_POINTS)
base_data = deque(maxlen=MAX_POINTS)
spike_data = deque(maxlen=MAX_POINTS)

line_raw, = ax_raw.plot([], [], linewidth=1, label="raw")
line_env, = ax_env.plot([], [], linewidth=1, label="env (LP)")
line_base, = ax_env.plot([], [], linewidth=1, label="baseline (Tonic Component)")
line_spike, = ax_env.plot([], [], linewidth=1, label="Phasic Component")
#ax_env.legend(loc="upper right")

sample_count = 0
csv_rows = []

# =========================
# Helpers
# =========================
def read_all_available():
    """Read all available integer samples from serial buffer (non-blocking)."""
    samples = []
    while True:
        try:
            if ser.in_waiting <= 0:
                break
            line_in = ser.readline().decode(errors="ignore").strip()
            if not line_in:
                continue
            try:
                v = int(line_in)
                if MIN_VALID <= v <= MAX_VALID:
                    samples.append(v)
            except ValueError:
                pass
        except Exception:
            break
    return samples

def dc_remove(raw):
    """Remove midscale and slow residual offset drift."""
    global off
    y = raw - 2048.0
    off += alpha * (y - off)
    return y - off

def envelope_rms(x):
    """
    RMS over a sliding window of length N.
    Uses a running sum for speed.
    Returns None until window is full.
    """
    global sq_sum
    x2 = x * x

    if len(sq_window) == sq_window.maxlen:
        sq_sum -= sq_window[0]

    sq_window.append(x2)
    sq_sum += x2

    if len(sq_window) < sq_window.maxlen:
        return None

    return math.sqrt(sq_sum / N)

# =========================
# Animation
# =========================
def animate(_frame):
    global sample_count, env_lp, base_lp, spike_lp, t0_wall_dt, t0_wall_epoch

    new_samples = read_all_available()

    for value in new_samples:
        # Start absolute host timestamp on FIRST VALID SAMPLE
        if t0_wall_dt is None:
            t0_wall_dt = datetime.now()
            t0_wall_epoch = time.time()
            save_first_timestamp_txt()

        t_s = sample_count / FS
        abs_dt = t0_wall_dt + timedelta(seconds=t_s)
        abs_hms_ms = fmt_hms_ms(abs_dt)
        abs_iso_ms = fmt_iso_ms(abs_dt)

        # Top plot: raw ADC
        x_data.append(t_s)
        raw_data.append(value)

        # Envelope pipeline
        x = dc_remove(value)
        rms = envelope_rms(x)

        if rms is None:
            env = env_lp
            base = base_lp
            spike = spike_lp
        else:
            env_lp += k_env * (rms - env_lp)
            env = env_lp

            base_lp += k_base * (env - base_lp)
            base = base_lp

            spike = env - base
            spike_lp += k_spike * (spike - spike_lp)
            spike = spike_lp

        env_data.append(env_lp)
        base_data.append(base_lp)
        spike_data.append(spike_lp)

        csv_rows.append([
            t_s,
            abs_hms_ms,
            abs_iso_ms,
            value,
            env_lp,
            base_lp,
            spike_lp
        ])

        sample_count += 1

    # Update plots
    if len(x_data) > 1:
        line_raw.set_data(list(x_data), list(raw_data))
        line_env.set_data(list(x_data), list(env_data))
        line_base.set_data(list(x_data), list(base_data))
        line_spike.set_data(list(x_data), list(spike_data))

        ax_raw.set_xlim(x_data[0], x_data[-1])
        ax_env.set_xlim(x_data[0], x_data[-1])
        

        ax_raw.set_title(
            f"a) Raw GSR Signal"
        )
        ax_env.set_title(
            f"b) GSR Phasic Component"
        )

    return line_raw, line_env, line_base, line_spike

ani = animation.FuncAnimation(
    fig,
    animate,
    interval=20,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()

try:
    plt.show()
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    ser.close()

    out_dir = Path("HOW_recordings")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "spike_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s",
            "abs_time_hms_ms",
            "abs_time_iso_ms",
            "raw_adc",
            "env_lp",
            "base_lp",
            "spike_lp"
        ])
        writer.writerows(csv_rows)

    print(f"Total samples received: {sample_count}")
    print(f"CSV saved as {csv_path}")