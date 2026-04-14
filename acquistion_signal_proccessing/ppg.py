import serial
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =========================
# Settings
# =========================
PORT = "COM5"
BAUD = 921600
MAX_POINTS = 400   # about 20 seconds at 20 Hz

# Choose what to show on startup: "diff", "t1", or "t2"
PLOT_MODE = "diff"

# =========================
# Serial
# =========================
ser = serial.Serial(PORT, BAUD, timeout=0.1)

# =========================
# Data buffers
# =========================
t1_data = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)
t2_data = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)
diff_data = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)

# =========================
# Plot setup
# =========================
fig, ax = plt.subplots()

line_t1, = ax.plot(range(MAX_POINTS), list(t1_data), label="t1")
line_t2, = ax.plot(range(MAX_POINTS), list(t2_data), label="t2")
line_diff, = ax.plot(range(MAX_POINTS), list(diff_data), label="t2 - t1")

ax.set_title("Live PPG Debug")
ax.set_xlabel("Samples")
ax.set_ylabel("ADC / Difference")
ax.set_xlim(0, MAX_POINTS - 1)
ax.set_ylim(-500, 4096)
ax.grid(True)
ax.legend()

# Show only one trace at startup if desired
line_t1.set_visible(PLOT_MODE == "t1")
line_t2.set_visible(PLOT_MODE == "t2")
line_diff.set_visible(PLOT_MODE == "diff")


def parse_line(raw: str):
    parts = raw.split(",")
    if len(parts) != 3:
        return None

    try:
        t1 = int(parts[0])
        t2 = int(parts[1])
        diff = int(parts[2])
    except ValueError:
        return None

    if not (0 <= t1 <= 4095 and 0 <= t2 <= 4095):
        return None

    return t1, t2, diff


# =========================
# Update function
# =========================
def update(frame):
    try:
        while ser.in_waiting:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue

            parsed = parse_line(raw)
            if parsed is None:
                continue

            t1, t2, diff = parsed
            t1_data.append(t1)
            t2_data.append(t2)
            diff_data.append(diff)

    except Exception:
        pass

    line_t1.set_ydata(list(t1_data))
    line_t2.set_ydata(list(t2_data))
    line_diff.set_ydata(list(diff_data))

    return line_t1, line_t2, line_diff


# =========================
# Animate
# =========================
ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.show()

