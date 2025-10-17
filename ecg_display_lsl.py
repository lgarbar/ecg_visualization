import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks, detrend
import matplotlib
import time
import collections
from pylsl import StreamInlet, resolve_byprop

matplotlib.rcParams['font.family'] = 'DejaVu Sans Mono'

streams = resolve_byprop('name', 'OpenSignals', timeout=10)
if len(streams) == 0:
    raise RuntimeError("No OpenSignals stream found.")
inlet = StreamInlet(streams[0])
info = inlet.info()

channels = info.desc().child('channels')
ECG_CH = None
chan = channels.child('channel')
idx = 0
while chan is not None:
    label = chan.child_value('label')
    name = chan.child_value('name')
    if (label and 'ECG' in label) or (name and 'ECG' in name):
        ECG_CH = idx
        break
    idx += 1
    chan = chan.next_sibling()

if ECG_CH is None:
    raise RuntimeError("No channel with 'ECG' in the label or name found")

print(f"Found ECG channel at index {ECG_CH}")

window_seconds = 2
display_fs = 100
sr = int(info.nominal_srate())
downsample_factor = int(sr / display_fs)

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_title('Populating Buffer. Please wait a few seconds...',
             color='white', fontsize=18, weight='bold', pad=12)

line, = ax.plot([], [], color='lime', lw=1.2)
ax.set_xlim(0, window_seconds)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, color='darkgreen', linestyle='-', linewidth=0.3)

rect_ecg = plt.Rectangle((0, -2.5), window_seconds, 5.0,
                         edgecolor='white', facecolor='none', lw=1.0)
ax.add_patch(rect_ecg)
ax.set_ylim(-2.5, 2.5)

hr_box_width = 0.15
hr_box_height = 0.15
hr_box = plt.Rectangle((0.98 - hr_box_width, 0.775), hr_box_width, hr_box_height,
                       transform=ax.transAxes, edgecolor='white', facecolor='none', lw=1.0)
ax.add_patch(hr_box)

hr_text = ax.text(0.98 - hr_box_width/2, 0.85, '', transform=ax.transAxes,
                  fontsize=16, color='red', weight='bold',
                  ha='center', va='center')

def fast_hr(ecg_slice, sr=1000):
    if len(ecg_slice) < sr * 3:
        return 0
    peaks, _ = find_peaks(ecg_slice, distance=sr*0.5, height=np.mean(ecg_slice))
    if len(peaks) < 2:
        return 0
    rr_intervals = np.diff(peaks) / sr
    return int(60 / np.mean(rr_intervals))

buffer = collections.deque(maxlen=int(2 * sr))
hr_buffer = collections.deque(maxlen=int(5 * sr))
last_hr = 0
hr_update_interval = 5
last_update_time = time.time()
y_scale = 1

def on_key(event):
    global y_scale
    if event.key == 'up':
        y_scale = max(0.5, y_scale * 0.8)
    elif event.key == 'down':
        y_scale = min(10.0, y_scale * 1.25)
    ax.set_ylim(-y_scale, y_scale)

fig.canvas.mpl_connect('key_press_event', on_key)

def init():
    line.set_data([], [])
    hr_text.set_text('')
    return line, hr_text

def update(frame):
    global last_hr, last_update_time

    samples, _ = inlet.pull_chunk(timeout=0.0)
    if samples:
        for s in samples:
            val = s[ECG_CH]
            buffer.append(val)
            hr_buffer.append(val)

    if len(buffer) < sr:
        ax.set_title("Populating Buffer. Please wait a moment...")
        return line, hr_text
    else:
        ax.set_title("HR Monitor")

    display_signal = detrend(np.array(buffer)[::downsample_factor])
    t = np.arange(len(display_signal)) / display_fs
    line.set_data(t, display_signal)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-y_scale, y_scale)

    if time.time() - last_update_time > hr_update_interval and len(hr_buffer) >= 3 * sr:
        last_hr = fast_hr(np.array(hr_buffer), sr)
        last_update_time = time.time()

    hr_text.set_text(f'{last_hr} bpm')
    return line, hr_text

ani = animation.FuncAnimation(fig, update, init_func=init,
                              blit=False, interval=20)

plt.show()
