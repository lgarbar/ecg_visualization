import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks
import matplotlib
import sys
sys.path.append('/Users/danielgarcia-barnett/Desktop/Coding/MoBI_QC/src/MOBI_QC/important_files')
import ecg_qc  

# Safe default font
matplotlib.rcParams['font.family'] = 'DejaVu Sans Mono'

fname = '/Users/danielgarcia-barnett/Desktop/Coding/MoBI_QC/test_data/nwb/sub-M10932848_ses-MOBI2C_task-nasa_run-01_ecg.csv'
df = pd.read_csv(fname, index_col=0)
ecg_signal = df['ECG'].values
sr = ecg_qc.get_sampling_rate(df)  

window_seconds = 5
display_fs = 100
downsample_factor = int(sr / display_fs)
window_points = window_seconds * display_fs

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_title('HR Monitor', color='white', fontsize=18, weight='bold', pad=12)

line, = ax.plot([], [], color='lime', lw=1.5)
ax.set_ylim(np.min(ecg_signal)*1.2, np.max(ecg_signal)*1.2)
ax.set_xlim(0, window_seconds)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True, color='darkgreen', linestyle='-', linewidth=0.3)

# ECG box
rect_ecg = plt.Rectangle((0, np.min(ecg_signal)*1.2), window_seconds, 
                         np.max(ecg_signal)*1.2 - np.min(ecg_signal)*1.2,
                         edgecolor='white', facecolor='none', lw=1.5)
ax.add_patch(rect_ecg)

# HR box
hr_box_width = 0.15
hr_box_height = 0.15
hr_box = plt.Rectangle((0.98 - hr_box_width, 0.775), hr_box_width, hr_box_height,
                       transform=ax.transAxes, edgecolor='white', facecolor='none', lw=1.5)
ax.add_patch(hr_box)

hr_text = ax.text(0.98 - hr_box_width/2, 0.85, '', transform=ax.transAxes,
                  fontsize=16, color='red', weight='bold',
                  ha='center', va='center')

def init():
    line.set_data([], [])
    hr_text.set_text('')
    return line, hr_text

step = display_fs
last_hr = 0
hr_update_interval = 5  # seconds

def fast_hr(ecg_slice, sr=1000):
    if len(ecg_slice) < sr * 3:
        return 0
    peaks, _ = find_peaks(ecg_slice, distance=sr*0.5, height=np.mean(ecg_slice))
    if len(peaks) < 2:
        return 0
    rr_intervals = np.diff(peaks) / sr
    return int(60 / np.mean(rr_intervals))

def update(frame):
    global last_hr
    frame = int(frame)
    start_idx = int(max(0, frame - window_points*downsample_factor))
    display_signal = ecg_signal[start_idx:frame:downsample_factor]

    if len(display_signal) < 2:
        return line, hr_text

    t = np.arange(len(display_signal)) / display_fs
    line.set_data(t, display_signal)
    ax.set_xlim(t[0], t[-1])

    # update HR every hr_update_interval seconds
    if frame % int(sr * hr_update_interval) < step:
        raw_slice_start = max(0, frame - int(sr*5))
        raw_slice = ecg_signal[raw_slice_start:frame]
        last_hr = fast_hr(raw_slice, sr)

    hr_text.set_text(f'{last_hr} bpm')

    return line, hr_text

frames = np.arange(window_points*downsample_factor, len(ecg_signal), step)

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                              blit=True, interval=10000/display_fs)

plt.show()