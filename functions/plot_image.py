import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from obspy import UTCDateTime

import matplotlib.font_manager as fm
font_dirs = ['/storage/vast-gfz-hpc-01/home/kshitkar/fonts/arial']  # replace with the path to your font file
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)
plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 6,        # Set legend font size
    'figure.figsize': (7, 3.5) # Set figure size in inches
})

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # Increase the chunk size limit
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5  # Adjust this value if needed

def plot_image(st, predicted_output, target_output, timestamps,
                image_dir:str, test_julday, val_julday, interval, trim=True):    
    print(f"{'Plotting Image':-^30}")
    times = [UTCDateTime(t).matplotlib_date for t in np.concatenate(timestamps)]
    target_output = np.concatenate(target_output)
    predicted_output = np.concatenate(predicted_output)
    if trim:
        time_window = pd.read_csv("./label/correct_metrics_time_window.csv", index_col=False)
        time_window['Start_Time'] = time_window['Start_Time'].apply(lambda x: UTCDateTime(x))
        time_window['End_Time'] = time_window['End_Time'].apply(lambda x: UTCDateTime(x))
        time_window = time_window[time_window['Start_Time'] < UTCDateTime(year=2020, julday = 1)]
        time_window['Julday'] = time_window['Start_Time'].apply(lambda x: x.julday)
        time_window = time_window[time_window['Julday'] == test_julday]
        time_window.reset_index(inplace=True, drop=True)
        print(time_window)
        start_time, end_time = UTCDateTime(time_window.iloc[0,0]), UTCDateTime(time_window.iloc[0,-2])
        print(start_time, end_time)
        st.trim(starttime=start_time, endtime=end_time)
        mat_start_time = start_time.matplotlib_date
        mat_end_time = end_time.matplotlib_date
        idx_start = np.where(np.equal(times, mat_start_time))[0][0]  # Index of mat_start_time
        idx_end = np.where(np.equal(times, mat_end_time))[0][0]      # Index of mat_end_time
        times = times[idx_start: idx_end]
        target_output = target_output[idx_start: idx_end]
        predicted_output = predicted_output[idx_start: idx_end]

    fig, ax1 = plt.subplots(1,1)
    ax1.plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
    ax1.set_ylabel(r"Amplitude (mm/s)");
    ax1.set_ylim(-1.7, 1.7);
    ax = ax1.twinx()
    ax.plot(times, target_output, label="Impact Force Target [kN]", alpha=0.8, color='r',linewidth=1)
    ax.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("Normal Force [kN]");
    ax.set_ylim(0,350);
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f"{image_dir}/{test_julday}_{val_julday}_{interval}.png", dpi=300)
    plt.close()
    return None

def plot_image_test(st, predicted_output, timestamps,
                        image_dir:str, julday, interval):    
    print(f"{'Plotting Image':-^30}")
    times = [UTCDateTime(t).matplotlib_date for t in np.concatenate(timestamps)]
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.8)
    ax1.set_ylabel(r"Amplitude (mm/s)");
    ax1.set_ylim(-1.5, 1.5);
    ax = ax1.twinx()
    ax.plot(times, np.concatenate(predicted_output), label="Model Prediction", alpha=0.8, color='b',linewidth=1)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("Normal Force [kN]");
    ax.set_ylim(0,350);
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(f"{image_dir}/{julday}.png", dpi=300)
    plt.close()
    return None