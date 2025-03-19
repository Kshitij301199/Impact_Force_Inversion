import numpy as np
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
                image_dir:str, test_julday, val_julday, interval):    
    print(f"{'Plotting Image':-^30}")
    times = [UTCDateTime(t).matplotlib_date for t in np.concatenate(timestamps)]
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.8)
    ax1.set_ylabel(r"Amplitude (mm/s)");
    ax1.set_ylim(-1.5, 1.5);
    ax = ax1.twinx()
    ax.plot(times, np.concatenate(target_output), label="Impact Force Target [kN]", alpha=0.6, color='r',linewidth=1)
    ax.plot(times, np.concatenate(predicted_output), label="Model Prediction", alpha=0.8, color='b',linewidth=1)
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