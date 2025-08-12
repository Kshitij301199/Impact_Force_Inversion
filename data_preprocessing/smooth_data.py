import argparse
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy.core import UTCDateTime
import os
import sys
import json

import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    'font.size': 8,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 8,        # Set legend font size
    'figure.figsize': (5.5, 3.5) # Set figure size in inches
})
# plt.style.use('science')  # Use the science style for plots

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
from utils import fill_missing_from_Gaussian, get_mean_and_std

def do_smoothing(input_dir):
    min_Fv, min_Fv_std = get_mean_and_std()
    sigma = 1
    for date in tqdm(["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-03", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"], desc="Smoothing data"):
        df = pd.read_csv(f"{input_dir}/{date}.csv", index_col=None)
        for n in [10, 30, 60]:
            window_size = 2 * n + 1  # total window size (symmetric around the center)
            df[f'moving_avg_{n}'] = df['Fv [kN]'].rolling(window=window_size, center=True).mean().apply(lambda x: np.round(x, 4))
            df[f'moving_avg_{n}'] = df[f'moving_avg_{n}'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
        df.to_csv(f"{input_dir}/{date}.csv", index=False)
    return None

def make_plot(date):
    df = pd.read_csv(f"../label/data_processed_dynamic/ILL11/{date}.csv")
    df['Time'] = df['Time'].apply(UTCDateTime)

    for n in [10, 30, 60]:
        window_size = 2 * n + 1  # total window size (symmetric around the center)
        df[f'moving_avg_{n}'] = df['Fv [kN]'].rolling(window=window_size, center=True).mean()

    threshold = 20
    df_thres = df[df['Fv [kN]'] > threshold]

    fig, ax = plt.subplots()
    ax.plot(df_thres['Time'].apply(lambda x: x.matplotlib_date), df_thres['Fv [kN]'], color='red', label="Raw_Data")
    ax.plot(df_thres['Time'].apply(lambda x: x.matplotlib_date), df_thres['moving_avg_10'], color='blue', label=r"$\pm$10 MovAvg")
    ax.plot(df_thres['Time'].apply(lambda x: x.matplotlib_date), df_thres['moving_avg_30'], color='green', label=r"$\pm$30 MovAvg")
    ax.plot(df_thres['Time'].apply(lambda x: x.matplotlib_date), df_thres['moving_avg_60'], color='yellow', label=r"$\pm$60 MovAvg")
    # ax.hlines(thres, xmin=UTCDateTime("2019-07-26").matplotlib_date, xmax=UTCDateTime("2019-07-27").matplotlib_date)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    # ax.set_xlim(UTCDateTime("2019-07-26T17:00:00"), UTCDateTime("2019-07-26T20:00:00"))
    ax.set_ylabel("Impact Force")
    ax.set_xlabel("Time")
    ax.legend()
    fig.tight_layout()
    os.makedirs("./smoothing/img", exist_ok = True)
    fig.savefig("./smoothing/img/2019-08-20.png", dpi=300);
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,4))

    sns.histplot(data=df_thres, x='Fv [kN]', ax=ax[0], color='red', label="Raw_Data", kde=True, stat='density', alpha=0.8)
    sns.histplot(data=df_thres, x='moving_avg_10', ax=ax[0], color='green', label=r"$\pm$10 MovAvg", kde=True, stat='density', alpha=0.3)
    sns.histplot(data=df_thres, x='Fv [kN]', ax=ax[1], color='red', label="Raw_Data", kde=True, stat='density', alpha=0.8)
    sns.histplot(data=df_thres, x='moving_avg_30', ax=ax[1], color='green', label=r"$\pm$30 MovAvg", kde=True, stat='density', alpha=0.3)
    sns.histplot(data=df_thres, x='Fv [kN]', ax=ax[2], color='red', label="Raw_Data", kde=True, stat='density', alpha=0.8)
    sns.histplot(data=df_thres, x='moving_avg_60', ax=ax[2], color='green', label=r"$\pm$60 MovAvg", kde=True, stat='density', alpha=0.3)

    for axes in ax:
        # axes.set_xlim(0,350);
        axes.legend(loc='upper right')
        axes.set_xlabel("Impact Force Distribution")

    fig.tight_layout()
    os.makedirs("./smoothing/dist", exist_ok = True)
    fig.savefig("./smoothing/dist/2019-08-20.png", dpi=300);

    return None

def main(time_shift:str, smooth:bool, plot:bool):
    input_dir = f"../label/data_processed_{time_shift}/ILL11"
    
    if smooth:
        do_smoothing(input_dir)
    
    if plot:
        for date in ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-03", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]:
            make_plot(date)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth data and make plots")
    parser.add_arguement("--time_shift", type=str, default="average", help="Time shift method to use for smoothing")
    parser.add_argument("--smooth", action="store_true", help="Whether to smooth the data")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the data")
    args = parser.parse_args()

    main(time_shift=args.time_shift, smooth=args.smooth, plot=args.plot)




