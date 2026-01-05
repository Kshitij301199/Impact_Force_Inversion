import os
import sys
import json
import argparse

with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
    paths = json.load(file)
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
    data_params = json.load(file)
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/event_id_map.json", "r") as file:
    time_config = json.load(file)

sys.path.append(paths['BASE_DIR'])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from obspy.core import UTCDateTime

plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 6,        # Set legend font size
    'figure.figsize': (5.5, 3.5), # Set figure size in inches
    'axes.formatter.limits': (-3, 6),
    'axes.formatter.use_mathtext': True,
    'font.weight': 'bold',
    'axes.labelweight': 'bold'
})

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # Increase the chunk size limit
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 0.5  # Adjust this value if needed

from functions.data_processing.read_data import load_seismic_data_test

def main(julday:int, year:int, station:str, interval_seconds:int, output_dir:str, test_dir:str):
    mapping = {161 : 1, 172 : 2, 182 : 3, 183 : 4, 196 : 5, 207 : 6, 223 : 7, 232 : 8}
    model = 'xLSTM'
    test_dir = f"{test_dir}/{station}"
    # for year in [2021, 2023]:
        # if year == 2019:
        #     juldays = [161, 171, 172, 182, 183, 184, 196, 207, 223, 232]
        # elif year == 2020:
        #     juldays = [156, 159, 160, 162, 168, 169, 181, 210, 229, 243]
        # elif year == 2022:
        #     juldays = [156, 185]
        # elif year == 2021:
        #     juldays = [131, 136, 141, 142, 156, 173, 175, 187, 194, 197, 219, 262]
        # elif year == 2023:
        #     juldays = [153, 161, 193]
        # for julday in juldays:
            # for model in models:            
    output_dir = f"{test_dir}/{model}_{interval_seconds}/{year}"
    img_dir = f"{output_dir}/img"
    df_dir = f"{output_dir}/df"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(df_dir, exist_ok=True)

    df = None
    for model_julday in [161, 172, 182, 183, 196, 207, 223, 232]:
        if df is None:
            df = pd.read_csv(f"{test_dir}/{model}_{interval_seconds}/{year}/{mapping[model_julday]}/df/{julday}.csv")
            df['Predicted_Output'] = df['Predicted_Output']
            df.rename(columns={'Predicted_Output': f'Predicted_Output_{mapping[model_julday]}'}, inplace=True)
        else:
            temp = pd.read_csv(f"{test_dir}/{model}_{interval_seconds}/{year}/{mapping[model_julday]}/df/{julday}.csv")
            assert np.all(df['Timestamps'] == temp['Timestamps'])
            df[f"Predicted_Output_{mapping[model_julday]}"] = temp["Predicted_Output"]
            del temp

    df['Predicted_Output_Mean'] = df[[f"Predicted_Output_{mapping[j]}" for j in mapping]].mean(axis=1)
    df['Predicted_Output_Std'] = df[[f"Predicted_Output_{mapping[j]}" for j in mapping]].std(axis=1)
    df['Predicted_Output_Max'] = df[[f"Predicted_Output_{mapping[j]}" for j in mapping]].max(axis=1)
    df['Predicted_Output_Min'] = df[[f"Predicted_Output_{mapping[j]}" for j in mapping]].min(axis=1)

    df['Timestamps'] = df['Timestamps'].apply(lambda x: UTCDateTime(x))
    df.to_csv(f"{df_dir}/{julday}.csv", index=False)

    times = df['Timestamps'].apply(lambda x: x.matplotlib_date).to_list()

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), sharex=True)
    
    st = load_seismic_data_test(julday=julday, station=station, year=year, component="EHZ", network="9S")
    ax.plot(st[0].times('matplotlib'), st[0].data, color='black', alpha=0.4, linewidth=0.5)
    ax.set_ylabel('Seismic Amplitude');
    ax_twin = ax.twinx()
    if station == "ILL11":
        ax.set_ylim(-1.5, 1.5);
        ax_twin.set_ylim(0, 45)
    elif station == "ILL12":
        ax.set_ylim(-0.5, 0.5);
        ax_twin.set_ylim(0, 20)
    
    ax_twin.plot(times, df['Predicted_Output_Mean'], label='Mean Prediction', color='black', linewidth=0.5)
    ax_twin.fill_between(times, 
                    df['Predicted_Output_Mean'] - df['Predicted_Output_Std'], 
                    df['Predicted_Output_Mean'] + df['Predicted_Output_Std'], 
                    alpha=0.2, label='Standard Deviation', color='red')
    ax_twin.fill_between(times, 
                    df['Predicted_Output_Min'], 
                    df['Predicted_Output_Max'], 
                    alpha=0.3, label='Min-Max', color='blue')
    del df
    # if year == 2019 and station == 'ILL11':
    #     try:
    #         data = load_label2([date_mapping[julday]], station='ILL11', interval_seconds=interval_seconds, time_shift_minutes="average", smoothing=None, divide_by=None)
    #         data['Timestamp'] = data['Timestamp'].apply(lambda x: UTCDateTime(x).matplotlib_date)
    #         ax[1].plot(data['Timestamp'].to_numpy(), data['Fv [kN]'], label='Ground Truth', color='green', linewidth=0.8, alpha=0.7)
    #         del data
    #     except Exception as e:
    #         print(f"Error loading ground truth data for : {e}")

    ax_twin.set_xlim(times[0], times[-1])
    ax_twin.set_ylabel('Predicted Impact Force (kN)')
    ax_twin.set_xlabel('Time')

    for axis in ax:
        axis.xaxis_date()
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        axis.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    fig.savefig(f"{img_dir}/{julday}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.clf()
    plt.close('all')
    
    return None