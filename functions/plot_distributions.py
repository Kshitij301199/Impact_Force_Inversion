import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 8,        # Set legend font size
    'figure.figsize': (8, 5) # Set figure size in inches
})

def main(model_type, interval, task, config):
    output_dir = f"../{task}/dist_plots/{config}/{model_type}/{interval}/" 
    psd_output_dir = f"../{task}/psd_plots/{config}/{model_type}/{interval}/"
    os.makedirs(output_dir, exist_ok= True)
    os.makedirs(psd_output_dir, exist_ok=True)
    data = pd.read_csv(f"../{task}/model_evaluation/best_combinations.csv", index_col=False)
    df = data[(data['Interval'] == interval) & (data['Model'] == model_type)]
    df.reset_index(drop=True, inplace= True)

    all_preds, all_trues = [], []
    for i, row in df.iterrows():
        preds, trues = [], []
        t, v = int(row['Test']), int(row['Val'])
        temp = pd.read_csv(f"../{task}/output_df/{config}/{interval}/{model_type}_t{t}_v{v}.csv")
        trues.extend(temp['Output'].to_numpy())
        preds.extend(temp['Predicted_Output'].to_numpy())
        all_trues.extend(temp['Output'].to_numpy())
        all_preds.extend(temp['Predicted_Output'].to_numpy())
        
        bins = np.arange(5, 350, 10)

        plt.hist(trues, bins= bins, color= "blue", alpha= 0.6, label='True');
        plt.hist(preds, bins= bins, color= "red", alpha= 0.6, label='Predicted');
        # plt.xscale('log')
        plt.xlabel("Normal Force [kN]")
        plt.ylabel("Frequency")
        plt.title(f"{model_type} {interval} test {t}")
        plt.legend(loc='best')
        plt.savefig(f"{output_dir}/t{t}_v{v}_distplot.png", dpi=300)
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        fs = 1 / interval
        frequencies_true, times_true, Sxx_true = spectrogram(temp['Output'].to_numpy(), fs=fs)
        frequencies_pred, times_pred, Sxx_pred = spectrogram(temp['Predicted_Output'].to_numpy(), fs=fs)

        # Plot it
        im1 = ax[0].pcolormesh(times_true, frequencies_true, 10 * np.log10(Sxx_true), shading='gouraud', vmin=-60, vmax=60)
        ax[0].set_ylabel('Frequency [Hz]')
        ax[0].set_xlabel('Time [sec]')
        # ax[0].set_suptitle('Spectrogram (Power Spectral Density)')
        fig.colorbar(im1, ax=ax[0], label='Power [dB]')

        im2 = ax[1].pcolormesh(times_pred, frequencies_pred, 10 * np.log10(Sxx_pred), shading='gouraud', vmin=-60, vmax=60)
        ax[1].set_ylabel('Frequency [Hz]')
        ax[1].set_xlabel('Time [sec]')
        # ax[1].set_suptitle('Spectrogram (Power Spectral Density)')
        fig.colorbar(im2, ax=ax[1], label='Power [dB]')
        
        fig.tight_layout()
        fig.savefig(f"{psd_output_dir}/t{t}_v{v}_distplot.png", dpi=300)
        plt.close()
    
    bins = np.arange(5, 350, 10)

    plt.hist(all_trues, bins= bins, color= "blue", alpha= 0.6, label='True');
    plt.hist(all_preds, bins= bins, color= "red", alpha= 0.6, label='Predicted');
    # plt.xscale('log')
    plt.xlabel("Normal Force [kN]")
    plt.ylabel("Frequency")
    plt.title(f"{model_type} {interval}")
    plt.legend(loc='best')
    plt.savefig(f"../{task}/dist_plots/{config}/{model_type}_{interval}_distplot.png", dpi=300)
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help= "interval seconds")
    parser.add_argument("--model_type", type=str, default='LSTM', help= "model type")
    parser.add_argument("--task", type=str, default="comparison_baseline", help="input the name of task")
    parser.add_argument("--config", type=str, default="default", help="input name of the config")

    args = parser.parse_args()
    print(f"Running main with {args.interval} {args.model_type} {args.task}")
    main(args.model_type, args.interval, args.task)