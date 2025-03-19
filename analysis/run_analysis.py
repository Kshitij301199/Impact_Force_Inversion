import os
import sys
import argparse
sys.path.append("..")
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
from tqdm import tqdm
import seaborn as sns

plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 6,        # Set legend font size
    'figure.figsize': (5.5, 3.5) # Set figure size in inches
})
sns.set_context("notebook", font_scale=1)  # Ensures Seaborn uses updated fonts/sizes

from functions.plot_distributions import main as main2
from data_processing.read_data import load_label, load_seismic_data

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    return np.round(np.mean((y_true - y_pred) ** 2), 2)

# Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    return np.round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 2)

# Symmetric Mean Absolute Percentage Error (sMAPE)
def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
    Returns:
        float: sMAPE value.
    """
    return np.round(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100, 2)

# Pearson's Correlation Coefficient (PCC)
def pcc(y_true, y_pred):
    """
    Calculate Pearson's Correlation Coefficient (PCC).
    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
    Returns:
        float: PCC value.
    """
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    return np.round(numerator / denominator, 2)

def make_zero_shift_plots(data:pd.DataFrame, base_dir:str, model_output_dir:str):
    print("Making zero shift plots!\n")
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="DF Progess"):
        julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
        date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
        
        test_julday, val_julday, interval, model_type, config = row["Test"], row['Val'], row['Interval'], row['Model'], row['Config']
        model_output = pd.read_csv(f"{model_output_dir[config]}/{interval}/{model_type}_t{test_julday}_v{val_julday}.csv", index_col=False)
        zero_label = load_label([date_list.pop(julday_list.index(test_julday))], "ILL11", interval, 0)
        zero_label['Timestamp'] = zero_label['Timestamp'].apply(UTCDateTime)
        plot_df = pd.merge(model_output, zero_label, how="left", left_on="Timestamps", right_on="Timestamp")

        times = [UTCDateTime(i).matplotlib_date for i in plot_df['Timestamps'].to_numpy()]
        target_output = plot_df['Fv [kN]'].to_numpy()
        predicted_output = plot_df['Predicted_Output'].to_numpy()
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        ax.plot(times, target_output, label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
        ax.plot(times, predicted_output, label="Model Prediction", alpha=0.6, color='b',linewidth=1)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        ax.set_xlim(times[0], times[-1]);
        ax.set_ylabel("Normal Force [kN]");
        ax.set_ylim(0,350);
        ax.legend(loc='best')
        save_dir = f"{base_dir}/new_images/{model_type}/{config}/{interval}"
        os.makedirs(save_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/{test_julday}_{val_julday}.png", dpi=300)
        plt.close()
    return None

def make_fourier_transform_plots_and_metrics(data, base_dir, model_output_dir):
    print("Making fourier transform calculations and plots!\n")
    ff_rmse, ff_smape = [], []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="DF Progess FT"):
        test_julday, val_julday, interval, model_type, config = row["Test"], row['Val'], row['Interval'], row['Model'], row['Config']
        input_dir = model_output_dir[config]
        output_dir = f"{base_dir}/fourier_transform/{model_type.lower()}/{config}/{interval}"
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(f"{input_dir}/{interval}/{model_type}_t{test_julday}_v{val_julday}.csv", index_col=False)
        # Convert timestamps to datetime and compute time intervals
        df["Timestamps"] = pd.to_datetime(df["Timestamps"])
        time_diffs = (df["Timestamps"] - df["Timestamps"].iloc[0]).dt.total_seconds()

        # Sampling interval (assumes uniform spacing)
        dt = np.mean(np.diff(time_diffs))
        Fs = 1 / dt  # Sampling frequency
        # Perform FFT on both Output and Predicted_Output
        fft_output = np.fft.fft(df["Output"])
        fft_predicted = np.fft.fft(df["Predicted_Output"])

        # Compute frequencies
        frequencies = np.fft.fftfreq(len(df["Output"]), d=dt)
        # Keep only the positive half of the spectrum
        positive_freqs = frequencies[:len(frequencies)//2]
        magnitude_output = np.abs(fft_output[:len(frequencies)//2])
        magnitude_predicted = np.abs(fft_predicted[:len(frequencies)//2])
        ff_rmse.append(rmse(magnitude_output, magnitude_predicted))
        ff_smape.append(smape(magnitude_output, magnitude_predicted))
        # Plot
        fig, ax = plt.subplots()
        ax.plot(positive_freqs, magnitude_output, label="Output", marker='.', ms= 1, linewidth=1)
        ax.plot(positive_freqs, magnitude_predicted, label="Predicted Output", marker='.', ms= 1, linewidth=1)
        ax.set_title("Fourier Transform of Output and Predicted Output")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_xscale('log')
        ax.legend()
        ax.grid()
        fig.savefig(f"{output_dir}/t{test_julday}_v{val_julday}.png", dpi=300)
        plt.close()
    
    data['FT_RMSE'] = np.round(np.sqrt(ff_rmse), 2)
    data['FT_SMAPE'] = np.round(np.sqrt(ff_smape), 2)
    data.to_csv(f"{base_dir}/model_evaluation/evaluation_output_ft.csv", index=False)
    return None

def main(task:str, model_types:list[str], configs:list[str], time_shift:int=10):
    time_intervals = [15, 30, 60]
    if task == "comparison_baseline":
        hue = 'Model'
    else:
        hue = 'Config'
    julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
    date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    base_dir = f"../{task}/{time_shift}"
    output_dir = f"../{task}/{time_shift}/model_evaluation"
    model_output_dir = {c : f"../{task}/{time_shift}/output_df/{c}" for c in configs}
    data = pd.read_csv(f"{output_dir}/evaluation_output.txt", index_col=False)
    
    if not os.path.exists(f"{base_dir}/fourier_transform/"):
        make_fourier_transform_plots_and_metrics(data, base_dir, model_output_dir)
    else:
        pass
    
    if os.path.exists(f"{output_dir}/best_combinations.csv"):
        pass
    else:
        print("\tSelecting Best Combinations")
        data = pd.read_csv(f"{output_dir}/evaluation_output_ft.csv", index_col=False)
        best_combinations_df = pd.DataFrame(columns = data.columns.values)
        for model_type in tqdm(model_types, desc="Model Progress"):
            for config in tqdm(configs, desc="Config Progress"):
                temp_data = data[(data["Model"] == model_type) & (data['Config'] == config)]
                for interval in tqdm(time_intervals, desc=f"Interval Progress ({model_type}, {config})"):
                    for test_julday in tqdm([161, 172, 182, 183, 196, 207, 223, 232], desc=f"Julday Progress"):
                            temp = temp_data[(temp_data["Test"] == test_julday) & (temp_data["Interval"] == interval)]
                            temp.reset_index(inplace=True, drop=True)
                            temp = temp.iloc[temp.nsmallest(3, 'SMAPE_0').nlargest(2, "PCC_0").nsmallest(1, "MSE_0").index]
                            best_combinations_df.loc[len(best_combinations_df)] = temp.values[0]
        best_combinations_df.to_csv(f"{output_dir}/best_combinations.csv", index=False)
    
    if os.path.exists(f"{output_dir}/corrected_best_combinations.csv"):
        pass
    else:
        print("\tCorrecting Metrics")
        julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
        correct_window_df = pd.read_csv(f"../label/correct_metrics_time_window.csv", index_col=False)
        best_combinations_df = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
        corrected_df = pd.DataFrame(columns=best_combinations_df.columns.values)
        for model_type in tqdm(model_types, desc="Model Progress"):
            for config in tqdm(configs, desc="Config Progress"):
                for interval in tqdm(time_intervals, desc=f"Time Interval Progress ({model_type}, {config})"):
                    temp = best_combinations_df[(best_combinations_df['Config'] == config) & (best_combinations_df['Interval'] == interval) & (best_combinations_df['Model'] == model_type)]
                    for idx, row in tqdm(temp.iterrows(), total=len(temp), desc=f"Processing Rows ({config}, {interval})"):
                        output_df = pd.read_csv(f"{model_output_dir[config]}/{interval}/{model_type}_t{row['Test']}_v{row['Val']}.csv")
                        window_df = correct_window_df.iloc[julday_list.index(row['Test'])]
                        window_start, window_end = UTCDateTime(window_df['Start_Time']), UTCDateTime(window_df['End_Time'])

                        date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
                        target_df_0 = load_label([date_list.pop(julday_list.index(row['Test']))], "ILL11", interval, 0)
                        date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
                        target_df_10 = load_label([date_list.pop(julday_list.index(row['Test']))], "ILL11", interval, 10)
                        
                        target_df_0['Timestamp'] = target_df_0['Timestamp'].apply(UTCDateTime)
                        target_df_10['Timestamp'] = target_df_10['Timestamp'].apply(UTCDateTime)
                        merge_df_0 = pd.merge(output_df, target_df_0, how="inner", left_on="Timestamps", right_on="Timestamp")
                        merge_df_10 = pd.merge(output_df, target_df_10, how="inner", left_on="Timestamps", right_on="Timestamp")

                        cut_merge_df_0 = merge_df_0[merge_df_0['Timestamps'].between(window_start, window_end)]
                        cut_merge_df_10 = merge_df_10[merge_df_10['Timestamps'].between(window_start, window_end)]
                        target_0 = cut_merge_df_0['Fv [kN]'].to_numpy()
                        prediction_0 = cut_merge_df_0['Predicted_Output'].to_numpy()
                        target_10 = cut_merge_df_10['Fv [kN]'].to_numpy()
                        prediction_10 = cut_merge_df_10['Predicted_Output'].to_numpy()
                        # Sampling interval (assumes uniform spacing)
                        corrected_df.loc[len(corrected_df)] =  [model_type, config, row['Time_To_Train'], row['Test'], row['Val'], interval, 
                                                                mse(target_10, prediction_10), smape(target_10, prediction_10), pcc(target_10, prediction_10), 
                                                                row["DTW_Dist"], 
                                                                mse(target_0, prediction_0), smape(target_0, prediction_0), pcc(target_0, prediction_0), 
                                                                row['FT_RMSE'], row['FT_SMAPE']]
        corrected_df.to_csv(f"{output_dir}/corrected_best_combinations.csv", index=False)
    
    print("\tMaking Plots and Moving Images")
    data = pd.read_csv(f"{output_dir}/corrected_best_combinations.csv", index_col=False)
    # data = data[(data['Test'] != 172) & (data['Test'] != 182)]

    fig, ax = plt.subplots(1, 2, figsize=(10,3.5))
    sns.barplot(data= data, x='Interval', y='MSE_ts', hue=hue, palette="viridis", ax=ax[0], errorbar = 'se')
    # sns.barplot(data= data, x='Interval', y='SMAPE_10', hue=hue, palette="viridis", ax=ax[1], errorbar = 'se')
    sns.barplot(data= data, x='Interval', y='PCC_ts', hue=hue, palette="viridis", ax=ax[1], errorbar = 'se')
    ax[0].set_ylim(bottom=0)
    # ax[1].set_ylim(0, 100)
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{base_dir}/Comparison_Plot_10.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10,3.5))
    sns.barplot(data= data, x='Interval', y='MSE_0', hue=hue, palette="viridis", ax=ax[0], errorbar = 'se')
    # sns.barplot(data= data, x='Interval', y='SMAPE_0', hue=hue, palette="viridis", ax=ax[1], errorbar = 'se')
    sns.barplot(data= data, x='Interval', y='PCC_0', hue=hue, palette="viridis", ax=ax[1], errorbar = 'se')
    ax[0].set_ylim(bottom=0)
    # ax[1].set_ylim(0, 100)
    ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{base_dir}/Comparison_Plot_0.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1)
    sns.barplot(data= data, x='Interval', y='FT_RMSE', hue="Model", palette="viridis", ax=ax, errorbar='sd')
    fig.tight_layout()
    fig.savefig(f"{base_dir}/FT_RMSE.png", dpi=300)
    plt.close()

    df = pd.read_csv(f"{output_dir}/corrected_best_combinations.csv", index_col=False)
    for model_type in tqdm(model_types, desc="Model Progress"):
        for config in configs:
            for interval in time_intervals:
                temp = df[(df['Model'] == model_type) & (df['Config'] == config) & (df['Interval'] == interval)]
                for idx, row in temp.iterrows():
                    from_path = f"{base_dir}/test_results/{model_type.lower()}/{config}/{interval}/{row['Test']}_{row['Val']}_{interval}.png"
                    to_dir = f"{base_dir}/best_images/plot/{model_type.lower()}/{config}/{interval}"
                    os.makedirs(to_dir, exist_ok=True)
                    to_path = f"{to_dir}/{row['Test']}_{row['Val']}_{interval}.png"
                    shutil.copy(from_path, to_path)
                    if model_type == 'LSTM':
                        from_path = f"{base_dir}/fourier_transform/{model_type}/{config}/{interval}/t{row['Test']}_v{row['Val']}.png"
                    else:
                        from_path = f"{base_dir}/fourier_transform/{model_type}/{config}/{interval}/t{row['Test']}_v{row['Val']}.png"
                    to_dir = f"{base_dir}/best_images/FT/{model_type.lower()}/{config}/{interval}"
                    os.makedirs(to_dir, exist_ok=True)
                    to_path = f"{to_dir}/{row['Test']}_{row['Val']}_{interval}.png"
                    shutil.copy(from_path, to_path)

    if task == 'comparison_baseline':
        config = "default"
        for option in model_types:
            for interval in time_intervals:
                main2(option, interval, f"{task}/{time_shift}", config)
                plt.close()
    else:
        pass

    fig, ax = plt.subplots()
    df["Time_To_Train"] = pd.to_timedelta(df["Time_To_Train"]).dt.total_seconds()
    sns.barplot(data=df, x="Interval", y="Time_To_Train", hue=hue, palette="viridis", ax=ax, errorbar = 'se')
    ax.set_xlabel("Interval (s)")
    ax.set_ylabel("Mean Training Time (seconds)")
    ax.set_title("Comparison of Training Time by Interval and Model")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(f"{base_dir}/TimetoTrain.png", dpi=300)
    plt.close()

    print("\tMaking Explaination Plots")
    data = pd.read_csv(f"../comparison_baseline/{time_shift}/model_evaluation/best_combinations.csv", index_col=False)
    zoom_df = pd.read_csv("../label/correct_metrics_time_window.csv", index_col=False)

    output_file_dir = f"../comparison_baseline/{time_shift}/output_df/default"
    label_dir_0 = "../label/data_processed_0/ILL11"
    label_dir_10 = "../label/data_processed_0/ILL11"

    i = 0
    julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
    for julday in tqdm(julday_list, desc= "Julday Progress"):
        for interval in time_intervals:
            temp = data[(data['Test'] == julday) & (data['Interval'] == interval)]
            fig, ax = plt.subplots(2, 2, figsize=(12.0, 6.0))
            date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
            date = date_list.pop(julday_list.index(julday))
            target_output = load_label([date], "ILL11", interval, 0)
            for idx, row in temp.iterrows():
                interval = row['Interval']
                model_type = row['Model']
                test = row['Test']
                val = row['Val']
                st = load_seismic_data(test, 'ILL11')
                file = pd.read_csv(f"{output_file_dir}/{interval}/{model_type}_t{test}_v{val}.csv", index_col=False)
                times = [UTCDateTime(i).matplotlib_date for i in file['Timestamps'].to_numpy()]
                # target_output = file['Output'].to_numpy()
                predicted_output = file['Predicted_Output'].to_numpy()
                
                if model_type == 'xLSTM':
                    ax[0,0].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[0,0].set_ylabel(r"Amplitude (mm/s)");
                    ax[0,0].set_ylim(-1.5, 1.5);
                    ax1 = ax[0,0].twinx()
                    ax1.plot(times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    ax1.fill_between(times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
                    ax1.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
                    ax1.xaxis_date()
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
                    ax1.set_xlim(times[0], times[-1]);
                    ax1.set_ylabel("Normal Force [kN]");
                    ax1.set_ylim(bottom=0)
                    ax1.legend(loc='best');

                    ax[1,0].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[1,0].set_ylabel(r"Amplitude (mm/s)");
                    ax[1,0].set_ylim(-1.5, 1.5);
                    ax2 = ax[1,0].twinx()
                    ax2.plot(times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    ax2.fill_between(times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
                    ax2.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
                    ax2.xaxis_date()
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
                    ax2.set_xlim(UTCDateTime(zoom_df.iloc[i,0]).matplotlib_date, UTCDateTime(zoom_df.iloc[i,1]).matplotlib_date);
                    ax2.set_ylabel("Normal Force [kN]");
                    ax2.set_ylim(bottom=0)
                    ax2.legend(loc='best');

                elif model_type == 'LSTM':
                    ax[0,1].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[0,1].set_ylabel(r"Amplitude (mm/s)");
                    ax[0,1].set_ylim(-1.5, 1.5);
                    ax3 = ax[0,1].twinx()
                    ax3.plot(times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    ax3.fill_between(times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
                    ax3.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
                    ax3.xaxis_date()
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
                    ax3.set_xlim(times[0], times[-1]);
                    ax3.set_ylabel("Normal Force [kN]");
                    ax3.set_ylim(bottom=0)
                    ax3.legend(loc='best');

                    ax[1,1].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[1,1].set_ylabel(r"Amplitude (mm/s)");
                    ax[1,1].set_ylim(-1.5, 1.5);
                    ax4 = ax[1,1].twinx()
                    ax4.plot(times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    ax4.fill_between(times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
                    ax4.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
                    ax4.xaxis_date()
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
                    ax4.set_xlim(UTCDateTime(zoom_df.iloc[i,0]).matplotlib_date, UTCDateTime(zoom_df.iloc[i,1]).matplotlib_date);
                    ax4.set_ylabel("Normal Force [kN]");
                    ax4.set_ylim(bottom=0)
                    ax4.legend(loc='best');

            ax[0,0].set_title("xLSTM Model");
            ax[0,1].set_title("LSTM Model");
            fig.suptitle(f"Model comparison, Interval = {interval} seconds, Test Julday = {test}", fontdict={'fontsize':8, 'fontweight':'bold'});
            fig.tight_layout()
            os.makedirs(f"{base_dir}/plots/", exist_ok=True)
            fig.savefig(f"{base_dir}/plots/{date}_{interval}.png", dpi=300)
            plt.close()
        i += 1
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default=10, help= "enter label time shift")
    parser.add_argument("--task", type=str, default="comparison_baseline", help= "name of the task corresponding to parameter directory", required=True)
    parser.add_argument("--config", nargs="+", help="List of configuration values", required=True)
    parser.add_argument("--model_type", nargs="+", help="List of model types", required=True)

    args = parser.parse_args()

    main(args.task, args.model_type, args.config, args.time)






    