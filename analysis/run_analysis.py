import os
import sys
import argparse
sys.path.append("..")
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from obspy import UTCDateTime
from tqdm import tqdm
import seaborn as sns
from obspy import read
from sklearn.metrics import mean_squared_error

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

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


def plot_grouped_bar_with_error(data, x, y, hue, hue_order, ax,
                                palette="viridis", bar_width=0.35):
    """
    Plot grouped bar chart with mean and standard error using Matplotlib.

    Parameters:
    - data: pd.DataFrame
    - x: str, column for x-axis categories (e.g., "Interval")
    - y: str, column for y-axis values (e.g., "MSE_ts")
    - hue: str, column to split bars within each x (e.g., "Model")
    - hue_order: list of str, order of hue categories (e.g., ["LSTM", "xLSTM"])
    - ax: matplotlib.axes.Axes, axis to draw the plot on
    - palette: str or colormap, default "viridis"
    - bar_width: float, width of each bar
    """

    # Group data and compute mean and SE
    summary_df = data.groupby([x, hue]).agg(
        mean_y=(y, 'mean'),
        se_y=(y, lambda v: v.std(ddof=1) / np.sqrt(len(v)))
    ).reset_index()

    x_labels = summary_df[x].unique()
    x_pos = np.arange(len(x_labels))
    cmap = ['yellowgreen', 'turquoise']

    for i, hue_val in enumerate(hue_order):
        subset = summary_df[summary_df[hue] == hue_val]
        offset = (i - len(hue_order) / 2) * bar_width + bar_width / 2
        xpos = x_pos + offset

        bars = ax.bar(
            xpos,
            subset['mean_y'],
            yerr=subset['se_y'],
            width=bar_width,
            label=hue_val,
            color=cmap[i],
            capsize=5,
            edgecolor='black'
        )

        # Annotate bars with height
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=hue)
    ax.set_title(f"Mean {y} by {x} and {hue}")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    return None    

def make_evaluation_plots(data, intervals, hue, base_dir, output_file_name):
    # data = data[(data['Test'] != 182)]
    for interval in intervals:
        fig, ax = plt.subplots(1,4, figsize=(14, 3.5))
        plot_data = data[data['Interval'] == interval]
        
        plot_grouped_bar_with_error(data = plot_data, x="Test", y="MSE_ts", hue=hue, palette="viridis", ax=ax[0], hue_order=["LSTM", "xLSTM"])
        plot_grouped_bar_with_error(data = plot_data, x="Test", y="R2_ts", hue=hue, palette="viridis", ax=ax[1], hue_order=["LSTM", "xLSTM"])
        plot_grouped_bar_with_error(data = plot_data, x="Test", y='MAE_ts', hue=hue, palette="viridis", ax=ax[2], hue_order=["LSTM", "xLSTM"])
        plot_grouped_bar_with_error(data = plot_data, x="Test", y='PearsonR_ts', hue=hue, palette="viridis", ax=ax[3], hue_order=["LSTM", "xLSTM"])

        fig.tight_layout()
        fig.savefig(f"{base_dir}/{output_file_name}_{interval}_ts.png", dpi=300)
        plt.close(fig=fig)

        fig, ax = plt.subplots(1,4, figsize=(14, 3.5))
        plot_grouped_bar_with_error(data = plot_data, x="Test", y="MSE_0", hue=hue, palette="viridis", ax=ax[0], hue_order=["LSTM", "xLSTM"])
        plot_grouped_bar_with_error(data = plot_data, x="Test", y="R2_0", hue=hue, palette="viridis", ax=ax[1], hue_order=["LSTM", "xLSTM"])
        plot_grouped_bar_with_error(data = plot_data, x="Test", y='MAE_0', hue=hue, palette="viridis", ax=ax[2], hue_order=["LSTM", "xLSTM"])
        plot_grouped_bar_with_error(data = plot_data, x="Test", y='PearsonR_0', hue=hue, palette="viridis", ax=ax[3], hue_order=["LSTM", "xLSTM"])

        fig.tight_layout()
        fig.savefig(f"{base_dir}/{output_file_name}_{interval}_0.png", dpi=300)
        plt.close(fig=fig)

    # fig, ax = plt.subplots(1, 4, figsize=(14,3.5))
    # sns.barplot(data= data, x='Interval', y='MSE_ts', hue=hue, palette="viridis", ax=ax[0], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    # sns.barplot(data= data, x='Interval', y='R2_ts', hue=hue, palette="viridis", ax=ax[1], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    # sns.barplot(data= data, x='Interval', y='MAE_ts', hue=hue, palette="viridis", ax=ax[2], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    # sns.barplot(data= data, x='Interval', y='PearsonR_ts', hue=hue, palette="viridis", ax=ax[3], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    fig, ax = plt.subplots(1,1, figsize=(5.5,3.5))
    plot_grouped_bar_with_error(data= data, x='Interval', y='MSE_ts', hue=hue, palette="viridis", ax=ax, hue_order=["LSTM", "xLSTM"])
    ax.set_xlabel("Interval (s)")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Comparison of MSE by Interval and Model")
    ax.set_ylim(bottom=0)
    # # ax[3].set_ylim(0, 100)
    # # ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{base_dir}/{output_file_name}_ts.png", dpi=300)
    # plt.close()

    # fig, ax = plt.subplots(1, 4, figsize=(14,3.5))
    # sns.barplot(data= data, x='Interval', y='MSE_0', hue=hue, palette="viridis", ax=ax[0], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    # sns.barplot(data= data, x='Interval', y='R2_0', hue=hue, palette="viridis", ax=ax[1], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    # sns.barplot(data= data, x='Interval', y='MAE_0', hue=hue, palette="viridis", ax=ax[2], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    # sns.barplot(data= data, x='Interval', y='PearsonR_0', hue=hue, palette="viridis", ax=ax[3], errorbar = 'se', hue_order=["LSTM", "xLSTM"])
    fig, ax = plt.subplots(1,1, figsize=(5.5,3.5))
    plot_grouped_bar_with_error(data= data, x='Interval', y='MSE_0', hue=hue, palette="viridis", ax=ax, hue_order=["LSTM", "xLSTM"])
    ax.set_xlabel("Interval (s)")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Comparison of MSE by Interval and Model")
    ax.set_ylim(bottom=0)
    # # ax[3].set_ylim(0, 100)
    # # ax[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{base_dir}/{output_file_name}_0.png", dpi=300)
    # plt.close()

    # fig, ax = plt.subplots(1, 1)
    # sns.barplot(data= data, x='Interval', y='FT_RMSE', hue="Model", palette="viridis", ax=ax, errorbar='sd')
    # fig.tight_layout()
    # fig.savefig(f"{base_dir}/FT_RMSE.png", dpi=300)
    # plt.close()

    fig, ax = plt.subplots()
    data["Time_To_Train"] = pd.to_timedelta(data["Time_To_Train"]).dt.total_seconds()
    plot_grouped_bar_with_error(data=data, x="Interval", y="Time_To_Train", hue=hue, palette="viridis", ax=ax, hue_order=["LSTM", "xLSTM"])
    ax.set_xlabel("Interval (s)")
    ax.set_ylabel("Mean Training Time (seconds)")
    ax.set_title("Comparison of Training Time by Interval and Model")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(f"{base_dir}/TimetoTrain.png", dpi=300)
    plt.close()

    return None

def move_plots(df, model_types, configs, time_intervals, base_dir):
    mapping = {161 : 1, 172 : 2, 196 : 3, 207 : 4, 223 : 5, 232 : 6}
    for model_type in tqdm(model_types, desc="Model Progress"):
        for config in configs:
            for interval in time_intervals:
                temp = df[(df['Model'] == model_type) & (df['Config'] == config) & (df['Interval'] == interval)]
                for idx, row in tqdm(temp.iterrows(), desc=f"Moving Plots ({model_type}, {config}, {interval})"):
                    # Move best models
                    # if row['Test'] == 232:
                    from_path = f"{base_dir}/model/{config}/{interval}/t{row['Test']}_v{row['Val']}_{interval}_{model_type}_model.pt"
                    to_dir = f"{base_dir}/best_models/{mapping[row['Test']]}"
                    os.makedirs(to_dir, exist_ok=True)
                    with open(f"{to_dir}/best_models.txt", "a") as f:
                        f.write(f"{config} {interval} {model_type} {row['Test']} {row['Val']}\n")
                    to_path = f"{to_dir}/{interval}_{model_type}.pt"
                    shutil.copy(from_path, to_path)

def check_velocity_estimates(best_comb, task_dir):
    true_vel = pd.read_csv("../label/DF_characteristics.csv")
    true_vel['Event_Date'] = true_vel['Event_Date'].apply(lambda x: UTCDateTime(x).strftime('%Y-%m-%d'))
    true_vel = true_vel[true_vel['Year'] == 2019]
    print(true_vel)
    # true_vel = true_vel[true_vel['Event_Date'] < UTCDateTime('2020-01-01')]
    # true_vel['Julday'] = true_vel['Event_Date'].apply(lambda x: UTCDateTime(x).julday)
    check_list = []
    for i in true_vel['Julday']:
        if str(i) not in check_list:
            check_list.append(str(i))
        else:
            # print("Duplicate Julian day found:", i)
            check_list.remove(str(i))
            check_list.append(f"{i}_1")
            check_list.append(f"{i}_2")
    true_vel['Test'] = check_list

    peak_times = pd.read_csv("../data_preprocessing/peak_times.csv", index_col=0)
    peak_times['Seismic_Peak_Time'] = peak_times['Seismic_Peak_Time'].apply(lambda x: UTCDateTime(x))   

    time_window = pd.read_csv("../label/correct_metrics_time_window.csv", index_col=False)
    time_window['Start_Time'] = time_window['Start_Time'].apply(lambda x: UTCDateTime(x))
    time_window['End_Time'] = time_window['End_Time'].apply(lambda x: UTCDateTime(x))
    time_window = time_window[time_window['Start_Time'] < UTCDateTime(year=2020, julday = 1)]

    output_df = pd.DataFrame(columns = ['Model', 'Interval', "Test", 'Start_Time', "Time_Diff", "Pred_Velocity"])
    for time_idx, time_row in time_window.iterrows():
    # Get the start and end times for the current time window
        start_time = UTCDateTime(time_row['Start_Time'])
        end_time = UTCDateTime(time_row['End_Time'])
        julday = start_time.julday
        st = read(f"../data_srr/Illgraben/2019/ILL11/EHZ/9S.ILL11.EHZ.2019.{julday}.mseed")
        st.trim(starttime=start_time, endtime=end_time)
        # idxs = np.argpartition(st[0].data, -5)[-5:]
        # time_diff = [idx / st[0].stats.sampling_rate for idx in idxs]
        # max_idx = idxs[np.argmin(time_diff)]
        # peak_time = start_time + max_idx / st[0].stats.sampling_rate
        peak_time = peak_times.iloc[time_idx,0]
        temp = best_comb[best_comb['Test'] == julday].reset_index(drop=True)
        for idx, row in temp.iterrows():
            data = pd.read_csv(f"{task_dir}/output_df/{row['Config']}/{row['Interval']}/{row['Model']}_t{row['Test']}_v{row['Val']}.csv", index_col=False)
            data['Timestamps'] = data['Timestamps'].apply(lambda x: UTCDateTime(x))

            data = data[data['Timestamps'].between(start_time, end_time)].reset_index(drop=True)
            data['mpltime'] = data['Timestamps'].apply(lambda x: x.matplotlib_date)
            pred_peak_time = data.iloc[np.argmax(data['Predicted_Output'])].Timestamps
            pred_velocity = 300 / (pred_peak_time - peak_time)
            print(f"Peak Time: {peak_time}, Predicted Peak Time: {pred_peak_time}")
            print(f"Peak Time Difference: {pred_peak_time - peak_time} seconds")
            print(f"Estimated Velocity: {300 / (pred_peak_time - peak_time):.4f} m/s")
            if (julday == 161) & (time_idx == 0):
                # print(1)
                test_julday = f"{julday}_1"
            elif (julday == 161) & (time_idx == 1):
                # print(2)
                test_julday = f"{julday}_2"
            else:
                # print(3)
                test_julday = str(julday)
            output_df.loc[len(output_df)] = [row['Model'], 
                                            row['Interval'], 
                                            test_julday, 
                                            start_time, 
                                            pred_peak_time - peak_time, 
                                            np.round(pred_velocity, 2)]
            fig, ax = plt.subplots()
            ax.plot(st[0].times('matplotlib'), st[0].data * 10**3, color="black", label= "ILL11", alpha=0.8)
            ax.set_ylabel(r"Amplitude (mm/s)");
            ax.set_ylim(-1.5, 1.5);
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            ax.set_xlim(st[0].times('matplotlib')[0], st[0].times('matplotlib')[-1])

            ax.axvline(x=peak_time.matplotlib_date, color='blue', linestyle='--', label='Peak Time')
            ax.axvline(x=pred_peak_time.matplotlib_date, color='red', linestyle='--', label='Predicted Peak Time')

            axtwin = ax.twinx()
            axtwin.plot(data['mpltime'], data['Output'], color="blue", label= "True", alpha=0.6)
            axtwin.plot(data['mpltime'], data['Predicted_Output'], color="red", label= "Predicted", alpha=0.8)

            axtwin.set_ylim(bottom=0)
            fig.tight_layout()
            fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), bbox_transform=ax.transAxes)
            os.makedirs(f"{task_dir}/figures/{row['Config']}/{row['Interval']}/", exist_ok=True)
            fig.savefig(f"{task_dir}/figures/{row['Config']}/{row['Interval']}/{row['Model']}_t{row['Test']}_{time_idx}.png", dpi=300)
            plt.close(fig)

    output_df = output_df.merge(true_vel, how='left', left_on='Test', right_on='Test')
    output_df = output_df[output_df['Test'] != "161_2"]
    output_df['Difference'] = output_df['Pred_Velocity'] - output_df['Velocity']
    fig, ax = plt.subplots()
    sns.barplot(data= output_df, x='Interval', y='Difference', hue='Model', palette="viridis", errorbar = 'se', hue_order=["LSTM", "xLSTM"], ax=ax)
    fig.tight_layout()
    fig.savefig(f"{task_dir}/Velocity_Estimates.png", dpi=300)
    plt.close()
    return None

def calc_ref_scores(base_dir, output_dir):
    eval_out = pd.read_csv(f"{output_dir}/evaluation_output.txt")
    eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt")
    eval_out.sort_values(by=["Test", "Interval", "Val"], inplace=True)
    eval_out_cons.sort_values(by=["Test", "Interval", "Val"], inplace=True)

    old_test, old_interval = 1, 1
    list1, list2, list3, list4 = [], [], [], []

    constraint_df = pd.read_csv(f"../label/correct_metrics_time_window.csv", index_col=False)

    for idx, row in tqdm(eval_out.iterrows(), total=len(eval_out)):
        julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
        date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
        row_cons = eval_out_cons.loc[idx]
        test = row['Test']
        val = row['Val']
        interval = row['Interval']
        model = row['Model']

        if test == 161:
            window_df = constraint_df.iloc[:2]
            # print(window_df)
            window_start_1, window_end_1 = UTCDateTime(window_df['Start_Time'].iloc[0]), UTCDateTime(window_df['End_Time'].iloc[0])
            window_start_2, window_end_2 = UTCDateTime(window_df['Start_Time'].iloc[1]), UTCDateTime(window_df['End_Time'].iloc[1])
        else:
            window_df = constraint_df.iloc[julday_list.index(test) + 1]
            # print(window_df)
            window_start, window_end = UTCDateTime(window_df['Start_Time']), UTCDateTime(window_df['End_Time'])

        if (test == old_test) and (interval == old_interval):
            pass
        else:
            print(f"Loading Label {test} {interval}")
            julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
            date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
            label = load_label([date_list.pop(julday_list.index(test))], "ILL11", interval, 'dynamic', trim=False, smoothing=None)
            julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
            date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
            label_zero = load_label([date_list.pop(julday_list.index(test))], "ILL11", interval, 0, trim=False, smoothing=None)
            label['Timestamp'] = label['Timestamp'].apply(lambda x: UTCDateTime(x))
            label_zero['Timestamp'] = label_zero['Timestamp'].apply(lambda x: UTCDateTime(x))
            old_test = test
            old_interval = interval
        output_df = pd.read_csv(f"{base_dir}/output_df/{row['Config']}/{row['Interval']}/{model}_t{test}_v{val}.csv", index_col=None)
        output_df['Timestamps'] = output_df["Timestamps"].apply(lambda x: UTCDateTime(x))
        assert len(output_df) == len(label), f"Length mismatch {len(output_df)} -|- {len(label)}"
        # print("Calculating MSE")
        list1.append(np.round(mean_squared_error(label['Fv [kN]'].to_numpy(), output_df['Predicted_Output'].to_numpy()),4))
        list2.append(np.round(mean_squared_error(label_zero['Fv [kN]'].to_numpy(), output_df['Predicted_Output'].to_numpy()),4))

        # print("Calculating constrained MSE")
        if test == 161:
            label_temp = label[label['Timestamp'].between(window_start_1, window_end_1)]
            label_trim = pd.concat([label_temp, label[label['Timestamp'].between(window_start_2, window_end_2)]])
            zero_label_temp = label_zero[label_zero['Timestamp'].between(window_start_1, window_end_1)]
            label_zero_trim = pd.concat([zero_label_temp, label_zero[label_zero['Timestamp'].between(window_start_2, window_end_2)]])
            output_df_temp = output_df[output_df['Timestamps'].between(window_start_1, window_end_1)]
            output_df_trim = pd.concat([output_df_temp, output_df[output_df['Timestamps'].between(window_start_2, window_end_2)]])
        else:
            label_trim = label[label['Timestamp'].between(window_start, window_end)]
            label_zero_trim = label_zero[label_zero['Timestamp'].between(window_start, window_end)]
            output_df_trim = output_df[output_df['Timestamps'].between(window_start, window_end)]

        list3.append(np.round(mean_squared_error(label_trim['Fv [kN]'].to_numpy(), output_df_trim['Predicted_Output'].to_numpy()),4))
        list4.append(np.round(mean_squared_error(label_zero_trim['Fv [kN]'].to_numpy(), output_df_trim['Predicted_Output'].to_numpy()),4))

    eval_out['ref_MSE_ts'] = list1
    eval_out['ref_MSE_0'] = list2
    eval_out_cons['ref_MSE_ts'] = list3
    eval_out_cons['ref_MSE_0'] = list4

    eval_out.to_csv(f"{output_dir}/evaluation_output.txt", index=False)
    eval_out_cons.to_csv(f"{output_dir}/evaluation_output_constrained.txt", index=False)
    
    return None

def main(task:str, model_types:list[str], configs:list[str], time_shift:int=10):
    smoothing = 30
    time_intervals = [5, 15, 30]
    if task.startswith("comparison_baseline"):
        hue = 'Model'
    else:
        hue = 'Config'
    julday_list = [161, 172, 196, 207, 223, 232]
    date_list = ["2019-06-10", "2019-06-21", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    base_dir = f"../new_final_version/{task}/{time_shift}_{smoothing}"
    output_dir = f"../new_final_version/{task}/{time_shift}_{smoothing}/model_evaluation"
    # model_output_dir = {c : f"../{task}/{time_shift}_{smoothing}/output_df/{c}" for c in configs}
    data = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt", index_col=False)
    data = data[(data['Test'] != 182) & (data['Test'] != 183)]
    
    # if not os.path.exists(f"{base_dir}/fourier_transform/"):
    #     make_fourier_transform_plots_and_metrics(data, base_dir, model_output_dir)
    # else:
    #     pass
    
    calc_ref_scores(base_dir, output_dir)

    if os.path.exists(f"{output_dir}/best_combinations.csv"):
        pass
    else:
        print("\tSelecting Best Combinations")
        data = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt", index_col=False)
        best_combinations_df = pd.DataFrame(columns = data.columns.values)
        for model_type in tqdm(model_types, desc="Model Progress"):
            for config in tqdm(configs, desc="Config Progress"):
                temp_data = data[(data["Model"] == model_type) & (data['Config'] == config)]
                for interval in tqdm(time_intervals, desc=f"Interval Progress ({model_type}, {config})"):
                    for test_julday in tqdm([161, 172, 196, 207, 223, 232], desc=f"Julday Progress"):
                            temp = temp_data[(temp_data["Test"] == test_julday) & (temp_data["Interval"] == interval)]
                            temp.reset_index(inplace=True, drop=True)
                            temp = temp.iloc[temp.nsmallest(1, "MSE_ts").index]
                            best_combinations_df.loc[len(best_combinations_df)] = temp.values[0]
        best_combinations_df.to_csv(f"{output_dir}/best_combinations.csv", index=False)

    
    print("\tMaking Plots and Moving Images")
    fig, ax = plt.subplots()
    plot_grouped_bar_with_error(data= best_combinations_df, x='Interval', y='ref_MSE_ts', hue='Model', palette="viridis", ax=ax, hue_order=['LSTM', 'xLSTM'], bar_width=0.25)
    fig.savefig(f"{base_dir}/ref_MSE_ts.png", dpi=300)
    plt.close(fig=fig)
    fig, ax = plt.subplots()
    plot_grouped_bar_with_error(data= best_combinations_df, x='Interval', y='ref_MSE_0', hue='Model', palette="viridis", ax=ax, hue_order=['LSTM', 'xLSTM'], bar_width=0.25)
    fig.savefig(f"{base_dir}/ref_MSE_0.png", dpi=300)
    plt.close(fig=fig)
    # data = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt", index_col=False)
    # make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Comparison_Plot")
    data = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
    make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Best_Comparison_Plot")

    move_plots(data, model_types, configs, time_intervals, base_dir)

    print("Making distribution plots!")
    if os.path.exists(f"{base_dir}/dist_plot"):
        pass
    else:
        if 'comparison_baseline' in task:
            config = "default"
            for option in model_types:
                # option = "xLSTM"
                for interval in time_intervals:
                    main2(option, interval, f"new_final_version/{task}/{time_shift}_{smoothing}", config)
                    plt.close()
        else:
            pass

    # print("Checking Velocity estimates")
    # data = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
    # check_velocity_estimates(data, base_dir)

    print("\tMaking Explaination Plots")
    data = pd.read_csv(f"../new_final_version/{task}/{time_shift}_{smoothing}/model_evaluation/best_combinations.csv", index_col=False)
    zoom_df = pd.read_csv("../label/correct_metrics_time_window.csv", index_col=False)
    zoom_df = zoom_df.iloc[[0,2,5,6,7,8],:].reset_index(drop=True)

    output_file_dir = f"../new_final_version/{task}/{time_shift}_{smoothing}/output_df/default"

    i = 0
    julday_list = [161, 172, 196, 207, 223, 232]
    for julday in tqdm(julday_list, desc= "Julday Progress"):
        for interval in time_intervals:
            temp = data[(data['Test'] == julday) & (data['Interval'] == interval)]
            fig, ax = plt.subplots(2, 2, figsize=(12.0, 6.0), sharey=True)
            date_list = ["2019-06-10", "2019-06-21", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
            date = date_list.pop(julday_list.index(julday))
            target_output = load_label([date], "ILL11", interval, 0, trim=False, smoothing=smoothing)
            target_times = [UTCDateTime(i).matplotlib_date for i in target_output['Timestamp'].to_numpy()]
            for idx, row in temp.iterrows():
                interval = row['Interval']
                model_type = row['Model']
                test = row['Test']
                val = row['Val']
                st = load_seismic_data(test, 'ILL11', trim=False)
                file = pd.read_csv(f"{output_file_dir}/{interval}/{model_type}_t{test}_v{val}.csv", index_col=False)
                times = [UTCDateTime(i).matplotlib_date for i in file['Timestamps'].to_numpy()]
                # target_output = file['Output'].to_numpy()
                predicted_output = file['Predicted_Output'].to_numpy()
                
                if model_type == 'xLSTM':
                    ax[0,0].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[0,0].set_ylabel(r"Amplitude (mm/s)");
                    ax[0,0].set_ylim(-1.5, 1.5);
                    ax1 = ax[0,0].twinx()
                    ax1.plot(target_times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    # ax1.fill_between(target_times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
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
                    ax2.plot(target_times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    # ax2.fill_between(target_times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
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
                    ax3.plot(target_times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    # ax3.fill_between(target_times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
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
                    ax4.plot(target_times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    mean = target_output['Fv [kN]'].to_numpy()
                    std = target_output['Fv std'].to_numpy()
                    # ax4.fill_between(target_times, mean - std, mean + std, color='r', alpha=0.4, label="Std Dev")
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
    