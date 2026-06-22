import os
import sys
import json
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
import sys
sys.path.append(str(project_root))
with open(f"{project_root}/config/event_id_map.json", "r") as file:
    time_config = json.load(file)
import argparse
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

# from functions.evaluation.plot_distributions import main as main2
from functions.data_processing.read_data import load_label, load_seismic_data

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    return np.round(np.mean((y_true - y_pred) ** 2), 2)

def plot_grouped_bar_with_error(data, x, y, hue, hue_order, ax,
                                palette="viridis", bar_width=0.35, ylim=None, err_type='se'):
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
        se_y=(y, lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        min_y=(y, 'min'),
        max_y=(y, 'max')
    ).reset_index()

    x_labels = summary_df[x].unique()
    x_pos = np.arange(len(x_labels))
    cmap = ['yellowgreen', 'turquoise']

    for i, hue_val in enumerate(hue_order):
        subset = summary_df[summary_df[hue] == hue_val]
        offset = (i - len(hue_order) / 2) * bar_width + bar_width / 2
        xpos = x_pos + offset

        # Use min/max as error bars instead of SE
        yerr_min = subset['mean_y'] - subset['min_y']
        yerr_max = subset['max_y'] - subset['mean_y']
        yerr = np.array([yerr_min, yerr_max])
        if err_type == 'se':
            yerr = subset['se_y']
        else:
            yerr_min = subset['mean_y'] - subset['min_y']
            yerr_max = subset['max_y'] - subset['mean_y']
            yerr = np.array([yerr_min, yerr_max])
            
        bars = ax.bar(
            xpos,
            subset['mean_y'],
            # yerr=subset['se_y'],  # original: standard error
            yerr=yerr,              # new: min/max error bars
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
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(f"Mean {y} by {x} and {hue}")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    return None    

def make_evaluation_plots(data, intervals, hue, base_dir, output_file_name, features, image_dir, hue_order=["LSTM", "xLSTM"]):
    # data = data[(data['Test'] != 182)]
    for interval in intervals:
        fig, ax = plt.subplots(1, len(features), figsize=(len(features) * 4, 3.5))
        plot_data = data[data['Interval'] == interval]
        
        for idx, feature in enumerate(features):
            plot_grouped_bar_with_error(data = plot_data, x="Test", y=feature, hue=hue, palette="viridis", ax=ax[idx], hue_order=hue_order)

        fig.tight_layout()
        fig.savefig(f"{image_dir}/{output_file_name}_{interval}.png", dpi=300)
        plt.close(fig=fig)

    fig, ax = plt.subplots(1, len(features), figsize=(4*len(features),3.5))
    for idx, feature in enumerate(features):
        plot_grouped_bar_with_error(data= data, x='Interval', y=feature, hue=hue, palette="viridis", ax=ax[idx], hue_order=hue_order)
    for axis in ax:
        axis.set_xlabel("Interval (s)")
    ax[0].set_ylabel("Mean Squared Error (MSE)")
    ax[1].set_ylabel("Pearson Correlation Coefficient")
    ax[0].set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f"{image_dir}/{output_file_name}_ts.png", dpi=300)
    plt.close(fig=fig)

    fig, ax = plt.subplots()
    data["Time_To_Train"] = pd.to_timedelta(data["Time_To_Train"]).dt.total_seconds()
    plot_grouped_bar_with_error(data=data, x="Interval", y="Time_To_Train", hue=hue, palette="viridis", ax=ax, hue_order=hue_order)
    # plot_grouped_bar_with_error(data=data, x="Interval", y="Time_To_Train", hue=hue, palette="viridis", ax=ax, hue_order=["LinReg"])
    ax.set_xlabel("Interval (s)")
    ax.set_ylabel("Mean Training Time (seconds)")
    ax.set_title("Comparison of Training Time by Interval and Model")
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(f"{base_dir}/TimetoTrain.png", dpi=300)
    plt.close(fig=fig)

    return None

def move_plots(df, model_types, configs, time_intervals, base_dir):
    mapping = {161 : 1, 172 : 2, 182 : 3, 183 : 4, 196 : 5, 207 : 6, 223 : 7, 232 : 8}
    for model_type in tqdm(model_types, desc="Model Progress"):
        # model_type_file = "xLSTM"
        for config in configs:
            for interval in time_intervals:
                temp = df[(df['Model'] == model_type) & (df['Config'] == config) & (df['Interval'] == interval)]
                for idx, row in tqdm(temp.iterrows(), desc=f"Moving Plots ({model_type}, {config}, {interval})"):
                    # MOVING BEST MODELS
                    from_path = f"{base_dir}/model/{config}/{interval}/t{row['Test']}_v{row['Val']}_{interval}_{model_type}_model.pt"
                    to_dir = f"{base_dir}/best_models/{config}/{mapping[row['Test']]}"
                    os.makedirs(to_dir, exist_ok=True)
                    with open(f"{to_dir}/best_models.txt", "a") as f:
                        f.write(f"{config} {interval} {model_type} {row['Test']} {row['Val']}\n")
                    to_path = f"{to_dir}/{interval}_{model_type}.pt"
                    shutil.copy(from_path, to_path)
                    # MOVING BEST COMBINATION LOSS CURVES
                    from_path = f"{base_dir}/loss_curves/{config}/{interval}/{model_type}_t{row['Test']}_v{row['Val']}.txt"
                    to_dir = f"{base_dir}/best_loss_curves/{config}/"
                    os.makedirs(to_dir, exist_ok=True)
                    with open(f"{to_dir}/best_losscurves.txt", "a") as f:
                        f.write(f"{config} {interval} {model_type} {row['Test']} {row['Val']}\n")
                    to_path = f"{to_dir}/{model_type}_{interval}_{row['Test']}.txt"
                    shutil.copy(from_path, to_path)
                    # MOVING BEST DISTRIBUTIONS
                    # from_path = f"{base_dir}/dist_plots/test/{interval}/{row['Test']}/{model_type}_{row['Val']}.png"
                    # to_dir = f"{base_dir}/best_loss_curves/{config}/"
                    # os.makedirs(to_dir, exist_ok=True)
                    # with open(f"{to_dir}/best_losscurves.txt", "a") as f:
                    #     f.write(f"{config} {interval} {model_type} {row['Test']} {row['Val']}\n")
                    # to_path = f"{to_dir}/{model_type}_{interval}.txt"
                    # shutil.copy(from_path, to_path)

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

def calc_ref_scores(base_dir, output_dir, time_shift, filenum):
    if filenum == 1:
        eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt")
    else:
        eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_wo_noise.txt")
    if "ref_MSE" in eval_out_cons.columns:
        print("Reference scores already calculated. Skipping...")
        return None
    eval_out_cons.sort_values(by=["Test", "Interval", "Val"], inplace=True)

    old_test, old_interval = 1, 1
    list1 = []
    event_id_map = {161: "1", 172: "3", 182: "4", 183: "5", 196: "6", 207: "7", 223: "8", 232: "9"}

    for idx, row in tqdm(eval_out_cons.iterrows(), total=len(eval_out_cons), desc="Calculating Reference Scores"):
        test = row['Test']
        val = row['Val']
        interval = row['Interval']
        model = row['Model']
        event_id = event_id_map[test]

        if (test == old_test) and (interval == old_interval):
            pass
        else:
            print(f"Loading Label {test} {interval}")
            label = load_label([event_id], "ILL11", interval, time_shift, trim=True, smoothing=None, divide_by=None)
            label['Timestamp'] = label['Timestamp'].apply(lambda x: UTCDateTime(x))
            old_test = test
            old_interval = interval
        output_df = pd.read_csv(f"{base_dir}/output_df/{row['Config']}/{row['Interval']}/{model}_t{test}_v{val}.csv", index_col=None)
        output_df['Timestamps'] = output_df["Timestamps"].apply(lambda x: UTCDateTime(x))
        assert len(output_df) == len(label), f"Length mismatch {len(output_df)} -|- {len(label)}"
        list1.append(np.round(mean_squared_error(label['Fv [kN]'].to_numpy(), output_df['Predicted_Output'].to_numpy()),4))
        
    eval_out_cons['ref_MSE'] = list1
    if filenum == 1:
        eval_out_cons.to_csv(f"{output_dir}/evaluation_output_constrained.txt", index=False)
    else:
        eval_out_cons.to_csv(f"{output_dir}/evaluation_output_wo_noise.txt", index=False)
    
    return None

def recalc_hist_wmse(base_dir, output_dir, filenum):
    if filenum == 1:
        eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt")
    else:
        eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_wo_noise.txt")
    if "Hist_WMSE2" in eval_out_cons.columns:
        print("WMSE scores already calculated. Skipping...")
        return None
    eval_out_cons.sort_values(by=["Test", "Interval", "Val"], inplace=True)
    list1 = []
    event_id_map = {161: "1", 172: "3", 182: "4", 183: "5", 196: "6", 207: "7", 223: "8", 232: "9"}
    for idx, row in tqdm(eval_out_cons.iterrows(), total=len(eval_out_cons), desc="Calculating HistWMSE Scores"):
        test = row['Test']
        val = row['Val']
        interval = row['Interval']
        model = row['Model']
        event_id = event_id_map[test]

        output_df = pd.read_csv(f"{base_dir}/output_df/{row['Config']}/{row['Interval']}/{model}_t{test}_v{val}.csv", index_col=None)
        output_df['Timestamps'] = output_df["Timestamps"].apply(lambda x: UTCDateTime(x))
        fig, ax = plt.subplots()
        bins = np.arange(2, 41, 4)
        heights1, width, _ = ax.hist(output_df['Output'].to_numpy(), bins=bins, color='red', alpha=0.8, label="Impact Force [kN]", density=False)
        heights2, _, _ = ax.hist(output_df['Predicted_Output'].to_numpy(), bins=bins, color='blue', alpha=0.6, label="Model Prediction", density=False)
        plt.close(fig=fig)
        centers = width[:-1] + (width[1:] - width[:-1]) / 2
        weights = centers
        # weights = weights / np.sum(weights)
        n = len(weights)
        weights_geom = np.geomspace(0.01, 50.0, n)
        # list1.append(np.round(np.dot(weights, np.sqrt((heights2 - heights1) ** 2)) / n, 4))
        # list1.append(np.round(np.dot(weights, np.abs(heights2 - heights1)) / n, 4))
        list1.append(np.round(np.sqrt(np.dot(weights_geom, ((heights2 - heights1) ** 2)))/n , 4))
        # list1.append(np.round(np.dot(weights_geom, np.abs(heights2 - heights1)) / n, 4))
    eval_out_cons['Hist_WMSE2'] = list1
    if filenum == 1:
        eval_out_cons.to_csv(f"{output_dir}/evaluation_output_constrained.txt", index=False)
    else:
        eval_out_cons.to_csv(f"{output_dir}/evaluation_output_wo_noise.txt", index=False)
    return None

def get_peaks(base_dir, output_dir, filenum):
    if filenum == 1:
        eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt")
    else:
        eval_out_cons = pd.read_csv(f"{output_dir}/evaluation_output_wo_noise.txt")
    if "True_Peak" in eval_out_cons.columns:
        print("True Peak already calculated. Skipping...")
        return None
    eval_out_cons.sort_values(by=["Test", "Interval", "Val"], inplace=True)
    peak_values, peak_values_true, peak_diff, peak_diff_percent = [], [], [], []
    event_id_map = {161: "1", 172: "3", 182: "4", 183: "5", 196: "6", 207: "7", 223: "8", 232: "9"}
    for idx, row in tqdm(eval_out_cons.iterrows(), total=len(eval_out_cons), desc="Getting peak"):
        test = row['Test']
        val = row['Val']
        interval = row['Interval']
        model = row['Model']
        event_id = event_id_map[test]

        output_df = pd.read_csv(f"{base_dir}/output_df/{row['Config']}/{row['Interval']}/{model}_t{test}_v{val}.csv", index_col=None)
        output_df['Timestamps'] = output_df["Timestamps"].apply(lambda x: UTCDateTime(x))
        tp = np.round(np.max(output_df['Output'].to_numpy()), 3)
        p = np.round(np.max(output_df['Predicted_Output'].to_numpy()), 3)
        peak_values.append(p)
        peak_values_true.append(tp)
        peak_diff.append(np.round(np.abs(p - tp), 3))
        peak_diff_percent.append(np.round(np.abs(tp-p) * 100 / tp, 2))
    eval_out_cons['True_Peak'] = peak_values_true
    eval_out_cons['Peak'] = peak_values
    eval_out_cons['Peak_Diff'] = peak_diff
    eval_out_cons['Peak_Diff_percent'] = peak_diff_percent
    if filenum == 1:
        eval_out_cons.to_csv(f"{output_dir}/evaluation_output_constrained.txt", index=False)
    else:
        eval_out_cons.to_csv(f"{output_dir}/evaluation_output_wo_noise.txt", index=False)
    return None

def make_dist_plots(base_dir, output_dir):
    eval_out_cons = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
    eval_out_cons.sort_values(by=["Test", "Interval", "Val"], inplace=True)
    dist_dir = f"{base_dir}/dist_best/"
    os.makedirs(dist_dir, exist_ok=True)
    event_id_map = {161: "1", 172: "3", 182: "4", 183: "5", 196: "6", 207: "7", 223: "8", 232: "9"}
    for idx, row in tqdm(eval_out_cons.iterrows(), total=len(eval_out_cons), desc="Making Hist Plots"):
        test = row['Test']
        val = row['Val']
        interval = row['Interval']
        model = row['Model']
        config = row['Config']
        event_id = event_id_map[test]

        output_df = pd.read_csv(f"{base_dir}/output_df/{row['Config']}/{row['Interval']}/{model}_t{test}_v{val}.csv", index_col=None)
        output_df['Timestamps'] = output_df["Timestamps"].apply(lambda x: UTCDateTime(x))
        fig, ax = plt.subplots()
        bins = np.arange(3, 45, 2)
        heights1, width, _ = ax.hist(output_df['Output'].to_numpy(), bins=bins, color='red', alpha=0.8, label="Impact Force [kN]", density=False)
        heights2, _, _ = ax.hist(output_df['Predicted_Output'].to_numpy(), bins=bins, color='blue', alpha=0.6, label="Model Prediction", density=False)
        ax.set_xlabel("Normal Force [kN]")
        ax.set_ylabel("Count")
        ax.set_title(f"{model} {interval} test {test} val {val}")
        ax.legend(loc='best')
        os.makedirs(f"{dist_dir}/{config}/{interval}/", exist_ok=True)
        fig.savefig(f"{dist_dir}/{config}/{interval}/{model}_{test}_{val}.png", dpi=300)
        plt.close(fig=fig)

def main(task:str, model_types:list[str], configs:list[str], time_shift:int=10):
    divide_by = 45
    smoothing = 60
    time_intervals = [5, 15]
    
    if time_shift == "average":
        julday_list = [161, 172, 182, 196, 207, 223, 232]
        event_id_list = [1, 3, 4, 6, 7, 8, 9]
    elif time_shift == "dynamic":
        julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
        event_id_list = [1, 3, 4, 5, 6, 7, 8, 9]
    # task = "by150_"+task
    base_dir = f"../{task}/{time_shift}_{smoothing}_{divide_by}"
    image_dir1 = f"{base_dir}/images/with_noise"
    image_dir2 = f"{base_dir}/images/without_noise"
    os.makedirs(image_dir1, exist_ok=True)
    os.makedirs(image_dir2, exist_ok=True)
    output_dir = f"../{task}/{time_shift}_{smoothing}_{divide_by}/model_evaluation"
    
    calc_ref_scores(base_dir, output_dir, time_shift, filenum=1)
    calc_ref_scores(base_dir, output_dir, time_shift, filenum=2)
    recalc_hist_wmse(base_dir, output_dir, filenum=1)
    recalc_hist_wmse(base_dir, output_dir, filenum=2)
    get_peaks(base_dir, output_dir, filenum=1)
    get_peaks(base_dir, output_dir, filenum=2)
    
    if os.path.exists(f"{output_dir}/best_combinations.csv"):
        best_combinations_df = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
    else:
        print("\tSelecting Best Combinations")
        data = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt", index_col=False)
        best_combinations_df = pd.DataFrame(columns = data.columns.values)
        for model_type in tqdm(model_types, desc="Model Progress"):
            for config in tqdm(configs, desc="Config Progress"):
                temp_data = data[(data["Model"] == model_type) & (data['Config'] == config)]
                # print(temp_data)
                for interval in tqdm(time_intervals, desc=f"Interval Progress ({model_type}, {config})"):
                    for test_julday in tqdm(julday_list, desc=f"Julday Progress"):
                            # print(f"Processing {model_type} {config} {interval} {test_julday}")
                            temp = temp_data[(temp_data["Test"] == test_julday) & (temp_data["Interval"] == interval)]
                            temp.reset_index(inplace=True, drop=True)
                            # print(temp)
                            # temp = temp.iloc[temp.nsmallest(1, "MSE").index]
                            # if test_julday == 223:
                            #     temp = temp[temp['Val'] == 207]
                            # else:
                            temp = temp.iloc[temp.nsmallest(1, "MSE").index]
                            best_combinations_df.loc[len(best_combinations_df)] = temp.values[0]
        best_combinations_df.to_csv(f"{output_dir}/best_combinations.csv", index=False)

    if os.path.exists(f"{output_dir}/best_combinations_wo_noise.csv"):
        best_combinations_df = pd.read_csv(f"{output_dir}/best_combinations_wo_noise.csv", index_col=False)
    else:
        print("\tSelecting Best Combinations")
        data = pd.read_csv(f"{output_dir}/evaluation_output_wo_noise.txt", index_col=False)
        best_combinations_df = pd.DataFrame(columns = data.columns.values)
        for model_type in tqdm(model_types, desc="Model Progress"):
            for config in tqdm(configs, desc="Config Progress"):
                temp_data = data[(data["Model"] == model_type) & (data['Config'] == config)]
                # print(temp_data)
                for interval in tqdm(time_intervals, desc=f"Interval Progress ({model_type}, {config})"):
                    for test_julday in tqdm(julday_list, desc=f"Julday Progress"):
                            # print(f"Processing {model_type} {config} {interval} {test_julday}")
                            temp = temp_data[(temp_data["Test"] == test_julday) & (temp_data["Interval"] == interval)]
                            temp.reset_index(inplace=True, drop=True)
                            # print(temp)
                            # if test_julday == 223:
                            #     temp = temp[temp['Val'] == 207]
                            # else:
                            temp = temp.iloc[temp.nsmallest(1, "MSE").index]
                            best_combinations_df.loc[len(best_combinations_df)] = temp.values[0]
        best_combinations_df.to_csv(f"{output_dir}/best_combinations_wo_noise.csv", index=False)

    if os.path.exists(f"{output_dir}/best_combinations_ref.csv"):
        best_combinations_df = pd.read_csv(f"{output_dir}/best_combinations_ref.csv", index_col=False)
    else:
        print("\tSelecting Best Combinations")
        data = pd.read_csv(f"{output_dir}/evaluation_output_wo_noise.txt", index_col=False)
        best_combinations_df = pd.DataFrame(columns = data.columns.values)
        for model_type in tqdm(model_types, desc="Model Progress"):
            for config in tqdm(configs, desc="Config Progress"):
                temp_data = data[(data["Model"] == model_type) & (data['Config'] == config)]
                # print(temp_data)
                for interval in tqdm(time_intervals, desc=f"Interval Progress ({model_type}, {config})"):
                    for test_julday in tqdm(julday_list, desc=f"Julday Progress"):
                            # print(f"Processing {model_type} {config} {interval} {test_julday}")
                            temp = temp_data[(temp_data["Test"] == test_julday) & (temp_data["Interval"] == interval)]
                            temp.reset_index(inplace=True, drop=True)
                            # print(temp)
                            # if test_julday == 223:
                            #     temp = temp[temp['Val'] == 207]
                            # else:
                            temp = temp.iloc[temp.nsmallest(1, "ref_MSE").index]
                            best_combinations_df.loc[len(best_combinations_df)] = temp.values[0]
        best_combinations_df.to_csv(f"{output_dir}/best_combinations_ref.csv", index=False)

    make_dist_plots(base_dir, output_dir)

    print("\tMaking Plots and Moving Images")
    for config in configs:
        data = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
        data = data[data['Config'] == config]
        image_dir1t = image_dir1+f"/{config}"
        image_dir2t = image_dir2+f"/{config}"
        os.makedirs(image_dir1t, exist_ok=True)
        os.makedirs(image_dir2t, exist_ok=True)
        features = ['MSE', 'Hist_WMSE', "ref_MSE"]
        make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Best_Comparison_Plot", features, image_dir1t, ['LSTM', 'xLSTM'])
        data = data[data['Test'] != 183]
        make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Best_Comparison_Plot_without183", features, image_dir1t, ['LSTM', 'xLSTM'])
        data = pd.read_csv(f"{output_dir}/evaluation_output_constrained.txt", index_col=False)
        make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Comparison_Plot", features, image_dir1t, ['LSTM', 'xLSTM'])

        data = pd.read_csv(f"{output_dir}/best_combinations_wo_noise.csv", index_col=False)
        data = data[data['Config'] == config]
        features = ['MSE', 'Hist_WMSE', "ref_MSE"]
        make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Best_Comparison_Plot", features, image_dir2t, ['LSTM', 'xLSTM'])
        data = data[data['Test'] != 183]
        make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Best_Comparison_Plot_without183", features, image_dir2t, ['LSTM', 'xLSTM'])
        
        data = pd.read_csv(f"{output_dir}/evaluation_output_wo_noise.txt", index_col=False)
        make_evaluation_plots(data, time_intervals, 'Model', base_dir, "Comparison_Plot", features, image_dir2t, ['LSTM', 'xLSTM'])
    
    data = pd.read_csv(f"{output_dir}/best_combinations.csv", index_col=False)
    move_plots(data, model_types, configs, time_intervals, base_dir)
    
    print("\tMaking Explaination Plots")
    data = pd.read_csv(f"../{task}/{time_shift}_{smoothing}_{divide_by}/model_evaluation/best_combinations.csv", index_col=False)
    data = data[data['Config'] == 'v4']
    
    output_file_dir = f"../{task}/{time_shift}_{smoothing}_{divide_by}/output_df/v4"

    i = 0
    if time_shift == "average":
        event_id_list = [1, 3, 4, 5, 6, 7, 8, 9]
    elif time_shift == "dynamic":
        event_id_list = [3, 6, 7, 8]
    for event_id in tqdm(event_id_list, desc= "Julday Progress"):
        event_info = time_config[str(event_id)]
        julday = event_info['julday'] if type(event_info['julday']) is int else event_info['julday'][0]
        date = event_info['date'] if type(event_info['date']) is str else event_info['date'][0]
        for interval in time_intervals:
            temp = data[(data['Test'] == julday) & (data['Interval'] == interval)]
            fig, ax = plt.subplots(2, 1, figsize=(8.0, 6.0), sharey=True, sharex=True)
            target_output = load_label([str(event_id)], "ILL11", interval, time_shift, trim=False, smoothing=smoothing, divide_by=None)
            target_times = [UTCDateTime(i).matplotlib_date for i in target_output['Timestamp'].to_numpy()]
            for idx, row in temp.iterrows():
                interval = row['Interval']
                model_type = row['Model']
                test = row['Test']
                val = row['Val']
                st = load_seismic_data(event_id= event_id, station= 'ILL11', year= 2019, trim= False)
                # print(st)
                file = pd.read_csv(f"{output_file_dir}/{interval}/{model_type}_t{test}_v{val}.csv", index_col=False)
                times = [UTCDateTime(i).matplotlib_date for i in file['Timestamps'].to_numpy()]
                # target_output = file['Output'].to_numpy()
                predicted_output = file['Predicted_Output'].to_numpy()
                
                if model_type == 'xLSTM':
                    ax[0].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[0].set_ylabel(r"Amplitude (mm/s)");
                    ax[0].set_ylim(-1.5, 1.5);
                    ax2 = ax[0].twinx()
                    ax2.plot(target_times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    ax2.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
                    ax2.xaxis_date()
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
                    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                    ax2.set_xlim(UTCDateTime(event_info['start_time']).matplotlib_date, UTCDateTime(event_info['end_time']).matplotlib_date);
                    ax2.set_ylabel("Normal Force [kN]");
                    ax2.set_ylim(bottom=0)
                    ax2.legend(loc='best');
                    ax2.set_ylim(0, 50);

                elif model_type == 'LSTM':
                    ax[1].plot(st[0].times('matplotlib'), st[0].data, color="black", label= "ILL11", alpha=0.5)
                    ax[1].set_ylabel(r"Amplitude (mm/s)");
                    ax[1].set_ylim(-1.5, 1.5);
                    ax4 = ax[1].twinx()
                    ax4.plot(target_times, target_output['Fv [kN]'].to_numpy(), label="Impact Force Target [kN]", alpha=0.9, color='r',linewidth=1)
                    ax4.plot(times, predicted_output, label="Model Prediction", alpha=0.8, color='b',linewidth=1)
                    ax4.xaxis_date()
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
                    ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                    ax4.set_xlim(UTCDateTime(event_info['start_time']).matplotlib_date, UTCDateTime(event_info['end_time']).matplotlib_date);
                    ax4.set_ylabel("Normal Force [kN]");
                    ax4.set_ylim(bottom=0)
                    ax4.legend(loc='best');
                    ax4.set_ylim(0, 50);

            ax[0].set_title("xLSTM Model");
            ax[1].set_title("LSTM Model");
            fig.suptitle(f"Model comparison, Interval = {interval} seconds, Test Julday = {julday}", fontdict={'fontsize':8, 'fontweight':'bold'});
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

    # python run_analysis.py --time "dynamic" --task "comparison_baseline_cv_5_15_1" --config "v4" --model_type "xLSTM" "LSTM"
    