import os
import json
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
    paths = json.load(file)
import sys
sys.path.append(paths['BASE_DIR'])
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
# from dtaidistance import dtw
# from scipy.signal import savgol_filter
from obspy.core import UTCDateTime

from data_processing.read_data import load_label

def evaluate_model(model_type:str, test_julday:int, val_julday:int, interval_seconds:int, y_true, y_pred, smoothing:int,out_dir:str, time_to_train:str):
    print(f"{'Evaluating Model':-^50}")
    julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
    date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    constraint_df = pd.read_csv(f"{paths['BASE_DIR']}/label/correct_metrics_time_window.csv", index_col=False)

    if test_julday == 161:
        window_df = constraint_df.iloc[:2]
        print(window_df)
        window_start_1, window_end_1 = UTCDateTime(window_df['Start_Time'].iloc[0]), UTCDateTime(window_df['End_Time'].iloc[0])
        window_start_2, window_end_2 = UTCDateTime(window_df['Start_Time'].iloc[1]), UTCDateTime(window_df['End_Time'].iloc[1])
    else:
        window_df = constraint_df.iloc[julday_list.index(test_julday) + 1]
        print(window_df)
        window_start, window_end = UTCDateTime(window_df['Start_Time']), UTCDateTime(window_df['End_Time'])

    output_dir = f"{out_dir}/model_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/evaluation_output.txt"
    filename2 = f"{output_dir}/evaluation_output_constrained.txt"
    try:
        with open(filename, "x") as file:  # "x" mode creates a file if it does not exist
            file.write("Model,Config,Time_To_Train,Test,Val,Interval,MSE_ts,RMSE_ts,MAE_ts,R2_ts,Corr_ts,PearsonR_ts,MSE_0,RMSE_0,MAE_0,R2_0,Corr_0,PearsonR_0\n")
            print(f"File '{filename}' created with columns names")
        with open(filename2, "x") as file:  # "x" mode creates a file if it does not exist
            file.write("Model,Config,Time_To_Train,Test,Val,Interval,MSE_ts,RMSE_ts,MAE_ts,R2_ts,Corr_ts,PearsonR_ts,MSE_0,RMSE_0,MAE_0,R2_0,Corr_0,PearsonR_0\n")
            print(f"File '{filename}' created with columns names")
    except FileExistsError:
        pass

    zero_label = load_label([date_list.pop(julday_list.index(test_julday))], "ILL11", interval_seconds, 0, trim=False, smoothing=smoothing)
    zero_label['Timestamp'] = zero_label['Timestamp'].apply(UTCDateTime)
    zero_label = zero_label.iloc[:len(y_true)]
    zero_label['True_Value'] = y_true
    zero_label['Pred_Value'] = y_pred
    
    r1, _ = pearsonr(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy())
    r2, _ = pearsonr(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy())
    corr1 = np.correlate(zero_label['True_Value'].to_numpy() - np.mean(zero_label['True_Value'].to_numpy()),
                         zero_label['Pred_Value'].to_numpy() - np.mean(zero_label['Pred_Value'].to_numpy()),
                         mode='full')
    lag1 = np.argmax(corr1) - (len(zero_label['True_Value'].to_numpy()) - 1)
    corr2 = np.correlate(zero_label['Fv [kN]'].to_numpy() - np.mean(zero_label['Fv [kN]'].to_numpy()), 
                         zero_label['Pred_Value'].to_numpy() - np.mean(zero_label['Pred_Value'].to_numpy()), 
                         mode='full')
    lag2 = np.argmax(corr2) - (len(zero_label['Fv [kN]'].to_numpy()) - 1)
    with open(filename, "a") as f:
        string = (
    f"{model_type},{time_to_train},{test_julday},{val_julday},{interval_seconds},"
    f"{mean_squared_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{np.sqrt(mean_squared_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy())):.4f},"
    f"{mean_absolute_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{r2_score(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{lag1:.4f},"
    f"{r1:.4f},"
    f"{mean_squared_error(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{np.sqrt(mean_squared_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy())):.4f},"
    f"{mean_absolute_error(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{r2_score(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{lag2:.4f},"
    f"{r2:.4f}\n"
)
        f.write(string)

    # Constrained evaluation
    if test_julday == 161:
        print(len(zero_label))
        zero_label_temp = zero_label[zero_label['Timestamp'].between(window_start_1, window_end_1)]
        zero_label = pd.concat([zero_label_temp, zero_label[zero_label['Timestamp'].between(window_start_2, window_end_2)]])
        print(len(zero_label))
    else:
        zero_label = zero_label[zero_label['Timestamp'].between(window_start, window_end)]
    
    r1, _ = pearsonr(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy())
    r2, _ = pearsonr(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy())
    corr1 = np.correlate(zero_label['True_Value'].to_numpy() - np.mean(zero_label['True_Value'].to_numpy()), zero_label['Pred_Value'].to_numpy() - np.mean(zero_label['Pred_Value'].to_numpy()), mode='full')
    lag1 = np.argmax(corr1) - (len(zero_label['True_Value'].to_numpy()) - 1)
    corr2 = np.correlate(zero_label['Fv [kN]'].to_numpy() - np.mean(zero_label['Fv [kN]'].to_numpy()), zero_label['Pred_Value'].to_numpy() - np.mean(zero_label['Pred_Value'].to_numpy()), mode='full')
    lag2 = np.argmax(corr2) - (len(zero_label['Fv [kN]'].to_numpy()) - 1)

    with open(filename2, "a") as f:
        string = (
    f"{model_type},{time_to_train},{test_julday},{val_julday},{interval_seconds},"
    f"{mean_squared_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{np.sqrt(mean_squared_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy())):.4f},"
    f"{mean_absolute_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{r2_score(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{lag1:.4f},"
    f"{r1:.4f},"
    f"{mean_squared_error(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{np.sqrt(mean_squared_error(zero_label['True_Value'].to_numpy(), zero_label['Pred_Value'].to_numpy())):.4f},"
    f"{mean_absolute_error(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{r2_score(zero_label['Fv [kN]'].to_numpy(), zero_label['Pred_Value'].to_numpy()):.4f},"
    f"{lag2:.4f},"
    f"{r2:.4f}\n"
)
        f.write(string)
    return None

