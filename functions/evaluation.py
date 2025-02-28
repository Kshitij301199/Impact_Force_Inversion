import os
import json
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
    paths = json.load(file)
import sys
sys.path.append(paths['BASE_DIR'])
import numpy as np
from dtaidistance import dtw
from scipy.signal import savgol_filter
from obspy.core import UTCDateTime

from data_processing.read_data import load_label

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

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
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

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
    return np.where(denominator != 0, numerator / denominator, 1)

def dtw_distance_calc(seq1, seq2, interval):
    if interval == 60:
        window = 61
    elif interval == 30:
        window = 121
    elif interval == 15:
        window = 241
    elif interval == 5:
        window = 751
    seq1 = savgol_filter(seq1, window, 3, 0, mode='interp')
    seq2 = savgol_filter(seq2, window, 3, 0, mode='interp')
    return dtw.distance(seq1, seq2)

def evaluate_model(model_type, test_julday, val_julday, interval_seconds, y_true, y_pred, out_dir, time_to_train:str):
    output_dir = f"{out_dir}/model_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/evaluation_output.txt"
    try:
        with open(filename, "x") as file:  # "x" mode creates a file if it does not exist
            file.write("Model,Config,Time_To_Train,Test,Val,Interval,MSE,SMAPE,PCC,DTW_Dist,MSE_0,SMAPE_0,PCC_0\n")
            print(f"File '{filename}' created with columns names")
    except FileExistsError:
        print(f"File '{filename}' already exists.")

    julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
    date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    zero_label = load_label([date_list.pop(julday_list.index(test_julday))], "ILL11", interval_seconds, 0)
    zero_label['Timestamp'] = zero_label['Timestamp'].apply(UTCDateTime)
    
    mse_score = mse(y_true, y_pred)
    smape_score = smape(y_true, y_pred)
    pcc_score = pcc(y_true, y_pred)
    dtw_distance_score = dtw_distance_calc(y_true, y_pred, interval_seconds)

    with open(f"{output_dir}/evaluation_output.txt", "a") as f:
        string = (
    f"{model_type},{time_to_train},{test_julday},{val_julday},{interval_seconds},"
    f"{mse_score:.4f},{smape_score:.4f},{pcc_score:.4f},"
    f"{dtw_distance_score:.4f},"
    f"{mse(zero_label['Fv [kN]'].to_numpy(), y_pred):.4f},"
    f"{smape(zero_label['Fv [kN]'].to_numpy(), y_pred):.4f},"
    f"{pcc(zero_label['Fv [kN]'].to_numpy(), y_pred):.4f}\n"
)
        f.write(string)
    return None

