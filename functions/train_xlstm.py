import os
import json
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
    paths = json.load(file)
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
    data_params = json.load(file)
# Set CUDA environment variables
os.environ["CUDA_HOME"] = paths['CUDA_HOME']
os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = os.path.join(os.environ["CUDA_HOME"], "lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
import sys
sys.path.append(paths['BASE_DIR'])
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from obspy import UTCDateTime
from sklearn.preprocessing import MinMaxScaler

from data_processing.read_data import load_data, load_label, load_seismic_data
from data_processing.dataloader import SequenceDataset, DataLoader

from functions.utils import *
from functions.train import train_model
from functions.evaluation import evaluate_model
from functions.plot_image import plot_image

from models.xLSTM_model import xLSTMRegressor

def main(test_julday:int, val_julday:int, time_shift_minutes:int|str, station:str, interval_seconds:int, config_option:str, task:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    num_intervals = int((data_params['time_window'] * 60) // interval_seconds)
    model_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}/model/{config_option}/{interval_seconds}"
    image_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}/test_results/xlstm/{config_option}/{interval_seconds}"
    save_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}/output_df/{config_option}/{interval_seconds}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
    date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    
    test_date = date_list.pop(julday_list.index(test_julday))
    julday_list.remove(test_julday)
    val_date = date_list.pop(julday_list.index(val_julday))  
    julday_list.remove(val_julday)
    test_julday_list = [test_julday]
    test_date_list = [test_date]
    val_julday_list, val_date_list = [val_julday], [val_date]

    # LOAD DATA
    print(f"{'Loading Data':-^50}")
    total_data = load_data(julday_list, station)
    val_data = load_data(val_julday_list, station)
    test_data = load_data(test_julday_list, station)
    st_test = load_seismic_data(test_julday, station)
    print(f"Data --> Train : {len(total_data)} Test : {len(test_data)}")
    total_target = load_label(date_list= date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes)
    val_target = load_label(date_list= val_date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes)
    test_target = load_label(date_list= test_date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes)
    print(f"Target --> Train : {len(total_target)} Test : {len(test_target)}")
    print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")


    # INITIALIZE MODEL
    print("Initialising Model")
    with open(f"./config/{task}/xlstm_{config_option}_{interval_seconds}sec_config.json", "r") as f:
        config = json.load(f)
    with open(f"{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}/model_config.txt", "a") as f:
        string = f"xlstm :\n{config}\n"
        f.write(string)
    model = xLSTMRegressor(**config)
    criterion = nn.MSELoss()
    if interval_seconds == 1:
        lr = 5e-4
    else:
        lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = get_batch_size(interval_seconds)

    # INIT DATALOADERS
    print("Initialising Dataloaders")
    scaler = MinMaxScaler(feature_range=(0, 2))
    scaler.data_min_ = np.array([0])
    scaler.data_max_ = np.array([350])
    scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / (scaler.data_max_ - scaler.data_min_)
    scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
    rem = total_target['Fv [kN]'].to_numpy().shape
    train_dataset = SequenceDataset(total_data, scaler.transform(total_target['Fv [kN]'].to_numpy().reshape(-1,1)).reshape(rem),
                            total_target['Timestamp'].to_numpy(),
                            interval_count=num_intervals, sequence_length=interval_seconds * 100)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
    rem = val_target['Fv [kN]'].to_numpy().shape
    val_dataset = SequenceDataset(val_data, scaler.transform(val_target['Fv [kN]'].to_numpy().reshape(-1,1)).reshape(rem),
                            val_target['Timestamp'].to_numpy(),
                            interval_count=num_intervals, sequence_length=interval_seconds * 100)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Adjust batch size as needed
    rem = test_target['Fv [kN]'].to_numpy().shape
    test_dataset = SequenceDataset(test_data, scaler.transform(test_target['Fv [kN]'].to_numpy().reshape(-1,1)).reshape(rem), 
                                    test_target['Timestamp'].to_numpy(), 
                                    interval_count=num_intervals, sequence_length=interval_seconds * 100)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Adjust batch size as needed

    print("Training Model")
    in_seq, pred_out, target_out, timestamps, time_to_train = train_model(model, criterion, optimizer,
                                                           100, 10, interval_seconds, test_julday, val_julday,
                                                           'xLSTM', train_dataloader, val_dataloader,
                                                           test_dataloader, model_dir, scaler)
    times = [UTCDateTime(t) for t in np.concatenate(timestamps)]
    df = pd.DataFrame(data={"Timestamps":times, "Output":np.concatenate(target_out), "Predicted_Output":np.concatenate(pred_out)})
    df.to_csv(f"{save_dir}/xLSTM_t{test_julday}_v{val_julday}.csv", index=False)
    print("Making Plot")
    start_time = get_current_time()
    plot_image(st_test, pred_out, target_out, timestamps, image_dir, test_julday, val_julday, interval_seconds)
    evaluate_model(model_type=f"xLSTM,{config_option}", 
                   test_julday=test_julday, 
                   val_julday=val_julday, 
                   interval_seconds=interval_seconds, 
                   y_true=np.concatenate(target_out), 
                   y_pred=np.concatenate(pred_out), 
                   out_dir=f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}",
                   time_to_train=time_to_train,
                   )
    end_time = get_current_time()
    get_time_elapsed(start_time, end_time)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_julday", type=int, default=161, help= "test julday")
    parser.add_argument("--val_julday", type=int, default=172, help= "val julday")
    parser.add_argument("--time_shift_mins", default=10, help= "enter label time shift")
    parser.add_argument("--station", type=str, default="ILL13", help= "input station")
    parser.add_argument("--interval", type=int, default=30, help= "interval seconds")
    parser.add_argument("--config_op", type=str,default="default", help= "config option")
    parser.add_argument("--task", type=str, default="comparison_baseline", help= "name of the task corresponding to parameter directory")

    args = parser.parse_args()
    print(f"Running main with {args.test_julday} {args.station} {args.config_op} {args.task}")
    main(args.test_julday,
        args.val_julday, 
        args.time_shift_mins, 
        args.station, 
        args.interval, 
        args.config_op, 
        args.task)