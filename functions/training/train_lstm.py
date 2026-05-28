#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2026-05-30
#__author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
#__find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import sys
import json
import argparse
from pathlib import Path

# Dynamic path resolution
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

def load_config(filename):
    path = project_root / "config" / filename
    with open(path, "r") as file:
        return json.load(file)

paths = load_config("paths.json")
data_params = load_config("data_parameters.json")
time_config = load_config("event_id_map.json")

# Set CUDA environment variables
if 'CUDA_HOME' in paths:
    os.environ["CUDA_HOME"] = paths['CUDA_HOME']
    os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.path.join(os.environ["CUDA_HOME"], "lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")

sys.path.append(paths['BASE_DIR'])
import torch
torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from obspy import UTCDateTime
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from functions.data_processing.read_data import load_data, load_label, load_seismic_data
from functions.data_processing.dataloader import SequenceDataset, DataLoader

from functions.utils import *
from functions.training.train import ModelTrainer
from functions.evaluation.eval import evaluate_model, sanity_check_train
from functions.evaluation.plot_image import plot_image

from models.LSTM_model import LSTMRegressor

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_dirs(task: str, time_shift_minutes, smoothing, divide_by, interval_seconds, config_option, repeat):
    """Generic directory creation."""
    base_output = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmax']}_{repeat}"
    params_subdir = f"{time_shift_minutes}_{smoothing}_{divide_by}"
    output_dir = os.path.join(base_output, params_subdir)
    
    model_dir = os.path.join(output_dir, "model", config_option, str(interval_seconds))
    image_dir = os.path.join(output_dir, "test_results", "lstm", config_option, str(interval_seconds))
    save_dir = os.path.join(output_dir, "output_df", config_option, str(interval_seconds))
    curve_dir = os.path.join(output_dir, "loss_curves", config_option, str(interval_seconds))
    
    for d in [model_dir, image_dir, save_dir, curve_dir]:
        os.makedirs(d, exist_ok=True)
        
    return output_dir, model_dir, image_dir, save_dir, curve_dir

def main(test_id: int, val_id: int, time_shift_minutes: int | str, smoothing: int, divide_by: int, station: str, interval_seconds: int, config_option: str, task: str, repeat=1):
    """Main function to train and evaluate LSTM model for impact force inversion.
    """
    test_id, val_id = str(test_id), str(val_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    
    num_intervals = int((data_params['time_window'] * 60) // interval_seconds)
    output_dir, model_dir, image_dir, save_dir, curve_dir = make_dirs(task, time_shift_minutes, smoothing, divide_by, interval_seconds, config_option, repeat)
    
    # Standardized event list logic
    event_id_list = ["1", "3", "4", "5", "6", "7", "8", "9"]
    if time_shift_minutes not in ["average", "dynamic"]:
        if "5" in event_id_list: event_id_list.remove("5")
    
    test_info = time_config[str(test_id)]
    val_info = time_config[str(val_id)]
    test_julday = test_info['julday'] if isinstance(test_info['julday'], int) else test_info['julday'][0]
    val_julday = val_info['julday'] if isinstance(val_info['julday'], int) else val_info['julday'][0]

    train_id_list = [eid for eid in event_id_list if eid not in [test_id, val_id]]
    if test_id == val_id and test_id in event_id_list:
        train_id_list = [eid for eid in event_id_list if eid != test_id]

    curve_file = f"{curve_dir}/LSTM_t{test_julday}_v{val_julday}.txt"
    with open(curve_file, "w") as file:
        file.write("Epoch;Train_Loss;Val_Loss;LR;mean_g;max_g\n")

    # LOAD DATA
    train_juldays = [time_config[eid]['julday'] for eid in train_id_list]
    # Flatten if necessary
    train_juldays = [j if isinstance(j, int) else item for j in train_juldays for item in (j if isinstance(j, list) else [j])]

    print(f"Train Day List : {train_juldays}, Val Day List : {val_julday}, Test Day List : {test_julday}")
    
    print(f"{'Loading Data':-^50}")
    total_data, _ = load_data(train_id_list, station, trim=True, abs=True)
    val_data, _ = load_data([val_id], station, trim=False, abs=True)
    test_data, _ = load_data([test_id], station, trim=True, abs=True)
    st_test = load_seismic_data(test_id, station, year=2019, trim=True)
    print(f"Data --> Train : {len(total_data)} Test : {len(test_data)}")
    total_target = load_label(event_id_list= train_id_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                smoothing=smoothing, 
                                divide_by=divide_by)
    val_target = load_label(event_id_list= [val_id], station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                smoothing=smoothing, 
                                divide_by=divide_by,
                                trim=False)
    test_target = load_label(event_id_list= [test_id], station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                smoothing=smoothing, 
                                divide_by=divide_by,
                                trim=True)
    print(f"Target --> Train : {len(total_target)} Test : {len(test_target)}")
    print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")

    # INITIALIZE MODEL
    print("Initialising Model")
    config_path = project_root / "config" / task / f"lstm_{config_option}_{interval_seconds}sec_config.json"
    with open(config_path, "r") as f:
        model_params = json.load(f)
        
    with open(f"{output_dir}/model_config.txt", "a") as f:
        f.write(f"lstm :\n{model_params}\n")
        
    model = LSTMRegressor(**model_params)
    criterion = nn.MSELoss()

    lr = 5e-4 if interval_seconds == 1 else 1e-4

    batch_size = 128
    optimizer = optim.Adam(model.parameters(), lr=lr)

    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.50, patience=10)

    # INIT DATALOADERS
    print("Initialising Dataloaders")
    train_dataset = SequenceDataset(total_data, total_target['Fv [kN]'].to_numpy(),
                            total_target['Timestamp'].to_numpy(),
                            interval_count=num_intervals, sequence_length=interval_seconds * 100)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
    val_dataset = SequenceDataset(val_data, val_target['Fv [kN]'].to_numpy(),
                            val_target['Timestamp'].to_numpy(),
                            interval_count=num_intervals, sequence_length=interval_seconds * 100)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Adjust batch size as needed
    test_dataset = SequenceDataset(test_data, test_target['Fv [kN]'].to_numpy(), 
                                    test_target['Timestamp'].to_numpy(), 
                                    interval_count=num_intervals, sequence_length=interval_seconds * 100)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Adjust batch size as needed

    print("Training Model")
    trainer = ModelTrainer(model=model, criterion=criterion, optimizer=optimizer,
                           warmup_scheduler=None, main_scheduler=main_scheduler,
                           train_loader=train_dataloader, val_loader=val_dataloader, test_loader=test_dataloader,
                           model_dir=model_dir, curve_file=curve_file, interval=interval_seconds,
                           test_julday=test_julday, val_julday=val_julday, model_type="LSTM", device=device)
    print(f"{'Starting Training':-^50}")
    trainer.train(num_epochs=200, patience=25)
    print(f"{'End Training':-^50}")
    
    print("Sanity check the training")
    in_seq, pred_out, target_out, timestamps, time_to_train = trainer.check_train(mult_by=divide_by)
    sanity_check_train(target = np.concatenate(target_out),
                       pred = np.concatenate(pred_out),
                       model_type="LSTM",
                       interval_seconds=interval_seconds,
                       test_julday=test_julday, val_julday=val_julday,
                       out_dir=output_dir)
    
    print(f"{'Start Testing':-^50}")
    in_seq, pred_out, target_out, timestamps, time_to_train = trainer.test(mult_by=divide_by)
    print(f"Saving output to {save_dir}/LSTM_t{test_julday}_v{val_julday}.csv")
    times = [UTCDateTime(t) for t in np.concatenate(timestamps)]
    df = pd.DataFrame(data={"Timestamps":times, "Output":np.concatenate(target_out), "Predicted_Output":np.concatenate(pred_out)})
    df.to_csv(f"{save_dir}/LSTM_t{test_julday}_v{val_julday}.csv", index=False)
    print(f"{'End Testing':-^50}")

    
    print("Making Plot")
    start_time = get_current_time()
    plot_image(st_test, pred_out, target_out, timestamps, image_dir, test_id, val_id, interval_seconds, trim=True, smoothing=smoothing)
    evaluate_model(model_type=f"LSTM,{config_option}", 
                   test_id=test_id, 
                   val_id=val_id, 
                   interval_seconds=interval_seconds, 
                   y_true=np.concatenate(target_out), 
                   y_pred=np.concatenate(pred_out), 
                   smoothing=smoothing,
                   out_dir=output_dir,
                   time_to_train=time_to_train,
                   )
    end_time = get_current_time()
    get_time_elapsed(start_time, end_time)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_event_id", type=int, default=161, help= "test julday")
    parser.add_argument("--val_event_id", type=int, default=172, help= "val julday")
    parser.add_argument("--time_shift_mins", default=10, help= "enter label time shift")
    parser.add_argument("--station", type=str, default="ILL13", help= "input station")
    parser.add_argument("--interval", type=int, default=30, help= "interval seconds")
    parser.add_argument("--config_op", type=str,default="default", help= "config option")
    parser.add_argument("--task", type=str, default="comparison_baseline", help= "name of the task corresponding to parameter directory")
    parser.add_argument("--smoothing", type=int, default=30, help="enter a value used for smoothing the raw data")
    parser.add_argument("--divide_by", type=int, default=350, help="normalization constant")
    parser.add_argument("--repeat", type=int, default=1, help="Number to times to repeat process")

    args = parser.parse_args()
    print(f"Running main with {args.test_event_id} {args.station} {args.config_op} {args.task}")
    for repeat in range(1, args.repeat + 1):
        main(args.test_event_id,
            args.val_event_id, 
            args.time_shift_mins, 
            args.smoothing,
            args.divide_by,
            args.station, 
            args.interval, 
            args.config_op, 
            args.task,
            repeat)