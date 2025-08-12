import os
import sys
import json
import argparse

with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
    paths = json.load(file)
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
    data_params = json.load(file)

# Set CUDA environment variables
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
from functions.evaluation.eval import evaluate_model
from functions.evaluation.plot_image import plot_image

from models.LSTM_model import LSTMRegressor

def warmup_lambda(epoch):
    warmup_epochs = 5
    return min(1.0, (epoch + 1) / warmup_epochs)

def set_seed(seed=42):
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # disable benchmarking for reproducibility

def main(test_julday:int, val_julday:int, time_shift_minutes:int|str, smoothing:int,station:str, interval_seconds:int, config_option:str, task:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    set_seed()
    num_intervals = int((data_params['time_window'] * 60) // interval_seconds)
    if task == "abalation_study_1":
        output_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}" 
        model_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}/model/{test_julday}"
        image_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}/test_results/lstm/{test_julday}"
        save_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}/output_df/{test_julday}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    elif task == "slstm_v_mlstm":
        output_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}" 
        model_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}/model/{config_option}"
        image_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}/test_results/{config_option}"
        save_dir = f"{paths['BASE_DIR']}/{task}/{time_shift_minutes}_{smoothing}/output_df/{config_option}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    else:
        output_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}_{smoothing}" 
        model_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}_{smoothing}/model/{config_option}/{interval_seconds}"
        image_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}_{smoothing}/test_results/lstm/{config_option}/{interval_seconds}"
        save_dir = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmin']}_{data_params['fmax']}/{time_shift_minutes}_{smoothing}/output_df/{config_option}/{interval_seconds}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    if time_shift_minutes == "average":
        julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
        date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    elif time_shift_minutes == "dynamic":
        julday_list = [172, 182, 196, 207, 223]
        date_list = ["2019-06-21", "2019-07-01", "2019-07-15", "2019-07-26", "2019-08-11"]

    if task == "abalation_study_1":
        julday_list = julday_list[:test_julday]
        date_list = date_list[:test_julday]
        test_julday, test_julday_list, test_date_list = 232, [232], ["2019-08-20"]
        val_julday, val_julday_list, val_date_list = 223, [223], ["2019-08-11"]
    else:
        test_date = date_list.pop(julday_list.index(test_julday))
        julday_list.remove(test_julday)
        val_date = date_list.pop(julday_list.index(val_julday))  
        julday_list.remove(val_julday)
        test_julday_list = [test_julday]
        test_date_list = [test_date]
        val_julday_list, val_date_list = [val_julday], [val_date]

    # LOAD DATA
    smoothing = smoothing
    print(f"{'Loading Data':-^50}")
    total_data = load_data(julday_list, station, trim=True, abs=True)
    val_data = load_data(val_julday_list, station, trim=False, abs=True)
    test_data = load_data(test_julday_list, station, trim=False, abs=True)
    st_test = load_seismic_data(test_julday, station, year=2019, trim=False)
    print(f"Data --> Train : {len(total_data)} Test : {len(test_data)}")
    total_target = load_label(date_list= date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                smoothing=smoothing)
    val_target = load_label(date_list= val_date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                smoothing=smoothing,
                                trim=False)
    test_target = load_label(date_list= test_date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                smoothing=smoothing,
                                trim=False)
    print(f"Target --> Train : {len(total_target)} Test : {len(test_target)}")
    print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")

    # INITIALIZE MODEL
    print("Initialising Model")
    with open(f"./config/{task}/lstm_{config_option}_{interval_seconds}sec_config.json", "r") as f:
        config = json.load(f)
    with open(f"{output_dir}/model_config.txt", "a") as f:
        string = f"lstm :\n{config}\n"
        f.write(string)
    model = LSTMRegressor(**config)
    criterion = nn.MSELoss()
    if interval_seconds == 1:
        lr = 5e-4
    else:
        lr = 1e-4

    batch_size = 128
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Main scheduler: Reduce on plateau after warmup ends
    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # INIT DATALOADERS
    print("Initialising Dataloaders")
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.data_min_ = np.array([0])
    # scaler.data_max_ = np.array([350])
    # scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / (scaler.data_max_ - scaler.data_min_)
    # scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
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
    # in_seq, pred_out, target_out, timestamps, time_to_train = train_model(model, criterion, optimizer,
    #                                                        100, 15, interval_seconds, test_julday, val_julday,
    #                                                        'LSTM', train_dataloader, val_dataloader,
    #                                                        test_dataloader, model_dir, scaler, warmup_scheduler, main_scheduler)
    trainer = ModelTrainer(model=model, criterion=criterion, optimizer=optimizer,
                           warmup_scheduler=warmup_scheduler, main_scheduler=main_scheduler,
                           train_loader=train_dataloader, val_loader=val_dataloader, test_loader=test_dataloader,
                           model_dir=model_dir, interval=interval_seconds,
                           test_julday=test_julday, val_julday=val_julday, model_type="LSTM", device=device)
    print(f"{'Starting Training':-^50}")
    trainer.train(num_epochs=200, patience=15)
    print(f"{'Start Testing':-^50}")
    in_seq, pred_out, target_out, timestamps, time_to_train = trainer.test()
    print(f"Saving output to {save_dir}/LSTM_t{test_julday}_v{val_julday}.csv")
    print(f"{'End Testing':-^50}")

    times = [UTCDateTime(t) for t in np.concatenate(timestamps)]
    df = pd.DataFrame(data={"Timestamps":times, "Output":np.concatenate(target_out), "Predicted_Output":np.concatenate(pred_out)})
    df.to_csv(f"{save_dir}/LSTM_t{test_julday}_v{val_julday}.csv", index=False)
    print("Making Plot")
    start_time = get_current_time()
    plot_image(st_test, pred_out, target_out, timestamps, image_dir, test_julday, val_julday, interval_seconds, trim=True, smoothing=smoothing)
    evaluate_model(model_type=f"LSTM,{config_option}", 
                   test_julday=test_julday, 
                   val_julday=val_julday, 
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
    parser.add_argument("--test_julday", type=int, default=161, help= "test julday")
    parser.add_argument("--val_julday", type=int, default=172, help= "val julday")
    parser.add_argument("--time_shift_mins", default=10, help= "enter label time shift")
    parser.add_argument("--station", type=str, default="ILL13", help= "input station")
    parser.add_argument("--interval", type=int, default=30, help= "interval seconds")
    parser.add_argument("--config_op", type=str,default="default", help= "config option")
    parser.add_argument("--task", type=str, default="comparison_baseline", help= "name of the task corresponding to parameter directory")
    parser.add_argument("--smoothing", type=int, default=30, help="enter a value used for smoothing the raw data")

    args = parser.parse_args()
    print(f"Running main with {args.test_julday} {args.station} {args.config_op} {args.task}")
    main(args.test_julday,
        args.val_julday, 
        args.time_shift_mins,
        args.smoothing, 
        args.station, 
        args.interval, 
        args.config_op, 
        args.task)