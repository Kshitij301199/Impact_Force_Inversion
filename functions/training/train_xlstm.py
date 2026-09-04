#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-30
# __author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
# __find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
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
# data_params = load_config("data_parameters.json")
time_config = load_config("event_id_map.json")

# Set CUDA environment variables
if "CUDA_HOME" in paths:
    os.environ["CUDA_HOME"] = paths["CUDA_HOME"]
    os.environ["PATH"] = os.path.join(os.environ["CUDA_HOME"], "bin") + ":" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        os.path.join(os.environ["CUDA_HOME"], "lib64") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

sys.path.append(paths["BASE_DIR"])
import torch

torch.set_default_dtype(torch.float32)
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from obspy import UTCDateTime
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torchinfo import summary

from functions.data_processing.read_data import load_data, load_label, load_seismic_data
from functions.data_processing.dataloader import SequenceDataset, DataLoader, concat_sequence_datasets

from functions.utils import *
from functions.training.train import ModelTrainer
from functions.evaluation.eval import evaluate_model
from functions.evaluation.plot_image import plot_image

from models.xLSTM_model import xLSTMRegressor_v2


def make_warmup_lambda(warmup_epochs):
    def fn(epoch):
        return min(1.0, (epoch + 1) / max(1, warmup_epochs))

    return fn


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dirs(
    task: str, time_shift_minutes, smoothing, divide_by, interval_seconds, config_option, repeat, data_params
):
    """Generic directory creation."""
    base_output = f"{paths['BASE_DIR']}/{task}_{data_params['time_window']}_{data_params['fmax']}_{repeat}"
    params_subdir = f"{time_shift_minutes}_{smoothing}_{divide_by}"
    output_dir = os.path.join(base_output, params_subdir)

    model_dir = os.path.join(output_dir, "model", config_option, str(interval_seconds))
    image_dir = os.path.join(output_dir, "test_results", "xlstm", config_option, str(interval_seconds))
    save_dir = os.path.join(output_dir, "output_df", config_option, str(interval_seconds))
    curve_dir = os.path.join(output_dir, "loss_curves", config_option, str(interval_seconds))

    for d in [model_dir, image_dir, save_dir, curve_dir]:
        os.makedirs(d, exist_ok=True)

    return output_dir, model_dir, image_dir, save_dir, curve_dir


def main(
    test_id: str,
    val_id: str,
    time_shift_minutes: int | str,
    smoothing: int,
    divide_by: int,
    station: str,
    interval_seconds: int,
    config_option: str,
    task: str,
    args,
    repeat=1,
):
    """Main function to train and evaluate the xLSTM model."""
    test_id, val_id = str(test_id), str(val_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    data_params = load_config(f"{task}/data_parameters.json")

    num_intervals = int((data_params["time_window"] * 60) // interval_seconds)
    output_dir, model_dir, image_dir, save_dir, curve_dir = make_dirs(
        task, time_shift_minutes, smoothing, divide_by, interval_seconds, config_option, repeat, data_params
    )

    # Standardized event list logic
    event_id_list = ["1", "3", "4", "5", "6", "7", "8", "9"]
    # if time_shift_minutes not in ["average", "dynamic"]:
    #     if "5" in event_id_list: event_id_list.remove("5")

    test_info = time_config[str(test_id)]
    val_info = time_config[str(val_id)]
    test_julday = test_info["julday"] if isinstance(test_info["julday"], int) else test_info["julday"][0]
    val_julday = val_info["julday"] if isinstance(val_info["julday"], int) else val_info["julday"][0]

    train_id_list = [eid for eid in event_id_list if eid not in [test_id, val_id]]
    if test_id == val_id and test_id in event_id_list:
        train_id_list = [eid for eid in event_id_list if eid != test_id]

    curve_file = f"{curve_dir}/xLSTM_t{test_julday}_v{val_julday}.txt"
    with open(curve_file, "w") as file:
        file.write("Epoch;Train_Loss;Val_Loss;LR;mean_g;max_g\n")

    # LOAD DATA
    train_juldays = [time_config[eid]["julday"] for eid in train_id_list]
    # Flatten if necessary
    train_juldays = [
        j if isinstance(j, int) else item for j in train_juldays for item in (j if isinstance(j, list) else [j])
    ]

    print(f"Train Day List : {train_juldays}, Val Day List : {val_julday}, Test Day List : {test_julday}")

    print(f"{'Loading Data and Making Dataloader':-^50}")
    st_test = load_seismic_data(test_id, station, year=2019, trim=True)
    train_datasets = []
    for eid in train_id_list:
        data_eid, _ = load_data([eid], station, trim=True, abs=True)
        target_eid = load_label(
            event_id=eid,
            station=station,
            interval_seconds=interval_seconds,
            time_shift_minutes=time_shift_minutes,
            smoothing=smoothing,
            divide_by=divide_by,
        )
        ds = SequenceDataset(
            data_eid,
            target_eid["Fv [kN]"].to_numpy(),
            target_eid["Timestamp"].to_numpy(),
            interval_count=num_intervals,
            sequence_length=interval_seconds * 100,
        )
        train_datasets.append(ds)
    full_train_dataset = concat_sequence_datasets(train_datasets)
    val_datasets = []
    for eid in [val_id]:
        data_eid, _ = load_data([eid], station, trim=True, abs=True, val=True)
        target_eid = load_label(
            event_id=eid,
            station=station,
            interval_seconds=interval_seconds,
            time_shift_minutes=time_shift_minutes,
            smoothing=smoothing,
            divide_by=divide_by,
            val=True,
        )
        ds = SequenceDataset(
            data_eid,
            target_eid["Fv [kN]"].to_numpy(),
            target_eid["Timestamp"].to_numpy(),
            interval_count=num_intervals,
            sequence_length=interval_seconds * 100,
        )
        val_datasets.append(ds)
    full_val_dataset = concat_sequence_datasets(val_datasets)
    test_datasets = []
    for eid in [test_id]:
        data_eid, _ = load_data([eid], station, trim=True, abs=True)
        target_eid = load_label(
            event_id=eid,
            station=station,
            interval_seconds=interval_seconds,
            time_shift_minutes=time_shift_minutes,
            smoothing=smoothing,
            divide_by=divide_by,
        )
        ds = SequenceDataset(
            data_eid,
            target_eid["Fv [kN]"].to_numpy(),
            target_eid["Timestamp"].to_numpy(),
            interval_count=num_intervals,
            sequence_length=interval_seconds * 100,
        )
        test_datasets.append(ds)
    full_test_dataset = concat_sequence_datasets(test_datasets)
    batch_size = 256
    train_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(full_val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Target --> Train : {len(full_train_dataset)} Test : {len(full_test_dataset)}")
    print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")

    # INITIALIZE MODEL
    print("Initialising Model")
    if task.startswith("lr_test"):
        config_path = project_root / "config" / "lr_test" / f"xlstm_{config_option}_{interval_seconds}sec_config.json"
    else:
        config_path = project_root / "config" / task / f"xlstm_{config_option}_{interval_seconds}sec_config.json"
    with open(config_path, "r") as f:
        model_params = json.load(f)

    with open(f"{output_dir}/model_config.txt", "a") as f:
        f.write(f"xlstm :\n{model_params}\n")

    model = xLSTMRegressor_v2(**model_params)
    summary(
        model=model,
        input_size=(128, num_intervals, interval_seconds * 100),
        col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"],
        col_width=15,
        row_settings=["var_names"],
        depth=7,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model.__class__.__name__} trainable parameters: {n_params:,}")
    with open(f"{output_dir}/model_config.txt", "a") as f:
        f.write(f"trainable_params: {n_params}\n")

    criterion = nn.MSELoss()

    lr = args.lr if args.lr is not None else 5e-5  # keep old default as fallback only

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    if args.warmup_epochs > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=make_warmup_lambda(args.warmup_epochs))
    else:
        warmup_scheduler = None

    main_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.50, patience=10)

    print("Training Model")
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        warmup_scheduler=warmup_scheduler,
        main_scheduler=main_scheduler,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        model_dir=model_dir,
        curve_file=curve_file,
        interval=interval_seconds,
        test_julday=test_julday,
        val_julday=val_julday,
        model_type="xLSTM",
        device=device,
        grad_clip=args.grad_clip,
    )
    print(f"{'Starting Training':-^50}")
    trainer.train(num_epochs=200, patience=15)
    print(f"{'End Training':-^50}")

    print(f"{'Start Validation':-^50}")
    in_seq, pred_out, target_out, timestamps, time_to_train = trainer.val(mult_by=divide_by)
    print(f"Saving output to {save_dir}/xLSTM_t{test_julday}_v{val_julday}.csv")
    times = [UTCDateTime(t) for t in np.concatenate(timestamps)]
    df = pd.DataFrame(
        data={"Timestamps": times, "Output": np.concatenate(target_out), "Predicted_Output": np.concatenate(pred_out)}
    )
    df.to_csv(f"{save_dir}/xLSTM_t{test_julday}_v{val_julday}_val.csv", index=False)
    print(f"{'End Validation':-^50}")

    print(f"{'Start Testing':-^50}")
    in_seq, pred_out, target_out, timestamps, time_to_train = trainer.test(mult_by=divide_by)
    print(f"Saving output to {save_dir}/xLSTM_t{test_julday}_v{val_julday}.csv")
    times = [UTCDateTime(t) for t in np.concatenate(timestamps)]
    df = pd.DataFrame(
        data={"Timestamps": times, "Output": np.concatenate(target_out), "Predicted_Output": np.concatenate(pred_out)}
    )
    df.to_csv(f"{save_dir}/xLSTM_t{test_julday}_v{val_julday}.csv", index=False)
    print(f"{'End Testing':-^50}")

    print("Making Plot")
    start_time = get_current_time()
    plot_image(
        st_test,
        pred_out,
        target_out,
        timestamps,
        image_dir,
        test_id,
        val_id,
        time_shift_minutes,
        interval_seconds,
        trim=True,
        smoothing=smoothing,
    )
    evaluate_model(
        model_type=f"xLSTM,{config_option}",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_event_id", type=str, default="1", help="test julday")
    parser.add_argument("--val_event_id", type=str, default="3", help="val julday")
    parser.add_argument("--time_shift_mins", default="average", help="enter label time shift")
    parser.add_argument("--station", type=str, default="ILL11", help="input station")
    parser.add_argument("--interval", type=int, default=5, help="interval seconds")
    parser.add_argument("--config_op", type=str, default="v4", help="config option")
    parser.add_argument(
        "--task",
        type=str,
        default="comparison_baseline_cv",
        help="name of the task corresponding to parameter directory",
    )
    parser.add_argument("--smoothing", type=int, default=60, help="enter a value used for smoothing the raw data")
    parser.add_argument("--divide_by", type=int, default=45, help="normalization constant")
    parser.add_argument("--repeat", type=int, default=1, help="Number to times to repeat process")
    parser.add_argument("--lr", type=float, default=None, help="peak learning rate; overrides script default if set")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="0 disables warmup scheduler")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="max grad norm, 0 disables clipping")

    args = parser.parse_args()
    print(f"Running main with {args.test_event_id} {args.station} {args.config_op} {args.task}")
    for repeat in range(1, args.repeat + 1):
        main(
            args.test_event_id,
            args.val_event_id,
            args.time_shift_mins,
            args.smoothing,
            args.divide_by,
            args.station,
            args.interval,
            args.config_op,
            args.task,
            args,
            repeat,
        )
