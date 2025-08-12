import os
import sys
import json
import copy
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
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from obspy import UTCDateTime
from sklearn.preprocessing import MinMaxScaler

from functions.utils import *
from functions.data_processing.read_data import load_data, load_label, load_seismic_data
from functions.data_processing.dataloader import SequenceDataset, DataLoader
from functions.evaluation.plot_image import plot_image

import torch.nn.functional as F

from models.xLSTM_model import xLSTMRegressor

import optuna

def define_model(trial):
    n_repeats = trial.suggest_int("n_repeats", 1, 3, step=1)
    kernel_size = trial.suggest_int("conv1d_size", 4, 24, step=4)
    qkv_blocksize = trial.suggest_int("qkv_size", 4, 24, step=4)
    num_heads = trial.suggest_int("num_heads", 2, 16, step=2)
    emb_dim = trial.suggest_int("embedding_dimension", 16, 128, step=16)

    if (emb_dim % num_heads != 0):
        print(f"Invalid combination of embedding dimension ({emb_dim}) and number of heads ({num_heads}).")
        raise optuna.TrialPruned()  # Skip invalid combinations

    print(f"n_repeats: {n_repeats}, conv1d_size: {kernel_size}, qkv_size: {qkv_blocksize}, num_heads: {num_heads}, embedding_dimension: {emb_dim}")

    try:
        model  = xLSTMRegressor(input_size=500, 
                          conv1d_kernel_size= kernel_size,
                          qkv_proj_blocksize=qkv_blocksize,
                          num_heads=num_heads,
                          context_length=60,
                          num_blocks=8,
                          embedding_dim=emb_dim,
                          slstm_at=[1],
                          num_reps=n_repeats
                          )
    except AssertionError:
        print(f"Invalid combination of embedding dimension ({emb_dim}) and number of heads ({num_heads}).")
        raise optuna.TrialPruned()
    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory error with embedding dimension ({emb_dim}) and number of heads ({num_heads}).")
        raise optuna.TrialPruned()
    return model

def objective(trial):
    print(f"{f'Trial Number {trial.number}':-^100}")
    output_dir = f"{paths['BASE_DIR']}/xlstm_tuning_2/"
    os.makedirs(output_dir, exist_ok=True)
    model_dir = f"{paths['BASE_DIR']}/xlstm_tuning_2/model/"
    os.makedirs(model_dir, exist_ok=True)
    image_dir = f"{paths['BASE_DIR']}/xlstm_tuning_2/images/trial_{trial.number}/"
    os.makedirs(image_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the data
    test_julday = 232
    # val_julday_list = [161, 172]
    val_julday_list = [207, 223]
    # val_julday_list = [183, 196]
    savestr = "".join([str(i) for i in val_julday_list])
    
    station = 'ILL11'
    interval_seconds = 5
    time_shift_minutes = 'dynamic'
    num_intervals = int((5 * 60) // interval_seconds)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    print(f"Batch Size: {batch_size}")
    
    julday_list = [161, 172, 182, 183, 196, 207, 223, 232]
    date_list = ["2019-06-10", "2019-06-21", "2019-07-01", "2019-07-02", "2019-07-15", "2019-07-26", "2019-08-11", "2019-08-20"]
    
    test_date = date_list.pop(julday_list.index(test_julday))
    julday_list.remove(test_julday)
    val_date_list = []
    for val_julday in val_julday_list:
        val_date = date_list.pop(julday_list.index(val_julday))  
        julday_list.remove(val_julday)
        val_date_list.append(val_date)
    test_julday_list = [test_julday]
    test_date_list = [test_date]
    # val_julday_list = [val_julday]

    # LOAD DATA
    print(f"{'Loading Data':-^50}")
    total_data = load_data(julday_list, station, trim=True, abs=True)
    val_data = load_data(val_julday_list, station, trim=True, abs=True)
    test_data = load_data(test_julday_list, station, trim=False, abs=True)
    st_test = load_seismic_data(test_julday, station, trim=False)
    print(f"Data --> Train : {len(total_data)} Test : {len(test_data)}")
    total_target = load_label(date_list= date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes)
    val_target = load_label(date_list= val_date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes)
    test_target = load_label(date_list= test_date_list, station= station, 
                                interval_seconds= interval_seconds,
                                time_shift_minutes= time_shift_minutes,
                                trim=False)
    print(f"Target --> Train : {len(total_target)} Test : {len(test_target)}")
    print(f"RAM usage = {get_memory_usage_in_gb():.2f} GB")
    # Define the model
    model = define_model(trial).to(device)

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


    # Define the loss function and optimizer
    lr = trial.suggest_categorical("lr", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    criterion = trial.suggest_categorical("criterion", ["MSE", "MAE"])
    if criterion == "MSE":
        criterion = nn.MSELoss()
    elif criterion == "MAE":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Invalid criterion selected")
    print(f"Learning Rate: {lr}, Weight Decay: {weight_decay}, Criterion: {criterion}")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    print(f"{'Starting Training':-^50}")
    best_loss = float('inf')
    consecutive_increase = 0
    try:
        for epoch in range(50):
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            # prev_target = torch.zeros(get_batch_size(interval))
            for input_sequences, target_value, _ in train_dataloader:
                if input_sequences.dim() == 2:
                    continue
                model.to(device)
                input_sequences = input_sequences.float().to(device)  
                # prev_target = prev_target.float().to(device)
                target_value = target_value.float().to(device)        
                
                optimizer.zero_grad()
                output = model(input_sequences).squeeze(1)  # Forward pass and shape adjustment
                assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
                loss = criterion(output, target_value)     # Compute loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)  # Average loss for the epoch
            print(f"Epoch [{epoch+1}/50], Loss: {train_loss:.4f}")

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                for input_sequences, target_value, _ in val_dataloader:
                    if input_sequences.dim() == 2:
                        continue
                    model.to(device)
                    input_sequences = input_sequences.float().to(device) 
                    # prev_target = prev_target.float().to(device)
                    target_value = target_value.float().to(device)
                    # Forward pass: Get model predictions
                    output = model(input_sequences).squeeze(1)  
                    assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
                    loss = criterion(output, target_value) 
                    val_loss += loss.item()
                    # prev_target = output.cpu().detach()
            val_loss /= len(val_dataloader)
            print(f"Epoch [{epoch+1}/50], Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())  # Save the best weights
                consecutive_increase = 0  # Reset counter
            else:
                consecutive_increase += 1  # Increment counter
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if consecutive_increase > 5:  # Early stopping
                # Restore the best weights
                model.load_state_dict(best_weights)
                print(f"Early stopping triggered. Restoring model to epoch {best_epoch + 1} with lowest loss {best_loss:.4f}")
                break 
    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory error.")
        raise optuna.TrialPruned()
    except RuntimeError:
        print(f"Runtime error.")
        raise optuna.TrialPruned()
    
    model.load_state_dict(best_weights)
    print(f"Best Validation Loss: {best_loss:.4f}")
    model.load_state_dict(best_weights)  # Restore the best weights
    # Save the model
    model_path = f"{model_dir}model_{trial.number}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Test model
    start_time = get_current_time()
    print(f"{'Start Testing':-^50}")
    model.eval()
    in_sequence, predicted_output, target_output, timestamps = [], [], [], []
    test_epoch_loss = 0.0
    with torch.no_grad():
        # prev_target = torch.zeros(get_batch_size(interval))
        for input_sequences, target_value, test_timestamps in test_dataloader:
            if input_sequences.dim() == 2:
                continue
            model.to(device)
            # Move data to the appropriate device if using a GPU
            input_sequences = input_sequences.float().to(device)  # Shape: (batch_size, 20, 3000)
            # prev_target = prev_target.float().to(device)
            target_value = target_value.float().to(device)        # Shape: (batch_size,)

            # Forward pass: Get model predictions
            output = model(input_sequences).squeeze(1)  # Shape: (batch_size, 1)
            assert output.shape == target_value.shape, f"Shape mismatch {output.shape} <-> {target_value.shape}"
            loss = criterion(output, target_value) 
            test_epoch_loss += loss.item()
            # prev_target = output.cpu().detach()
            # Squeeze the output to match target shape
            if scaler is None:
                in_sequence.append(input_sequences.cpu().numpy())
                predicted_output.append(output.detach().cpu().numpy())
                target_output.append(target_value.cpu().numpy())
                timestamps.append(test_timestamps)
            else:
                in_sequence.append(input_sequences.cpu().numpy())
                rem = output.cpu().detach().cpu().numpy().shape
                predicted_output.append(scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1,1)).reshape(rem))
                target_output.append(scaler.inverse_transform(target_value.cpu().numpy().reshape(-1,1)).reshape(rem))
                timestamps.append(test_timestamps)
    print(f"Test loss : {test_epoch_loss / len(test_dataloader)}")
    with open(f"{output_dir}test_loss.txt", "a") as f:
        f.write(f"Trial {trial.number} - Test Loss: {test_epoch_loss / len(test_dataloader)}\n")
    end_time = get_current_time()
    time_to_test = get_time_elapsed(start_time, end_time)
    print(f"{'End Testing':-^50}")
    plot_image(st=st_test, predicted_output=predicted_output, target_output=target_output, 
               timestamps=timestamps, image_dir=image_dir, 
               test_julday=232, 
               val_julday=savestr, interval=interval_seconds)

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///xlstm_tuning.db",
        study_name="xlstm_tuning_2",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")
    print("  System attrs:")
    for key, value in trial.system_attrs.items():
        print(f"    {key}: {value}")
    print("  Intermediate values:")
    for key, value in trial.intermediate_values.items():
        print(f"    {key}: {value}")
    # print("  Pruned: ", trial.pruned)
    # print("  State: ", trial.state)
    # print("  Duration: ", trial.duration)
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])))
    print("  Number of complete trials: ", len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])))
    print("  Number of failed trials: ", len(study.get_trials(states=[optuna.trial.TrialState.FAIL])))
    
    