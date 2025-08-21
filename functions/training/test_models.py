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
from obspy import UTCDateTime

from functions.data_processing.read_data import load_data, load_seismic_data
from functions.data_processing.dataloader import SequenceDatasetTest, DataLoader

from functions.utils import *
# from functions.train import train_model
# from functions.evaluation import evaluate_model
from functions.evaluation.plot_image import plot_image_test
from models.LSTM_model import LSTMRegressor
from models.xLSTM_model import xLSTMRegressor_v2

def load_model(model_julday:int, model_type:str, interval:int):
    mapping = {161 : 1, 172 : 2, 196 : 3, 207 : 4, 223 : 5, 232 : 6}
    if model_type == 'LSTM':
        with open(f"./config/comparison_baseline/lstm_default_{interval}sec_config.json", "r") as f:
            config = json.load(f)
        model = LSTMRegressor(**config)
    elif model_type == 'xLSTM':
        with open(f"./config/comparison_baseline/xlstm_default_{interval}sec_config.json", "r") as f:
            config = json.load(f)
        model = xLSTMRegressor_v2(**config)
    else:
        print(f"Wrong model type entered : {model_type}!")
        exit()
    model.load_state_dict(torch.load(f=f"{paths['SAVED_MODEL_DIR']}/{mapping[model_julday]}/{interval}_{model_type}.pt", weights_only=True))
    return model

def main(network:str, station:str, component:str, year:int, julday:int, model_type:str, interval_seconds:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    julday = str(julday).zfill(3)
    num_intervals = int((5 * 60) // interval_seconds)

    mapping = {161 : 1, 172 : 2, 196 : 3, 207 : 4, 223 : 5, 232 : 6}
    
    # LOAD DATA
    print("\tLoading Data")
    st = load_seismic_data(julday= julday, station= station, raw= False, year= year, component= component,
                           network= network, trim=False)
    # data = np.abs(st[0].data[1:])
    data = load_data(julday_list = [julday], station=station, year=year, trim=False, abs=True)
    timestamps = [UTCDateTime(UTCDateTime(year=year, julday=int(julday)) + i).timestamp for i in range(data_params['time_window'] * 60, 60 * 60 * 24, interval_seconds)]

    # PREPARE DATALOADER
    print("\tPreparing Dataloader")
    dataset = SequenceDatasetTest(input_data= data, target_time= timestamps, 
                                interval_count= num_intervals, sequence_length= interval_seconds * 100)
    dataloader = DataLoader(dataset= dataset, batch_size= 8, shuffle=False)

    for model_julday in [161, 172, 196, 207, 223, 232]:
        output_dir = f"./model_test/{model_type}_{interval_seconds}/{year}/{mapping[model_julday]}/"
        output_file_dir = f"{output_dir}/df"
        output_img_dir = f"{output_dir}/img"
        os.makedirs(output_file_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)
        # LOAD MODEL
        model = load_model(model_julday, model_type, interval_seconds)

        # APPLY MODEL
        start_time = get_current_time()
        print(f"{'Start Testing':-^50}")
        # model.eval()
        in_sequence, predicted_output, model_timestamps = [], [], []
        test_epoch_loss = 0.0
        with torch.no_grad():
            for input_sequences, test_timestamps in dataloader:
                model.to(device)
                # Move data to the appropriate device if using a GPU
                input_sequences = input_sequences.float().to(device)  # Shape: (batch_size, 20, 3000)

                # Forward pass: Get model predictions
                output = model(input_sequences).squeeze(1)  # Shape: (batch_size, 1)
                # Squeeze the output to match target shape
                in_sequence.append(input_sequences.cpu().numpy())
                predicted_output.append(output.detach().cpu().numpy() * 350)
                model_timestamps.append(test_timestamps)
        end_time = get_current_time()
        time_to_test = get_time_elapsed(start_time, end_time)
        print(f"{f'End Testing : {str(timedelta(seconds=time_to_test))}':-^50}")

        # SAVE OUTPUT
        times = [UTCDateTime(t) for t in timestamps]
        assert len(times) == len(np.concatenate(predicted_output)), f"Length mismatch for saving output data {len(times)} != {len(np.concatenate(predicted_output))}"
        df = pd.DataFrame(data={"Timestamps":times, "Predicted_Output":np.concatenate(predicted_output)})
        df.to_csv(f"{output_file_dir}/{julday}.csv", index=False)

        # PLOT IMAGE
        plot_image_test(st=st, predicted_output= predicted_output, timestamps= model_timestamps, 
                        image_dir= output_img_dir, julday= julday, interval= interval_seconds)
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default='9S', help= "Seismic Network")
    parser.add_argument("--station", type=str, default="ILL13", help= "Input station")
    parser.add_argument("--component", type=str, default="EHZ", help= "Seismic signal component")
    parser.add_argument("--year", type=int, default=2019, help= "Enter year of testing seismic data")
    parser.add_argument("--julday", type=int, default=161, help= "Enter julday of seismic data")
    parser.add_argument("--interval", type=int, default=30, help= "interval seconds")
    parser.add_argument("--model_type", type=str, default='xLSTM', help= "Enter the model type to be tested")

    args = parser.parse_args()
    main(network= args.network, station= args.station, component= args.component, 
         year= args.year, julday= args.julday, model_type= args.model_type, interval_seconds= args.interval)