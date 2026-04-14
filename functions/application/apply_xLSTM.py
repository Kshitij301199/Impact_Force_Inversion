#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2026-02-01
#__author__ = Kshitij Kar, GFZ Helmholtz Centre for Geosciences
#__find me__ = kshitij.kar@gfz.de, kshitij787.ak@gmail.com, https://github.com/Kshitij301199
# Please do not distribute this code without the author's permission

import os
import json
from typing import List

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

from functions.data_processing.read_data import load_data_test, load_seismic_data_test
from functions.data_processing.dataloader import SequenceDatasetTest, DataLoader

from functions.utils import *
from functions.evaluation.plot_image import plot_image_test
from models.xLSTM_model import xLSTMRegressor_v2

def load_models(sub_interval) -> List:
    model_filename = f"{sub_interval}_xLSTM.pt"
    model_list = []
    for i in range(1,9):
        with open(f"./config/comparison_baseline_cv/xlstm_v5_{sub_interval}sec_config.json", "r") as f:
            config = json.load(f)
        model = xLSTMRegressor_v2(**config)
        model.load_state_dict(torch.load(f=f"{paths['SAVED_MODEL_DIR']}/v5/{i}/{model_filename}", weights_only=True))
        model_list.append(model)
    return model_list

def apply_model(input_sequences, timestamps, model_list):
    x_vals = []
    means = []
    mins = []
    maxs = []
    stds = []
    scaling_factor = 45
    with torch.no_grad():
        for i in range(input_sequences.shape[0]):
            outputs_i = []
            for model in model_list:
                out = model(input_sequences[i,:,:].unsqueeze(0)).squeeze()
                outputs_i.append(float(out.item() * scaling_factor))
            means.append(np.mean(outputs_i))
            mins.append(np.min(outputs_i))
            maxs.append(np.max(outputs_i))
            stds.append(np.std(outputs_i))
            x_vals.append(timestamps[i].item())

    x_vals = np.array(x_vals)
    means = np.array(means)
    mins = np.array(mins)
    maxs = np.array(maxs)
    stds = np.array(stds)
    return x_vals, means, mins, maxs, stds

