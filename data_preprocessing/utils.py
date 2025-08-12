import os
import random
import numpy as np
import pandas as pd

def fill_missing_from_Gaussian(column_val, mean, std):
    if np.isnan(column_val) == True: 
        column_val = np.abs(np.round(np.random.normal(mean, std, 1)[0], 3))
    else:
         column_val = column_val
    return column_val

def get_mean_and_std():
    min_Fvs, mean_Fv_stds = [], []
    for file in os.listdir(path= "../label/utc0_data"):
        df = pd.read_csv(f"../label/utc0_data/{file}")
        min_Fvs.append(np.min(df['Fv [kN]']))
        mean_Fv_stds.append(np.mean(df['Fv std']))
    min_Fv = np.min(min_Fvs)
    min_Fv_std = np.min(mean_Fv_stds)
   
    return min_Fv, min_Fv_std
