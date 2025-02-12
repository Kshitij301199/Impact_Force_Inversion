import random
import numpy as np

def fill_missing_from_Gaussian(column_val, mean, std):
    if np.isnan(column_val) == True: 
        column_val = np.round(np.random.normal(mean, std, 1)[0], 3)
    else:
         column_val = column_val
    return column_val