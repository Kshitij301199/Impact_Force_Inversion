import os
import json
try:
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
        paths = json.load(file)
except FileNotFoundError:
    with open("../config/paths.json", "r") as file:
        paths = json.load(file)
import pandas as pd
import numpy as np

from obspy.core import UTCDateTime # default is UTC+0 time zone

def shift_utc():
    output_dir = f"{paths["LOCAL_BASE_DIR"]}/{paths['UTC0_LABEL_DIR']}"
    input_dir = f"{paths["LOCAL_BASE_DIR"]}/{paths['RAW_LABEL_DIR']}/"
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        data = pd.read_csv(f"{input_dir}/{file}")
        data.rename(columns={"Unnamed: 0":"Time UTC+1"}, inplace=True)
        data["Time UTC+0"] = data['Time UTC+1'].apply(UTCDateTime) - 3600
        data.to_csv(f"{output_dir}{file}")

if __name__ == "__main__":
    shift_utc()
