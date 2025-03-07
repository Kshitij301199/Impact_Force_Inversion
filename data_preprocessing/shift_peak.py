import os
import sys
import json
try:
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
        paths = json.load(file)
except FileNotFoundError:
    with open("../config/paths.json", "r") as file:
        paths = json.load(file)
import numpy as np
import pandas as pd
from obspy.core import UTCDateTime
from tqdm import tqdm

from utils import *
sys.path.append("..")
from data_processing.read_data import load_seismic_data

def get_mean_and_std():
    min_Fvs, min_Fv_stds = [], []
    for file in os.listdir(path= f"{paths['LOCAL_BASE_DIR']}/{paths['UTC0_LABEL_DIR']}"):
        df = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['UTC0_LABEL_DIR']}/{file}")
        min_Fvs.append(np.min(df['Fv [kN]']))
        min_Fv_stds.append(np.min(df['Fv std']))
    min_Fv = np.min(min_Fvs)
    min_Fv_std = np.mean(min_Fv_stds)
    print(f"Minimum Impact Force : {min_Fv:.2f}kN, \nMinimum Standard Deviation : {min_Fv_std:.2f}kN")
    return min_Fv, min_Fv_std

def main(station, time_shift_minutes):
    input_dir = f"{paths['LOCAL_BASE_DIR']}/{paths['UTC0_LABEL_DIR']}"
    output_dir = f"{paths['LOCAL_BASE_DIR']}/label/data_processed_{time_shift_minutes}/{station}"
    os.makedirs(output_dir, exist_ok=True)
    event_data = pd.read_csv("2019-event-times.csv", index_col=0)
    min_Fv, min_Fv_std = get_mean_and_std()
    
    # Unique Cases 1
    print("Running Unique Case 1")
    start_date = "2019-06-10"
    end_date = "2019-06-11"
    df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(start_date),
                                                                                        UTCDateTime(end_date))][f'{station}Start']
    df_start_times = df_start_times.apply(UTCDateTime)
    data1 = pd.read_csv(f"{input_dir}/20190610_Fv_1.csv", index_col=0)
    data2 = pd.read_csv(f"{input_dir}/20190610_Fv_2.csv", index_col=0)
    data_start_time1, data_start_time2 = UTCDateTime(data1.iloc[0,-1]), UTCDateTime(data2.iloc[0,-1])
    # print(f"{data_start_time1.datetime} \n{data_start_time2.datetime}")
    time_diff1, time_diff2 = data_start_time1 - df_start_times.iloc[0], data_start_time2 - df_start_times.iloc[1]

    data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (6 * 60) + (time_shift_minutes * 60)
    data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (1 * 60) + (time_shift_minutes * 60)
    data = pd.concat([data1, data2])
    complete_data = pd.DataFrame(columns= ['Time'])
    start_time = UTCDateTime("2019-06-10T00:00:00")
    time_list1 = []
    for i in range(0, 1440 * 60 + 1):
        time_list1.append(start_time + i)
    complete_data['Time'] = time_list1
    merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
    merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
    merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": 3*min_Fv_std})
    merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
    merged_data.to_csv(f"{output_dir}/{start_date}.csv", index=False)

    # Unique Cases 2
    print("Running Unique Case 2")
    DataStart = "2019-07-01"
    DataEnd = "2019-07-04"
    df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(DataStart), UTCDateTime(DataEnd))][f'{station}Start']
    df_start_times = df_start_times.apply(UTCDateTime)
    data1 = pd.read_csv(f"{input_dir}/20190702_Fv.csv", index_col=0)
    data2 = pd.read_csv(f"{input_dir}/20190703_Fv.csv", index_col=0)
    data_start_time1, data_start_time2 = UTCDateTime(data1.iloc[0,-1]), UTCDateTime(data2.iloc[0,-1])
    # print(f"{data_start_time1.datetime} \n{data_start_time2.datetime}")
    time_diff1, time_diff2 = data_start_time1 - df_start_times.iloc[0], data_start_time2 - df_start_times.iloc[1]
    data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (61 * 60) + (time_shift_minutes * 60)
    data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (68 * 60) + (time_shift_minutes * 60)

    data = pd.concat([data1, data2])
    complete_data = pd.DataFrame(columns= ['Time'])
    start_time = UTCDateTime("2019-07-01T00:00:00")
    time_list1 = []
    for i in range(0, (3 * 1440 * 60) + 1):
        time_list1.append(start_time + i)
    complete_data['Time'] = time_list1
    merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
    merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
    merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": 3*min_Fv_std})
    merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
    merged_data[merged_data['Time'].between(UTCDateTime('2019-07-01'), UTCDateTime('2019-07-02'))].to_csv(f"{output_dir}/2019-07-01.csv", index=False)
    merged_data[merged_data['Time'].between(UTCDateTime('2019-07-02'), UTCDateTime('2019-07-03'))].to_csv(f"{output_dir}/2019-07-02.csv", index=False)
    merged_data[merged_data['Time'].between(UTCDateTime('2019-07-03'), UTCDateTime('2019-07-04'))].to_csv(f"{output_dir}/2019-07-03.csv", index=False)

    # General Cases
    print("Running General Case")
    # peak_shift_df = pd.read_csv("peak_times.csv", index_col=False)
    date_starts = ["2019-06-21","2019-07-15","2019-07-26","2019-08-11","2019-08-20"]
    date_ends = ["2019-06-22","2019-07-16","2019-07-27","2019-08-12","2019-08-21"]
    shift_values = [time_shift_minutes] * 5
    peak_diff_values = [59, 69, 61, 57, 62]
    for date_start, date_end, shift, peak_diff in tqdm(zip(date_starts, date_ends, shift_values, peak_diff_values), total=5):
        DataStart = date_start
        DataEnd = date_end
        df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(DataStart), UTCDateTime(DataEnd))][f'{station}Start']
        df_start_times = df_start_times.apply(UTCDateTime)
        data = pd.read_csv(f"{input_dir}/{date_start.replace('-','')}_Fv.csv", index_col=0)
        data_start_time1 = UTCDateTime(data.iloc[0,-1])
        # print(data_start_time1.datetime)
        time_diff1 = data_start_time1 - df_start_times.iloc[0] 
        data['Time'] = data.iloc[:,-1].apply(UTCDateTime) - (peak_diff * 60) + (shift * 60)
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime(date_start)
        time_list1 = []
        for i in range(0, 1440 * 60 + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on= 'Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": 3*min_Fv_std})
        merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
        merged_data.to_csv(f"{output_dir}/{date_start}.csv", index=False)

    return None

if __name__ == "__main__":
    for interval in [0, 5, 10]:
        main("ILL11", interval)







