import os
import sys
import json
import argparse
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
from functions.data_processing.read_data import load_seismic_data


def main(time_shift_minutes, from_velocity:bool=False, avg_peak_shift:bool=False):
    station = "ILL11"
    input_dir = f"{paths['LOCAL_BASE_DIR']}/{paths['UTC0_LABEL_DIR']}"
    velocity_dir = f"{paths['LOCAL_BASE_DIR']}/label"
    event_data = pd.read_csv("2019-event-times.csv", index_col=0)
    min_Fv, min_Fv_std = get_mean_and_std()
    # min_Fv, min_Fv_std = 0.5, 0.01
    sigma = 5
    print(f"Minimum Impact Force : {min_Fv:.2f}kN, \nMinimum Standard Deviation : {min_Fv_std:.2f}kN, Used Standard Deviation : {sigma * min_Fv_std:.2f}kN")
    
    if not from_velocity:
        output_dir = f"{paths['LOCAL_BASE_DIR']}/label/data_processed_{time_shift_minutes}/{station}"
        os.makedirs(output_dir, exist_ok=True)
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

        data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (5.2 * 60) + (time_shift_minutes * 60)
        data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (4.9 * 60) + (time_shift_minutes * 60)
        data = pd.concat([data1, data2])
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime("2019-06-10T00:00:00")
        time_list1 = []
        for i in range(0, 1440 * 60 + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
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
        data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (61.2 * 60) + (time_shift_minutes * 60)
        data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (68.1 * 60) + (time_shift_minutes * 60)

        data = pd.concat([data1, data2])
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime("2019-07-01T00:00:00")
        time_list1 = []
        for i in range(0, (3 * 1440 * 60) + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
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
        # peak_diff_values = [59, 69, 61, 57, 62] # OLD
        peak_diff_values = [61.2, 69.5, 61.3, 57.5, 62.4]

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
            merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
            merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
            merged_data.to_csv(f"{output_dir}/{date_start}.csv", index=False)
    elif from_velocity and not avg_peak_shift:
        output_dir = f"{paths['LOCAL_BASE_DIR']}/label/data_processed_dynamic/{station}"
        os.makedirs(output_dir, exist_ok=True)
        distance = 450 # in meters between ILL11 and Force Plate
        vel_df = pd.read_csv(f"{velocity_dir}/DF_characteristics.csv", index_col=False)
        vel_df['Event_Date'] = vel_df['Event_Date'].apply(lambda x: x.split('T')[0])
        print("Running Unique Case 1")
        start_date = "2019-06-10"
        end_date = "2019-06-11"
        temp = vel_df[vel_df['Event_Date'] == start_date]
        df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(start_date),
                                                                                            UTCDateTime(end_date))][f'{station}Start']
        df_start_times = df_start_times.apply(UTCDateTime)
        data1 = pd.read_csv(f"{input_dir}/20190610_Fv_1.csv", index_col=0)
        data2 = pd.read_csv(f"{input_dir}/20190610_Fv_2.csv", index_col=0)
        data_start_time1, data_start_time2 = UTCDateTime(data1.iloc[0,-1]), UTCDateTime(data2.iloc[0,-1])
        # print(f"{data_start_time1.datetime} \n{data_start_time2.datetime}")
        time_diff1, time_diff2 = data_start_time1 - df_start_times.iloc[0], data_start_time2 - df_start_times.iloc[1]

        print(f"date: {start_date}, vel: {temp.iloc[0,2]}, time_shift: {distance / temp.iloc[0,2] : .2f}")
        print(f"date: {start_date}, vel: {temp.iloc[1,2]}, time_shift: {distance / temp.iloc[1,2] : .2f}")

        data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (5.2 * 60) + int(distance / temp.iloc[0,2]) # - peak_to_peak difference + time shift by velocity
        data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (4.9 * 60) + int(distance / temp.iloc[1,2]) # - peak_to_peak difference + time shift by velocity
        data = pd.concat([data1, data2])
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime("2019-06-10T00:00:00")
        time_list1 = []
        for i in range(0, 1440 * 60 + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
        merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
        merged_data.to_csv(f"{output_dir}/{start_date}.csv", index=False)

        # Unique Cases 2
        print("Running Unique Case 2")
        DataStart = "2019-07-01"
        DataEnd = "2019-07-04"
        temp = vel_df[(vel_df['Event_Date'] == "2019-07-02") | (vel_df['Event_Date'] == "2019-07-03")]
        df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(DataStart), UTCDateTime(DataEnd))][f'{station}Start']
        df_start_times = df_start_times.apply(UTCDateTime)
        data1 = pd.read_csv(f"{input_dir}/20190702_Fv.csv", index_col=0)
        data2 = pd.read_csv(f"{input_dir}/20190703_Fv.csv", index_col=0)
        data_start_time1, data_start_time2 = UTCDateTime(data1.iloc[0,-1]), UTCDateTime(data2.iloc[0,-1])
        print(f"date: {DataStart}, vel: {temp.iloc[0,2]}, time_shift: {distance / temp.iloc[0,2] : .2f}")
        print(f"date: {DataStart}, vel: {temp.iloc[1,2]}, time_shift: {distance / temp.iloc[1,2] : .2f}")
        # print(f"{data_start_time1.datetime} \n{data_start_time2.datetime}")
        time_diff1, time_diff2 = data_start_time1 - df_start_times.iloc[0], data_start_time2 - df_start_times.iloc[1]
        data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (61.95 * 60) + int(distance / temp.iloc[0,2])
        data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (68.1 * 60) + int(distance / temp.iloc[1,2])
        data = pd.concat([data1, data2])
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime("2019-07-01T00:00:00")
        time_list1 = []
        for i in range(0, (3 * 1440 * 60) + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
        merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
        merged_data[merged_data['Time'].between(UTCDateTime('2019-07-01'), UTCDateTime('2019-07-02'))].to_csv(f"{output_dir}/2019-07-01.csv", index=False)
        merged_data[merged_data['Time'].between(UTCDateTime('2019-07-02'), UTCDateTime('2019-07-03'))].to_csv(f"{output_dir}/2019-07-02.csv", index=False)
        merged_data[merged_data['Time'].between(UTCDateTime('2019-07-03'), UTCDateTime('2019-07-04'))].to_csv(f"{output_dir}/2019-07-03.csv", index=False)

        # General Cases
        print("Running General Case")
        # peak_shift_df = pd.read_csv("peak_times.csv", index_col=False)
        date_starts = ["2019-06-21","2019-07-15","2019-07-26","2019-08-11","2019-08-20"]
        date_ends = ["2019-06-22","2019-07-16","2019-07-27","2019-08-12","2019-08-21"]
        # peak_diff_values = [60, 68, 62, 57, 62]
        peak_diff_values = [61.2, 69.5, 61.3, 57.5, 62.5]
        for date_start, date_end, peak_diff in tqdm(zip(date_starts, date_ends, peak_diff_values), total=5):
            temp = vel_df[vel_df['Event_Date'] == date_start]
            DataStart = date_start
            DataEnd = date_end
            df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(DataStart), UTCDateTime(DataEnd))][f'{station}Start']
            df_start_times = df_start_times.apply(UTCDateTime)
            data = pd.read_csv(f"{input_dir}/{date_start.replace('-','')}_Fv.csv", index_col=0)
            data_start_time1 = UTCDateTime(data.iloc[0,-1])
            # print(data_start_time1.datetime)
            time_diff1 = data_start_time1 - df_start_times.iloc[0] 
            print(f"date: {date_start}, vel: {temp.iloc[0,2]}, time_shift: {distance / temp.iloc[0,2] : .2f}")
            data['Time'] = data.iloc[:,-1].apply(UTCDateTime) - (peak_diff * 60) + int(distance / temp.iloc[0,2])
            complete_data = pd.DataFrame(columns= ['Time'])
            start_time = UTCDateTime(date_start)
            time_list1 = []
            for i in range(0, 1440 * 60 + 1):
                time_list1.append(start_time + i)
            complete_data['Time'] = time_list1
            merged_data = pd.merge(left= complete_data, right= data, how="left", on= 'Time')
            merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
            merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
            merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
            merged_data.to_csv(f"{output_dir}/{date_start}.csv", index=False)
    elif from_velocity and avg_peak_shift:
        output_dir = f"{paths['LOCAL_BASE_DIR']}/label/data_processed_average/{station}"
        os.makedirs(output_dir, exist_ok=True)
        distance = 450 # in meters between ILL11 and Force Plate
        vel_df = pd.read_csv(f"{velocity_dir}/DF_characteristics.csv", index_col=False)
        vel_df['Event_Date'] = vel_df['Event_Date'].apply(lambda x: x.split('T')[0])
        avg_velocity = np.round(vel_df['Velocity'].mean())
        print(f"Average Velocity : {avg_velocity:.2f} m/s, Time Shift : {distance / avg_velocity:.2f} seconds")
        print("Running Unique Case 1")
        start_date = "2019-06-10"
        end_date = "2019-06-11"
        temp = vel_df[vel_df['Event_Date'] == start_date]
        df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(start_date),
                                                                                            UTCDateTime(end_date))][f'{station}Start']
        df_start_times = df_start_times.apply(UTCDateTime)
        data1 = pd.read_csv(f"{input_dir}/20190610_Fv_1.csv", index_col=0)
        data2 = pd.read_csv(f"{input_dir}/20190610_Fv_2.csv", index_col=0)
        data_start_time1, data_start_time2 = UTCDateTime(data1.iloc[0,-1]), UTCDateTime(data2.iloc[0,-1])
        # print(f"{data_start_time1.datetime} \n{data_start_time2.datetime}")
        time_diff1, time_diff2 = data_start_time1 - df_start_times.iloc[0], data_start_time2 - df_start_times.iloc[1]

        print(f"date: {start_date}, vel: {avg_velocity}, time_shift: {distance / avg_velocity : .2f}")
        print(f"date: {start_date}, vel: {avg_velocity}, time_shift: {distance / avg_velocity : .2f}")

        data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (5.2 * 60) + int(distance / avg_velocity) # - peak_to_peak difference + time shift by velocity
        data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (4.9 * 60) + int(distance / avg_velocity) # - peak_to_peak difference + time shift by velocity
        data = pd.concat([data1, data2])
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime("2019-06-10T00:00:00")
        time_list1 = []
        for i in range(0, 1440 * 60 + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
        merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
        merged_data.to_csv(f"{output_dir}/{start_date}.csv", index=False)

        # Unique Cases 2
        print("Running Unique Case 2")
        DataStart = "2019-07-01"
        DataEnd = "2019-07-04"
        temp = vel_df[(vel_df['Event_Date'] == "2019-07-02") | (vel_df['Event_Date'] == "2019-07-03")]
        df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(DataStart), UTCDateTime(DataEnd))][f'{station}Start']
        df_start_times = df_start_times.apply(UTCDateTime)
        data1 = pd.read_csv(f"{input_dir}/20190702_Fv.csv", index_col=0)
        data2 = pd.read_csv(f"{input_dir}/20190703_Fv.csv", index_col=0)
        data_start_time1, data_start_time2 = UTCDateTime(data1.iloc[0,-1]), UTCDateTime(data2.iloc[0,-1])
        print(f"date: {DataStart}, vel: {avg_velocity}, time_shift: {distance / avg_velocity : .2f}")
        print(f"date: {DataStart}, vel: {avg_velocity}, time_shift: {distance / avg_velocity : .2f}")
        # print(f"{data_start_time1.datetime} \n{data_start_time2.datetime}")
        time_diff1, time_diff2 = data_start_time1 - df_start_times.iloc[0], data_start_time2 - df_start_times.iloc[1]
        data1['Time'] = data1.iloc[:,-1].apply(UTCDateTime) - (61.95 * 60) + int(distance / avg_velocity)
        data2['Time'] = data2.iloc[:,-1].apply(UTCDateTime) - (68.1 * 60) + int(distance / avg_velocity)

        data = pd.concat([data1, data2])
        complete_data = pd.DataFrame(columns= ['Time'])
        start_time = UTCDateTime("2019-07-01T00:00:00")
        time_list1 = []
        for i in range(0, (3 * 1440 * 60) + 1):
            time_list1.append(start_time + i)
        complete_data['Time'] = time_list1
        merged_data = pd.merge(left= complete_data, right= data, how="left", on='Time')
        merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
        merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
        merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
        merged_data[merged_data['Time'].between(UTCDateTime('2019-07-01'), UTCDateTime('2019-07-02'))].to_csv(f"{output_dir}/2019-07-01.csv", index=False)
        merged_data[merged_data['Time'].between(UTCDateTime('2019-07-02'), UTCDateTime('2019-07-03'))].to_csv(f"{output_dir}/2019-07-02.csv", index=False)
        merged_data[merged_data['Time'].between(UTCDateTime('2019-07-03'), UTCDateTime('2019-07-04'))].to_csv(f"{output_dir}/2019-07-03.csv", index=False)

        # General Cases
        print("Running General Case")
        # peak_shift_df = pd.read_csv("peak_times.csv", index_col=False)
        date_starts = ["2019-06-21","2019-07-15","2019-07-26","2019-08-11","2019-08-20"]
        date_ends = ["2019-06-22","2019-07-16","2019-07-27","2019-08-12","2019-08-21"]
        # peak_diff_values = [60, 68, 62, 57, 62]
        peak_diff_values = [61.2, 69.5, 61.3, 57.5, 62.5]

        for date_start, date_end, peak_diff in tqdm(zip(date_starts, date_ends, peak_diff_values), total=5):
            temp = vel_df[vel_df['Event_Date'] == date_start]
            DataStart = date_start
            DataEnd = date_end
            df_start_times = event_data[event_data[f'{station}Start'].apply(UTCDateTime).between(UTCDateTime(DataStart), UTCDateTime(DataEnd))][f'{station}Start']
            df_start_times = df_start_times.apply(UTCDateTime)
            data = pd.read_csv(f"{input_dir}/{date_start.replace('-','')}_Fv.csv", index_col=0)
            data_start_time1 = UTCDateTime(data.iloc[0,-1])
            # print(data_start_time1.datetime)
            time_diff1 = data_start_time1 - df_start_times.iloc[0] 
            print(f"date: {date_start}, vel: {avg_velocity}, time_shift: {distance / avg_velocity : .2f}")
            data['Time'] = data.iloc[:,-1].apply(UTCDateTime) - (peak_diff * 60) + int(distance / avg_velocity)
            complete_data = pd.DataFrame(columns= ['Time'])
            start_time = UTCDateTime(date_start)
            time_list1 = []
            for i in range(0, 1440 * 60 + 1):
                time_list1.append(start_time + i)
            complete_data['Time'] = time_list1
            merged_data = pd.merge(left= complete_data, right= data, how="left", on= 'Time')
            merged_data.drop(columns=['Time UTC+1', 'Time UTC+0'], inplace=True)
            merged_data['Fv [kN]'] = merged_data['Fv [kN]'].apply(fill_missing_from_Gaussian, **{"mean": min_Fv, "std": sigma*min_Fv_std})
            merged_data = merged_data.fillna(value={"Fv std": min_Fv_std, "Fv min": 0, "Fv max": 0})
            merged_data.to_csv(f"{output_dir}/{date_start}.csv", index=False)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift Impact Force Peak Times")
    parser.add_argument("--time_shift", type=int, default=None, help="Time shift in minutes, can be None")
    parser.add_argument("--from_velocity", action="store_true", help="Shift based on velocity")
    parser.add_argument("--avg_shift", action="store_true", help="Shift based on average peak shift")
    
    args = parser.parse_args()

    main(time_shift_minutes=args.time_shift, from_velocity=args.from_velocity, avg_peak_shift=args.avg_shift)







