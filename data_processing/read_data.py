import json
try:
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
        paths = json.load(file)
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
        data_params = json.load(file)
except FileNotFoundError:
    with open("../config/paths.json", "r") as file:
        paths = json.load(file)
    with open("../config/data_parameters.json", "r") as file:
        data_params = json.load(file)
import numpy as np
import pandas as pd
from obspy import read, Stream, read_inventory
from obspy.core import UTCDateTime # default is UTC+0 time zone

def load_data(julday_list:list, station:str, trim:bool, abs:bool) -> np.array:
    total_data = None
    for julday in julday_list:
        st = load_seismic_data(julday= julday, station= station, trim= trim)
        data = st[0].data[1:]
        if total_data is None:
            total_data = data
        else:
            total_data = np.concatenate([total_data, data])
    if abs:
        total_data = np.abs(total_data)
    return total_data

def load_seismic_data(julday:str|int, station:str, raw:bool=False, 
                      year:int=None, component:str=None, network:str=None, 
                      trim:bool = True) -> Stream:
    scaling = 1e3
    if not raw:
        if type(julday) is int:
            try:
                st = read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{julday}.mseed")
            except FileNotFoundError:
                st = read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{julday}.mseed")
            st[0].data = st[0].data * scaling
        elif type(julday) is str:
            st = read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{julday}.mseed")
            st[0].data = st[0].data * scaling
        # elif type(julday) is list:
        #     st = Stream()
        #     for jul in julday:
        #         st += read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{jul}.mseed")
        #         st.merge(method=1, fill_value='latest', interpolation_samples=0)
        #         st._cleanup()
        #         st.detrend('linear')
        #         st.detrend('demean')
        #         st.filter("bandpass", freqmin=data_params['fmin'], freqmax=data_params['fmax'])
            st[0].data = st[0].data * scaling
        else:
            print(f"Wrong julday type : {type(julday)}")
            raise TypeError
        if trim:
            try:
                trim_df = pd.read_csv(f"{paths['BASE_DIR']}/label/correct_metrics_time_window.csv")
            except FileNotFoundError:
                trim_df = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/label/correct_metrics_time_window.csv")
            trim_df['Start_Time'] = trim_df['Start_Time'].apply(UTCDateTime)
            trim_df['End_Time'] = trim_df['End_Time'].apply(UTCDateTime)
            trim_df['Julday'] = trim_df['Start_Time'].apply(lambda x: x.julday)
            if julday == 161:
                st.trim(starttime=trim_df.iloc[0,0] - (data_params['time_window'] * 120), endtime=trim_df.iloc[1,1])
            else:
                trim_df = trim_df[trim_df['Julday'] == julday]
                st.trim(starttime=trim_df.iloc[0,0] - (data_params['time_window'] * 120), endtime=trim_df.iloc[0,1] + (data_params['time_window'] * 120))
        return st
    else:
        st = read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/{component}/{network}.{station}.{component}.{year}.{julday}.mseed")
        st.merge(method=1, fill_value='latest', interpolation_samples=0)
        st._cleanup()
        st.detrend('linear')
        st.detrend('demean')
        inv = read_inventory(f"{paths['META_DATA_DIR']}/9S_2017_2020.xml")
        st.remove_response(inventory=inv)
        st.filter("bandpass", freqmin=data_params['fmin'], freqmax=data_params['fmax'])
        st[0].data = st[0].data * scaling
        return st

def load_label(date_list: list, station: str, interval_seconds: int, time_shift_minutes, trim:bool = True, smoothing: int | None = 30) -> pd.DataFrame:
    time_window = data_params['time_window']

    if smoothing == None or smoothing == 0:
        data_col = "Fv [kN]"
    else:
        data_col = f"moving_avg_{smoothing}"
    
    total_target = None
    try:
        trim_df = pd.read_csv(f"{paths['BASE_DIR']}/label/correct_metrics_time_window.csv")
    except FileNotFoundError:
        trim_df = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/label/correct_metrics_time_window.csv")
    trim_df['Start_Time'] = trim_df['Start_Time'].apply(UTCDateTime)
    trim_df['End_Time'] = trim_df['End_Time'].apply(UTCDateTime)
    trim_df['Julday'] = trim_df['Start_Time'].apply(lambda x: x.julday)
    for i, date in enumerate(date_list):
        julday = UTCDateTime(date).julday
        present_trim = trim_df[trim_df['Julday'] == julday]
        if i == 0:
            if trim:
                # Use the first row of trim_df for the first date
                target_start_time = present_trim.iloc[0, 0] - (data_params['time_window'] * 60)  # Offset by 5/10 minutes
                if julday == 161:
                    # Special case for julday 161, use the second row for the end time
                    target_end_time = present_trim.iloc[1, 1] 
                else:
                    target_end_time = present_trim.iloc[0, 1] + (data_params['time_window'] * 120)
            else:
                target_start_time = UTCDateTime(f"{date}") + (data_params['time_window'] * 60)  # Offset by 5/10 minutes
                target_end_time = None
        else:
            if trim:
                # Use the second row of trim_df for subsequent dates
                target_start_time = present_trim.iloc[0, 0] - (data_params['time_window'] * 120)
                if julday == 161:
                    # Special case for julday 161, use the second row for the end time
                    target_end_time = present_trim.iloc[1, 1]
                else:
                    target_end_time = present_trim.iloc[0, 1] + (data_params['time_window'] * 120)
            else:
                target_start_time = UTCDateTime(f"{date}")
                target_end_time = None
        # Attempt to read CSV file from different paths
        try:
            target = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")
        except FileNotFoundError:
            target = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")

        # Filter data to start after the target start time
        target = target[target['Time'] >= target_start_time]
        if target_end_time is not None:
            target = target[target['Time'] <= target_end_time]

        # Convert Time to Timestamp
        target['Timestamp'] = target['Time'].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)

        if interval_seconds != 1:
            # Apply sliding window mean using NumPy
            num_windows = len(target) // interval_seconds  # Number of full windows
            target = target.iloc[:num_windows * interval_seconds]  # Trim excess data

            # Reshape data for window-based averaging
            reshaped_values = target[data_col].values.reshape(num_windows, interval_seconds)
            averaged_values = np.mean(reshaped_values, axis=1)
            std_values = np.std(reshaped_values, axis=1)

            # Create new DataFrame
            target = pd.DataFrame({
                'Timestamp': target['Timestamp'].values[::interval_seconds],  # Take every stride-th timestamp
                'Fv [kN]': averaged_values,  # Store the computed mean
                'Fv std': std_values
            })
        else:
            target = pd.DataFrame({
                'Timestamp' : target['Timestamp'].values,
                'Fv [kN]' : target['Fv [kN]'].values,
                'Fv std' : target['Fv std'].values
            })
        # Concatenate results
        if total_target is None:
            total_target = target
        else:
            total_target = pd.concat([total_target, target])

    total_target.reset_index(drop=True, inplace=True)
    
    return total_target

