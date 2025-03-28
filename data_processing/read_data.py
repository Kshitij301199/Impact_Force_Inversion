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

def load_data(julday_list:list, station:str) -> np.array:
    total_data = None
    for julday in julday_list:
        st = load_seismic_data(julday= julday, station= station)
        data = st[0].data[1:]
        if total_data is None:
            total_data = data
        else:
            total_data = np.concatenate([total_data, data])
    return total_data


def load_seismic_data(julday:str|int|list, station:str, raw:bool=False, 
                      year:int=None, component:str=None, network:str=None) -> Stream:
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
        elif type(julday) is list:
            st = Stream()
            for jul in julday:
                st += read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{jul}.mseed")
                st.merge(method=1, fill_value='latest', interpolation_samples=0)
                st._cleanup()
                st.detrend('linear')
                st.detrend('demean')
                st.filter("bandpass", freqmin=data_params['fmin'], freqmax=data_params['fmax'])
            st[0].data = st[0].data * scaling
        else:
            print(f"Wrong julday type : {type(julday)}")
            raise TypeError
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

def load_label(date_list: list, station: str, interval_seconds: int, time_shift_minutes) -> pd.DataFrame:
    total_target = None

    for i, date in enumerate(date_list):
        if i == 0:
            target_start_time = UTCDateTime(f"{date}") + (data_params['time_window'] * 60)  # Offset by 10 minutes
        else:
            target_start_time = UTCDateTime(f"{date}")
        # Attempt to read CSV file from different paths
        try:
            target = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")
        except FileNotFoundError:
            target = pd.read_csv(f"../{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")

        # Filter data to start after the target start time
        target = target[target['Time'] >= target_start_time]

        # Convert Time to Timestamp
        target['Timestamp'] = target['Time'].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)

        if interval_seconds != 1:
            # Apply sliding window mean using NumPy
            num_windows = len(target) // interval_seconds  # Number of full windows
            target = target.iloc[:num_windows * interval_seconds]  # Trim excess data

            # Reshape data for window-based averaging
            reshaped_values = target['Fv [kN]'].values.reshape(num_windows, interval_seconds)
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

