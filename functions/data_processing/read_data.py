import json
try:
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
        paths = json.load(file)
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
        data_params = json.load(file)
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/event_id_map.json", "r") as file:
        time_config = json.load(file)
except FileNotFoundError:
    with open("../config/paths.json", "r") as file:
        paths = json.load(file)
    with open("../config/data_parameters.json", "r") as file:
        data_params = json.load(file)
    with open("../config/event_id_map.json", "r") as file:
        time_config = json.load(file)

import numpy as np
import pandas as pd
import obspy
import obspy.signal
import obspy.signal.filter
from obspy import read, Stream, read_inventory
from obspy.core import UTCDateTime # default is UTC+0 time zone

def load_seismic_data(event_id:str|int, station:str, 
                      year:int=None, component:str='EHZ', network:str="9S", 
                      trim:bool = True) -> Stream:
    """
    This function loads the seismic data for model training
    Input -
        event_id - refer to config/event_id_map for more info
        station - seismic station
        year - year of the event event
        component - component of seismic signal, default EHZ
        network - associated the seismic network, default 9S
        trim - cut the data for the debris flow period, default True
    Output -
        Stream() object containing the seismic data without sensor response
    """
    scaling = 1e3
    time_window = data_params['time_window']
    event_id = str(event_id)
    event_info = time_config[event_id]
    julday = event_info['julday']
    # LOAD THE DATA AND SCALE
    if type(julday) is int:
        try:
            st = read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}_{data_params['fmax']}/Illgraben/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(julday).zfill(3)}.mseed")
        except FileNotFoundError:
            st = read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}_{data_params['fmax']}/Illgraben/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(julday).zfill(3)}.mseed")
        st[0].data = st[0].data * scaling
    elif type(julday) is list:
        st = Stream()
        for j in julday:
            try:
                st += read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}_{data_params['fmax']}/Illgraben/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(j).zfill(3)}.mseed")
            except:
                st += read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}_{data_params['fmax']}/Illgraben/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(j).zfill(3)}.mseed")
        st.merge(method=1, fill_value='latest', interpolation_samples=0)
        st[0].data = st[0].data * scaling
    else:
        print(f"Wrong julday type : {type(julday)}")
        raise TypeError
    # TRIM THE DATA
    if trim:
        st.trim(starttime=UTCDateTime(event_info['start_time']) - (time_window * 60 * 2), 
                endtime=UTCDateTime(event_info['end_time']) + (time_window * 60 * 2))
    return st

def load_seismic_data_test(julday:int|str|list, station:str, 
                      year:int=None, component:str='EHZ', network:str="9S", freq = None,
                      ) -> Stream:
    """
    This function loads the seismic data for model training
    Input -
        julday - julian day of the debris flow event
        station - seismic station
        year - year of the event event
        component - component of seismic signal, default EHZ
        network - associated the seismic network, default 9S
        freq - upper bound for the frequency, either 15 or 45
    Output -
        Stream() object containing the seismic data without sensor response
    """
    scaling = 1e3
    data_freq = data_params['fmax'] if freq is None else freq
    # LOAD THE DATA AND SCALE
    if type(julday) is int or type(julday) is str:
        try:
            st = read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}_{data_freq}/Illgraben/{year}/{station}/{component}/{network}.{station}.{component}.{year}.{str(julday).zfill(3)}.mseed")
        except FileNotFoundError:
            st = read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}_{data_freq}/Illgraben/{year}/{station}/{component}/{network}.{station}.{component}.{year}.{str(julday).zfill(3)}.mseed")
        st[0].data = st[0].data * scaling
    elif type(julday) is list:
        st = Stream()
        for j in julday:
            try:
                st += read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}_{data_freq}/Illgraben/{year}/{station}/{component}/{network}.{station}.{component}.{year}.{str(j).zfill(3)}.mseed")
            except:
                st += read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}_{data_freq}/Illgraben/{year}/{station}/{component}/{network}.{station}.{component}.{year}.{str(j).zfill(3)}.mseed")
        st.merge(method=1, fill_value='latest', interpolation_samples=0)
        st[0].data = st[0].data * scaling
    else:
        print(f"Wrong julday type : {type(julday)}")
        raise TypeError
    return st

def load_data(event_id_list:list, station:str, year:int=2019, trim:bool=True, abs:bool=True, env:bool=True) -> np.array:
    total_data = None
    total_times = None
    for event_id in event_id_list:
        st = load_seismic_data(event_id = str(event_id), station= station, year=year, trim= trim)
        if env:
            data_envelope = obspy.signal.filter.envelope(st[0].data)
            data = data_envelope
            data = data[1:]
        else:
            data = st[0].data[1:]
        times = st[0].times("matplotlib")[1:]
        if total_data is None:
            total_data = data
            total_times = times
        else:
            total_data = np.concatenate([total_data, data])
            total_times = np.concatenate([total_times, times])
    if abs:
        total_data = np.abs(total_data)
    return total_data, total_times

def load_data_test(julday_list:list, station:str, year:int=2019, abs:bool=True, env:bool=True) -> np.array:
    total_data = None
    total_times = None
    for julday in julday_list:
        st = load_seismic_data_test(julday = julday, station= station, year=year)
        if env:
            data_envelope = obspy.signal.filter.envelope(st[0].data)
            data = data_envelope
            data = data[1:]
        else:
            data = st[0].data[1:]
        times = st[0].times("matplotlib")[1:]
        if total_data is None:
            total_data = data
            total_times = times
        else:
            total_data = np.concatenate([total_data, data])
            total_times = np.concatenate([total_times, times])
    if abs:
        total_data = np.abs(total_data)
    return total_data, total_times

def load_label(event_id_list: list, station: str, interval_seconds: int, time_shift_minutes, trim:bool = True, smoothing: int | None = 30, divide_by: int | None = 45) -> pd.DataFrame:
    time_window = data_params['time_window']
    # SELECT COLUMN FOR WHICH THE DATA IS REQUESTED
    if smoothing == 0 or smoothing is None:
        data_col = "Fv [kN]"
    else:
        data_col = f"moving_avg_{smoothing}"
    # LOAD DATA
    total_target = None
    for i, event_id in enumerate(event_id_list):
        event_id = str(event_id)
        julday = time_config[event_id]['julday'] if type(time_config[event_id]['julday']) is int else time_config[event_id]['julday'][0]
        date = time_config[event_id]['date']
        start_time, end_time = UTCDateTime(time_config[event_id]['start_time']), UTCDateTime(time_config[event_id]['end_time'])

        if type(date) is str:
            try:
                target = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")
            except FileNotFoundError:
                target = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")
        elif type(date) is list:
            target = None
            for d in date:
                if target is None:
                    try:
                        target = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                    except FileNotFoundError:
                        target = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                else:
                    try:
                        temp = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                    except FileNotFoundError:
                        temp = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                    target = pd.concat([target, temp.iloc[1:]])
                    target.reset_index(inplace=True, drop=True)
                    del temp

        # Filter data to start after the target start time
        if trim:
            if i == 0:
                target = target[target['Time'].between(start_time - (time_window * 60 * 1) , end_time + (time_window * 60 * 2))]
            else:
                target = target[target['Time'].between(start_time - (time_window * 60 * 2) , end_time + (time_window * 60 * 2))]
        else:
            if i == 0:
                if type(date) is list:
                    target = target[target['Time'] >= UTCDateTime(f"{date[0]}") + (time_window * 60)]
                else:
                    target = target[target['Time'] >= UTCDateTime(f"{date}") + (time_window * 60)]
            else:
                pass

        # Convert Time to Timestamp
        target['Timestamp'] = target['Time'].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)
        # Force <-> Pressure Conversion with plate area 8m*m
        target[data_col] = target[data_col] / 8

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
                'Fv [kN]' : target[data_col].values,
                'Fv std' : target['Fv std'].values
            })
        # Concatenate results
        if total_target is None:
            total_target = target
        else:
            total_target = pd.concat([total_target, target])

    total_target.reset_index(drop=True, inplace=True)
    if divide_by is not None:
        total_target['Fv [kN]'] = total_target['Fv [kN]'] / divide_by  # Divide by 350
    
    return total_target

def load_label2(date_list: list, station: str, interval_seconds: int, time_shift_minutes, smoothing: int | None = 30, divide_by: int | None = 350) -> pd.DataFrame:
    time_window = data_params['time_window']
    # SELECT COLUMN FOR WHICH THE DATA IS REQUESTED
    if smoothing == 0 or smoothing is None:
        data_col = "Fv [kN]"
    else:
        data_col = f"moving_avg_{smoothing}"
    # LOAD DATA
    total_target = None
    for i, date in enumerate(date_list):
        # event_id = str(event_id)
        # julday = time_config[event_id]['julday'] if type(time_config[event_id]['julday']) is int else time_config[event_id]['julday'][0]
        # date = time_config[event_id]['date']
        # start_time, end_time = UTCDateTime(time_config[event_id]['start_time']), UTCDateTime(time_config[event_id]['end_time'])

        if type(date) is str:
            try:
                target = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")
            except FileNotFoundError:
                target = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{date}.csv")
        elif type(date) is list:
            target = None
            for d in date:
                if target is None:
                    try:
                        target = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                    except FileNotFoundError:
                        target = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                else:
                    try:
                        temp = pd.read_csv(f"{paths['BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                    except FileNotFoundError:
                        temp = pd.read_csv(f"{paths['LOCAL_BASE_DIR']}/{paths['LABEL_DIR']}_{time_shift_minutes}/{station}/{d}.csv")
                    target = pd.concat([target, temp.iloc[1:]])
                    target.reset_index(inplace=True, drop=True)
                    del temp

        # Filter data to start after the target start time
        if i == 0:
            if type(date) is list:
                target = target[target['Time'] >= UTCDateTime(f"{date[0]}") + (time_window * 60)]
            else:
                target = target[target['Time'] >= UTCDateTime(f"{date}") + (time_window * 60)]
        else:
            pass

        # Convert Time to Timestamp
        target['Timestamp'] = target['Time'].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)
        # Force <-> Pressure Conversion with plate area 8m*m
        target[data_col] = target[data_col] / 8

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
                'Fv [kN]' : target[data_col].values,
                'Fv std' : target['Fv std'].values
            })
        # Concatenate results
        if total_target is None:
            total_target = target
        else:
            total_target = pd.concat([total_target, target])

    total_target.reset_index(drop=True, inplace=True)
    if divide_by is not None:
        total_target['Fv [kN]'] = total_target['Fv [kN]'] / divide_by  # Divide by 350
    
    return total_target


