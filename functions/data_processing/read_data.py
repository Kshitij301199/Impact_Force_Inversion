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
from obspy import read, Stream, read_inventory
from obspy.core import UTCDateTime # default is UTC+0 time zone

def load_data(event_id_list:list, station:str, year:int=2019, trim:bool=True, abs:bool=True) -> np.array:
    total_data = None
    for event_id in event_id_list:
        st = load_seismic_data(event_id = str(event_id), station= station, year=year, trim= trim)
        data = st[0].data[1:]
        if total_data is None:
            total_data = data
        else:
            total_data = np.concatenate([total_data, data])
    if abs:
        total_data = np.abs(total_data)
    return total_data

def load_seismic_data(event_id:str|int, station:str, 
                      year:int=None, component:str='EHZ', network:str="9S", 
                      trim:bool = True) -> Stream:
    scaling = 1e3
    time_window = data_params['time_window']
    event_id = str(event_id)
    event_info = time_config[event_id]
    julday = event_info['julday']
    # LOAD THE DATA AND SCALE
    if type(julday) is int:
        try:
            st = read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(julday).zfill(3)}.mseed")
        except FileNotFoundError:
            st = read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(julday).zfill(3)}.mseed")
        st[0].data = st[0].data * scaling
    elif type(julday) is list:
        st = Stream()
        for j in julday:
            try:
                st += read(f"{paths['BASE_DIR']}/{paths['DATA_DIR']}/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(j).zfill(3)}.mseed")
            except:
                st += read(f"{paths['LOCAL_BASE_DIR']}/{paths['DATA_DIR']}/{year}/{station}/{component}/9S.{station}.{component}.{year}.{str(j).zfill(3)}.mseed")
        st.merge(method=1, fill_value='latest', interpolation_samples=0)
        st[0].data = st[0].data * scaling
    else:
        print(f"Wrong julday type : {type(julday)}")
        raise TypeError
    # TRIM THE DATA
    if trim:
        st.trim(starttime=UTCDateTime(event_info['start_time']) - (time_window * 120), 
                endtime=UTCDateTime(event_info['end_time']) + (time_window * 120))
    return st

def load_label(event_id_list: list, station: str, interval_seconds: int, time_shift_minutes, trim:bool = True, smoothing: int | None = 30, divide_by: int | None = 350) -> pd.DataFrame:
    time_window = data_params['time_window']
    # SELECT COLUMN FOR WHICH THE DATA IS REQUESTED
    # if smoothing is None:
    #     data_col = ['Fv [kN]', 'moving_avg_10', 'moving_avg_30', 'moving_avg_60']
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
            target = target[target['Time'].between(start_time - (time_window * 60) , end_time + (time_window * 120))]
        else:
            if type(date) is list:
                target = target[target['Time'] >= UTCDateTime(f"{date[0]}") + (time_window * 60)]
            else:
                target = target[target['Time'] >= UTCDateTime(f"{date}") + (time_window * 60)]

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

