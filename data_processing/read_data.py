import json
with open("../config/paths.json", "r") as file:
    paths = json.load(file)
import numpy as np
import pandas as pd
from obspy import read, Stream
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


def load_seismic_data(julday:str|int|list, station:str) -> Stream:
    if type(julday) is int:
        st = read(f"{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{julday}.mseed")
        st[0].data = st[0].data * 1e3
    elif type(julday) is str:
        st = read(f"{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{julday}.mseed")
        st[0].data = st[0].data * 1e3
    elif type(julday) is list:
        st = Stream()
        for jul in julday:
            st += read(f"{paths['DATA_DIR']}/2019/{station}/EHZ/9S.{station}.EHZ.2019.{jul}.mseed")
            st.merge(method=1, fill_value='latest', interpolation_samples=0)
            st._cleanup()
            st.detrend('linear')
            st.detrend('demean')
        st[0].data = st[0].data * 1e3
    else:
        print(f"Wrong julday type : {type(julday)}")
        raise TypeError
    return st

def load_label(date_list:list, station:str, interval_seconds:int) -> pd.DataFrame:
    total_target = None
    for date in date_list:    
        target_start_time = UTCDateTime(f"{date}") + (10*60)
        target = pd.read_csv(f"{paths['LABEL_DIR']}/{station}/{date}.csv", index_col=0)
        target = target[target['Time'] >= target_start_time]
        target['Timestamp'] = target['Time'].apply(UTCDateTime).apply(UTCDateTime._get_timestamp)
        target = target.iloc[::interval_seconds]
        if total_target is None:
            total_target = target
        else:
            total_target = pd.concat([total_target, target])
    total_target.reset_index(inplace=True)
        
    return total_target
