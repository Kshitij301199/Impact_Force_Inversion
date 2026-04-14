import os
import argparse
import json
try:
    with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/paths.json", "r") as file:
        paths = json.load(file)
except FileNotFoundError:
    with open("../config/paths.json", "r") as file:
        paths = json.load(file)
with open("/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/config/data_parameters.json", "r") as file:
    data_params = json.load(file)
from obspy import UTCDateTime, read, read_inventory, Stream, ObsPyException

def load_write_data(year:str, julday:str, station:str) -> None:
    """
    This function reads the seismic data for 3 consecutive days (if available) and
    removes the sensor response, following which it saves the data for download.
    Input :
        - year: str -- year of seismic data
        - julday: str -- julian day of data
        - station: str -- station of seismic data
    Output :
        - None
    """
    prev_julday = int(julday) - 1
    next_julday = int(julday) + 1
    st = Stream()
    print("\tLoading data for : ", year, julday, station)
    try:
        st += read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{str(prev_julday).zfill(3)}.mseed").resample(sampling_rate=100.0).merge(method=1, fill_value='latest', interpolation_samples=0)
    except FileNotFoundError:
        print("\t\tPrevious day data not found!")
    try:
        st += read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{str(julday).zfill(3)}.mseed").resample(sampling_rate=100.0).merge(method=1, fill_value='latest', interpolation_samples=0)
    except FileNotFoundError:
        print("\t\tCurrent day data not found!")
    try:    
        st += read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{str(next_julday)}.mseed").resample(sampling_rate=100.0).merge(method=1, fill_value='latest', interpolation_samples=0)
    except FileNotFoundError:
        print("\t\tNext day data not found!")
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    print("\tReading inventory and removing response")
    inv = read_inventory(f"{paths["META_DATA_DIR"]}/9S_2017_2023.xml")
    try:
        st.remove_response(inventory=inv)
    except ValueError:
        st.trim(starttime=UTCDateTime(year=int(year), julday=int(julday))-3600, endtime=UTCDateTime(year=int(year), julday=int(next_julday))+3600)
        st.remove_response(inventory=inv)
    # st.filter("bandpass", freqmin=data_params['fmin'], freqmax=data_params['fmax'])
    st.filter("bandpass", freqmin=1, freqmax=15)
    st.trim(starttime=UTCDateTime(year=int(year), julday=int(julday)), endtime=UTCDateTime(year=int(year), julday=int(next_julday)))
    # output_dir = f"./data_srr_{data_params['fmax']}/Illgraben/{year}/{station}/EHZ"
    output_dir = f"./data_srr_15/Illgraben/{year}/{station}/EHZ"
    os.makedirs(output_dir, exist_ok=True)
    try:
        st.write(f'{output_dir}/9S.{station}.EHZ.{year}.{julday}.mseed', format="MSEED")
    except ObsPyException:
        print("DATA MISSING")
        with open(f"{output_dir}/missing_data.txt", "a") as file:
            file.write(f"DATA MISSING\t{year}\t{station}\t{julday}\n")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="2019")
    parser.add_argument("--julday", type=str, default="161")
    parser.add_argument("--station", type=str, default="ILL11")

    args = parser.parse_args()

    load_write_data(args.year, args.julday, args.station)