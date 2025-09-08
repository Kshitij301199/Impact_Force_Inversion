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
from obspy import UTCDateTime, read, read_inventory, Stream

def load_write_data(year:str, julday:str, station:str):
    prev_julday = int(julday) - 1
    next_julday = int(julday) + 1
    st = Stream()
    try:
        st += read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{str(prev_julday).zfill(3)}.mseed")
    except FileNotFoundError:
        print("Previous day data not found!")
    try:
        st += read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{str(julday).zfill(3)}.mseed")
    except FileNotFoundError:
        print("Current day data not found!")
    try:    
        st += read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{str(next_julday)}.mseed") 
    except FileNotFoundError:
        print("Next day data not found!")
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    if year == "2022":
        inv = read_inventory(f"/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/meta_data/9S_2022.xml")
    else:
        inv = read_inventory(f"{paths["META_DATA_DIR"]}/9S_2017_2020.xml")
    st.remove_response(inventory=inv)
    st.filter("bandpass", freqmin=data_params['fmin'], freqmax=data_params['fmax'])
    st.trim(starttime=UTCDateTime(year=int(year), julday=int(julday)), endtime=UTCDateTime(year=int(year), julday=int(next_julday)))
    output_dir = f"./data_srr/Illgraben/{year}/{station}/EHZ"
    os.makedirs(output_dir, exist_ok=True)
    st.write(f'{output_dir}/9S.{station}.EHZ.{year}.{julday}.mseed', format="MSEED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="2019")
    parser.add_argument("--julday", type=str, default="161")
    parser.add_argument("--station", type=str, default="ILL11")

    args = parser.parse_args()

    load_write_data(args.year, args.julday, args.station)