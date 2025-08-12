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
from obspy import read, read_inventory

def load_write_data(year:str, julday:str, station:str):
    st = read(f"{paths['SEISMIC_DATA_DIR']}/{year}/{station}/EHZ/9S.{station}.EHZ.{year}.{julday}.mseed")
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    inv = read_inventory(f"{paths["META_DATA_DIR"]}/9S_2017_2020.xml")
    st.remove_response(inventory=inv)
    st.filter("bandpass", freqmin=data_params['fmin'], freqmax=data_params['fmax'])
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