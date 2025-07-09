#!/bin/bash
#SBATCH -t 8:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=test_models           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-8%1            # job array id, adjusted for the total number of commands (stations * juldays * intervals)
#SBATCH --mem-per-cpu=16G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A40:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/t18_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/t18_%A_%a.txt    # Standard Error Log File (for Job Arrays)

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Define parameters
network="9S"
station_list=("ILL11")
component="EHZ"
year=2018
julday_list=(162 163 206 220)
intervals=(5)
models=('xLSTM' 'LSTM')

# Compute total job count
total_jobs=$(( ${#station_list[@]} * ${#julday_list[@]} * ${#intervals[@]} * ${#models[@]} ))

# Get SLURM job array index (1-based)
index=$((SLURM_ARRAY_TASK_ID - 1))

# Determine station, julday, interval, and model based on job index
station_index=$(( index / ( ${#julday_list[@]} * ${#intervals[@]} * ${#models[@]} ) ))
julday_index=$(( (index / ( ${#intervals[@]} * ${#models[@]} )) % ${#julday_list[@]} ))
interval_index=$(( (index / ${#models[@]}) % ${#intervals[@]} ))
model_index=$(( index % ${#models[@]} ))

station=${station_list[$station_index]}
julday=${julday_list[$julday_index]}
interval=${intervals[$interval_index]}
model=${models[$model_index]}

# Run Python script with selected parameters
srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/test_models.py \
    --network "$network" \
    --station "$station" \
    --component "$component" \
    --year "$year" \
    --julday "$julday" \
    --interval "$interval" \
    --model_type "$model"



