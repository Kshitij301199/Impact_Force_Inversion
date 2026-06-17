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
#SBATCH --output=./logs/out/t17_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/t17_%A_%a.txt    # Standard Error Log File (for Job Arrays)

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Define parameters
NETWORK="9S"
STATIONS=("ILL02")
COMPONENT="EHZ"
YEAR=2017
JULDAYS=(139 148 154 165)
INTERVALS=(5)
MODELS=('xLSTM' 'LSTM')

# Calculate indices (Mapping 1D array task ID to N-dimensional parameter space)
idx=$((SLURM_ARRAY_TASK_ID - 1))

n_mod=${#MODELS[@]};    model=${MODELS[$(( idx % n_mod ))]};     idx=$(( idx / n_mod ))
n_int=${#INTERVALS[@]}; interval=${INTERVALS[$(( idx % n_int ))]}; idx=$(( idx / n_int ))
n_jul=${#JULDAYS[@]};   julday=${JULDAYS[$(( idx % n_jul ))]};     idx=$(( idx / n_jul ))
n_sta=${#STATIONS[@]};  station=${STATIONS[$(( idx % n_sta ))]}

echo "Task $SLURM_ARRAY_TASK_ID: Station=$station, Julday=$julday, Interval=$interval, Model=$model"

# Run Python script with selected parameters
# Using python -u for unbuffered output; --gres is inherited from #SBATCH headers
srun python -u /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/application/test_models.py \
    --network "$NETWORK" \
    --station "$station" \
    --component "$COMPONENT" \
    --year "$YEAR" \
    --julday "$julday" \
    --interval "$interval" \
    --model_type "$model"
