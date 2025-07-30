#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=abl_lstm           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-6%2            # job array id, adjusted for the total number of commands (8 test days * 7 validation days * 4 intervals)
#SBATCH --mem-per-cpu=24G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A30:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/abs_lstm_out_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/abs_lstm_err_%A_%a.txt    # Standard Error Log File (for Job Arrays)

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

num_days=({1..6})
num_day=${num_days[$SLURM_ARRAY_TASK_ID-1]}

# Log the current parameters
echo "Running for:"
echo "Number of Training days: $num_day"
echo "Test Julian Day: 232"
echo "Validation Julian Day: 223"
echo "Interval: 5"
echo "Hypothesis Option: default"

# Run the Python script with the selected parameters
srun --gres=gpu:A30:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/train_lstm.py \
    --test_julday "$num_day" \
    --val_julday 0 \
    --time_shift_mins 'dynamic' \
    --interval 5 \
    --station "ILL11" \
    --config_op "default" \
    --task "abalation_study_1" \
    --smoothing 30

