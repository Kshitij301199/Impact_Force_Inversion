#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=abl_xlstm           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-36%2            # job array id, adjusted for the total number of commands (8 test days * 7 validation days * 4 intervals)
#SBATCH --mem-per-cpu=16G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A30:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/abs_xlstm_out_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/abs_xlstm_err_%A_%a.txt    # Standard Error Log File (for Job Arrays)

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Calculate the index for test_julday/val_julday and num_day based on SLURM_ARRAY_TASK_ID (1-36)
test_juldays=(161 172 196 207 223 232)
val_juldays=(207 207 161 183 183 161)
num_days=(1 2 3 4 5 6)

# SLURM_ARRAY_TASK_ID is 1-based
task_id=$((SLURM_ARRAY_TASK_ID - 1))
test_val_idx=$((task_id / 6))
num_day_idx=$((task_id % 6))

test_julday="${test_juldays[$test_val_idx]}"
val_julday="${val_juldays[$test_val_idx]}"
num_day="${num_days[$num_day_idx]}"

echo "Running for:"
echo "Number of Training days: $num_day"
echo "Test Julian Day: $test_julday"
echo "Validation Julian Day: $val_julday"
echo "Interval: 5"
echo "Hypothesis Option: default"

# Run the Python script with the selected parameters
srun --gres=gpu:A30:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/training/train_xlstm.py \
    --test_julday $test_julday \
    --val_julday $val_julday \
    --time_shift_mins 'average' \
    --interval 5 \
    --station "ILL11" \
    --config_op "default" \
    --task "abalation_study_1" \
    --smoothing 30 \
    --num_days $num_day
