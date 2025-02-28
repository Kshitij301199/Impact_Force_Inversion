#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=ratios          # job name
#SBATCH --ntasks=1                # single task per job array
#SBATCH --array=1-72%3              # Adjusted for (8 test days * 4 intervals * 3 hyp_options)
#SBATCH --mem-per-cpu=16G         # Memory per CPU
#SBATCH --gres=gpu:A40:1          # Request GPU
#SBATCH --reservation=GPU         # Reserve GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/svm_out_%A_%a.txt  # Standard Output Log File
#SBATCH --error=./logs/err/svm_err_%A_%a.txt   # Standard Error Log File

# Load required modules
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Define the test and validation day lists (1-to-1 linked)
# 10secs
test_juldays=(161 172 182 183 196 207 223 232)
val_juldays=(182 223 183 223 161 232 207 183)  # Ensure corresponding indices match
# 5secs
# test_juldays=(161 172 182 183 196 207 223 232)
# val_juldays=(162 173 184 185 197 208 224 233)  # Ensure corresponding indices match

# Define intervals and hypothesis options
intervals=(15 30 60)
hyp_options=('mlstm' 'slstm' 'xlstm')

# Get the number of test days
num_test_days=${#test_juldays[@]}
num_intervals=${#intervals[@]}
num_hyp_options=${#hyp_options[@]}
num_combinations=$(( num_intervals * num_hyp_options ))  # Per test day

# Compute indices for test day
test_day_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / $num_combinations ))
remaining_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % $num_combinations ))

interval_idx=$(( $remaining_idx / $num_hyp_options ))
hyp_option_idx=$(( $remaining_idx % $num_hyp_options ))

# Get the current test and validation Julian days (1-to-1 mapping)
test_julday=${test_juldays[$test_day_idx]}
val_julday=${val_juldays[$test_day_idx]}  # Directly linked

# Get the current interval
interval=${intervals[$interval_idx]}

# Get the current hypothesis option
hyp_option=${hyp_options[$hyp_option_idx]}

# Log parameters
echo "Running for:"
echo "Test Julian Day: $test_julday"
echo "Validation Julian Day: $val_julday"
echo "Interval: $interval"
echo "Hypothesis Option: $hyp_option"

# Execute Python script with parameters
srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/train_xlstm.py \
    --test_julday "$test_julday" \
    --val_julday "$val_julday" \
    --interval "$interval" \
    --station "ILL11" \
    --config_op "$hyp_option" \
    --task "slstm_v_mlstm" \
    --time_shift_mins 10
