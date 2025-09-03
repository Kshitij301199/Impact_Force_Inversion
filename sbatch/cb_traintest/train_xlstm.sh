#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=base_xlstm           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-14%4            # job array id, adjusted for the total number of commands (8 test days * 7 validation days * 4 intervals)
#SBATCH --mem-per-cpu=16G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A40:1             # load GPU A100 could be replace by A40/A40, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/xlstm_out_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/xlstm_err_%A_%a.txt    # Standard Error Log File (for Job Arrays)

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Define the arrays
intervals=(15 30)
juldays=(161 172 183 196 207 223 232)  # 84
# juldays=(172 196 207 223) # 12
# juldays=(161 183 232) # 6
hyp_options=('default')
# hyp_options=('mlstm' 'slstm')


# Calculate total combinations (only cases where test_julday == val_julday)
num_intervals=${#intervals[@]}
num_hyp_options=${#hyp_options[@]}
num_juldays=${#juldays[@]}

total_combinations=$((num_intervals * num_hyp_options * num_juldays))

# Get the zero-based index for this SLURM array task
task_index=$((SLURM_ARRAY_TASK_ID - 1))

# Map the task index to the parameter combinations
interval_idx=$((task_index / (num_hyp_options * num_juldays) % num_intervals))
hyp_option_idx=$((task_index / num_juldays % num_hyp_options))
julday_idx=$((task_index % num_juldays))

interval=${intervals[$interval_idx]}
hyp_option=${hyp_options[$hyp_option_idx]}
test_julday=${juldays[$julday_idx]}
val_julday=${juldays[$julday_idx]}

# Log the current parameters
echo "Running for:"
echo "Test Julian Day: $test_julday"
echo "Validation Julian Day: $val_julday"
echo "Interval: $interval"
echo "Hypothesis Option: $hyp_option"

# Run the Python script with the selected parameters
srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/training/train_xlstm.py \
    --test_julday "$test_julday" \
    --val_julday "$val_julday" \
    --time_shift_mins 'average' \
    --interval "$interval" \
    --station "ILL11" \
    --config_op "$hyp_option" \
    --task "comparison_baseline" \
    --smoothing 30


