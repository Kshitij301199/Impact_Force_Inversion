#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=s_v_m           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-336             # job array id, adjusted for the total number of commands (8 test days * 7 validation days * 4 intervals)
#SBATCH --mem-per-cpu=16G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A40:1             # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU            # reserve the GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/svm_out_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/svm_err_%A_%a.txt    # Standard Error Log File (for Job Arrays)

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Define the arrays
intervals=(5 15 30)
juldays=(161 172 182 183 196 207 223 232)
hyp_options=('mlstm' 'slstm')

# Calculate the total number of combinations per test day
num_val_days=$(( ${#juldays[@]} - 1 ))
num_intervals=${#intervals[@]}
num_hyp_options=${#hyp_options[@]}
num_combinations=$(( num_val_days * num_intervals * num_hyp_options ))

# Calculate the indices for the current task
test_day_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / $num_combinations ))
remaining_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % $num_combinations ))

val_day_idx=$(( $remaining_idx / (num_intervals * num_hyp_options) ))
remaining_idx=$(( $remaining_idx % (num_intervals * num_hyp_options) ))

interval_idx=$(( $remaining_idx / $num_hyp_options ))
hyp_option_idx=$(( $remaining_idx % $num_hyp_options ))

# Get the current test day
test_julday=${juldays[$test_day_idx]}

# Get the validation days (exclude the test day)
val_juldays=("${juldays[@]:0:$test_day_idx}" "${juldays[@]:$((test_day_idx + 1))}")
val_julday=${val_juldays[$val_day_idx]}

# Get the current interval
interval=${intervals[$interval_idx]}

# Get the current hyperparameter option
hyp_option=${hyp_options[$hyp_option_idx]}

# Log the current parameters
echo "Running for:"
echo "Test Julian Day: $test_julday"
echo "Validation Julian Day: $val_julday"
echo "Interval: $interval"
echo "Hypothesis Option: $hyp_option"

# Run the Python script with the selected parameters
srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/train_xlstm.py \
    --test_julday "$test_julday" \
    --val_julday "$val_julday" \
    --interval "$interval" \
    --station "ILL11" \
    --config_op "$hyp_option" \
    --task "sLSTM_v_mLSTM"

