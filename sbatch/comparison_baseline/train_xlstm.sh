#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=base_xlstm           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-112%4            # job array id, adjusted for the total number of commands (8 test days * 7 validation days * 4 intervals)
#SBATCH --mem-per-cpu=16G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A40:1             # load GPU A100 could be replace by _A40/_A40, 509-510 nodes has 4_A100_80G
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
event_ids=(1 3 4 5 6 7 8 9)
hyp_options=('default')
# hyp_options=('mlstm' 'slstm')
# smoothings=(0 30 60)
smoothings=(30)

# Calculate the total number of combinations per test day
num_event_ids=${#event_ids[@]}
num_val_ids=$(( num_event_ids - 1 ))
num_intervals=${#intervals[@]}
num_smoothings=${#smoothings[@]}
num_hyp_options=${#hyp_options[@]}
num_combinations=$(( num_val_ids * num_intervals * num_smoothings * num_hyp_options ))

# Calculate the indices for the current task
test_id_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / $num_combinations ))
remaining_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % $num_combinations ))

val_id_idx=$(( $remaining_idx / (num_intervals * num_smoothings * num_hyp_options) ))
remaining_idx=$(( $remaining_idx % (num_intervals * num_smoothings * num_hyp_options) ))

interval_idx=$(( $remaining_idx / (num_smoothings * num_hyp_options) ))
remaining_idx=$(( $remaining_idx % (num_smoothings * num_hyp_options) ))

smoothing_idx=$(( $remaining_idx / num_hyp_options ))
hyp_option_idx=$(( $remaining_idx % num_hyp_options ))

# Get the current test event id
test_event_id=${event_ids[$test_id_idx]}

# Get the validation event ids (exclude the test event id)
val_event_ids=("${event_ids[@]:0:$test_id_idx}" "${event_ids[@]:$((test_id_idx + 1))}")
val_event_id=${val_event_ids[$val_id_idx]}

# Get the current interval, smoothing, and hyp option
interval=${intervals[$interval_idx]}
smoothing=${smoothings[$smoothing_idx]}
hyp_option=${hyp_options[$hyp_option_idx]}

# Log the current parameters
echo "Running for:"
echo "Test Event ID: $test_event_id"
echo "Validation Event ID: $val_event_id"
echo "Interval: $interval"
echo "Smoothing: $smoothing"
echo "Hyp Option: $hyp_option"

# Run the Python script with the selected parameters
srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/training/train_xlstm.py \
    --test_event_id "$test_event_id" \
    --val_event_id "$val_event_id" \
    --time_shift_mins 'average' \
    --interval "$interval" \
    --station "ILL11" \
    --task "comparison_baseline" \
    --smoothing "$smoothing" \
    --config_op "$hyp_option" \
    --divide_by 20
