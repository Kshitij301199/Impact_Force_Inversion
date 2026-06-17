#!/bin/bash
#SBATCH -t 96:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=base_xlstm           # job name
#SBATCH --ntasks=1                # each task in the job array will have a single task associated with it
#SBATCH --array=1-35%6            # 1 interval * 7 events * 1 div * 2 hyps * 3 smooths = 42
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
intervals=(5)
event_ids=(1 3 4 6 7 8 9)
divide_bys=(45)
hyp_options=("v1" "v2" "v3" "v4" "v5")
smoothings=(60)

# Decompose SLURM_ARRAY_TASK_ID into indices
task_index=$(( SLURM_ARRAY_TASK_ID - 1 ))

n_smooth=${#smoothings[@]}
n_div=${#divide_bys[@]}
n_event=${#event_ids[@]}
n_hyp=${#hyp_options[@]}

smoothing_idx=$(( task_index % n_smooth ))
task_index=$(( task_index / n_smooth ))

divide_by_idx=$(( task_index % n_div ))
task_index=$(( task_index / n_div ))

event_id_idx=$(( task_index % n_event ))
task_index=$(( task_index / n_event ))

hyp_option_idx=$(( task_index % n_hyp ))
task_index=$(( task_index / n_hyp ))

interval_idx=$(( task_index ))

interval="${intervals[$interval_idx]}"
hyp_option="${hyp_options[$hyp_option_idx]}"
event_id="${event_ids[$event_id_idx]}"
divide_by="${divide_bys[$divide_by_idx]}"
smoothing="${smoothings[$smoothing_idx]}"

echo "------------------------------------------------"
echo "Running Task Index: $SLURM_ARRAY_TASK_ID"
echo "Event: $event_id | Interval: $interval | Hyp: $hyp_option"
echo "Divide By: $divide_by"
echo "Smoothing: $smoothing"

mkdir -p logs/out logs/err

# Run the Python script with the selected parameters
srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/training/train_xlstm.py \
    --test_event_id "$event_id" \
    --val_event_id "$event_id" \
    --time_shift_mins "average" \
    --interval "$interval" \
    --station "ILL11" \
    --config_op "$hyp_option" \
    --task "comparison_baseline_tt" \
    --smoothing "$smoothing" \
    --divide_by "$divide_by" \
    --repeat 1 
