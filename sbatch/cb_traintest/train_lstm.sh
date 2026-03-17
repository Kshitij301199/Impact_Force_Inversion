#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH --job-name=base_lstm
#SBATCH --ntasks=1
#SBATCH --array=1-32%4  # 2 intervals * 1 hyp_option * 8 event_ids * 5 divide_bys * 1 smoothings = 80
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:A40:1
#SBATCH --reservation=GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/lstm_out_%A_%a.txt
#SBATCH --error=./logs/err/lstm_err_%A_%a.txt

module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

intervals=(15 30)
event_ids=(1 3 4 5 6 7 8 9)
divide_bys=(45)
hyp_options=("v1" "v2" "v3" "v4")
smoothings=(0 30 60)

num_intervals=${#intervals[@]}
num_hyp_options=${#hyp_options[@]}
num_event_ids=${#event_ids[@]}
num_divide_bys=${#divide_bys[@]}
num_smoothings=${#smoothings[@]}

total_combinations=$((num_intervals * num_hyp_options * num_event_ids * num_divide_bys * num_smoothings))

task_index=$((SLURM_ARRAY_TASK_ID - 1))

interval_idx=$((task_index / (num_hyp_options * num_event_ids * num_divide_bys * num_smoothings) % num_intervals))
hyp_option_idx=$((task_index / (num_event_ids * num_divide_bys * num_smoothings) % num_hyp_options))
event_id_idx=$((task_index / (num_divide_bys * num_smoothings) % num_event_ids))
divide_by_idx=$((task_index / num_smoothings % num_divide_bys))
smoothing_idx=$((task_index % num_smoothings))

interval=${intervals[$interval_idx]}
hyp_option=${hyp_options[$hyp_option_idx]}
event_id=${event_ids[$event_id_idx]}
divide_by=${divide_bys[$divide_by_idx]}
smoothing=${smoothings[$smoothing_idx]}

echo "Running for:"
echo "Event ID (Test): $event_id"
echo "Event ID (Validation): $event_id"
echo "Interval: $interval"
echo "Hypothesis Option: $hyp_option"
echo "Divide By: $divide_by"
echo "Smoothing: $smoothing"

srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/training/train_lstm.py \
    --test_event_id "$event_id" \
    --val_event_id "$event_id" \
    --time_shift_mins 'average' \
    --interval "$interval" \
    --station "ILL11" \
    --config_op "$hyp_option" \
    --task "comparison_baseline_tt" \
    --smoothing "$smoothing" \
    --divide_by "$divide_by" \
    --repeat 1