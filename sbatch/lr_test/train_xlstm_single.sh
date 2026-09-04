#!/bin/bash
#SBATCH -t 96:00:00
#SBATCH --job-name=xlstm_lr_gc_sweep
#SBATCH --ntasks=1
#SBATCH --array=0-0%8              # 7 LRs * 4 grad_clip values = 28 combos, 4 concurrent
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:A40:1
#SBATCH --reservation=GPU
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/xlstm_lrgcsweep_%A_%a.txt
#SBATCH --error=./logs/err/xlstm_lrgcsweep_%A_%a.txt

# GFZ Configuration with GPUs
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Define the sweep grids
lrs=(5e-5)
grad_clips=(0.0)
# weight_decays=(0.0 0.05 0.01)

n_lrs=${#lrs[@]}
n_clips=${#grad_clips[@]}
n_wds=${#weight_decays[@]}

lr_idx=$(( SLURM_ARRAY_TASK_ID % n_lrs ))
gc_idx=$(( (SLURM_ARRAY_TASK_ID / n_lrs) % n_clips ))
wd_idx=$(( SLURM_ARRAY_TASK_ID / (n_lrs * n_clips) ))

lr=${lrs[$lr_idx]}
grad_clip=${grad_clips[$gc_idx]}
weight_decay=${weight_decays[$wd_idx]}

interval=5

srun --gres=gpu:A40:1 --unbuffered python /storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/functions/training/train_xlstm.py \
    --test_event_id "3" \
    --val_event_id "7" \
    --time_shift_mins "average" \
    --interval "$interval" \
    --station "ILL11" \
    --task "lr_test_lr${lr}_gc${grad_clip}" \
    --smoothing "60" \
    --config_op "v4" \
    --divide_by 45 \
    --repeat 4 \
    --lr "$lr" \
    --grad_clip "$grad_clip"