#!/bin/bash
#SBATCH -t 1:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=rem_resp     # job name
#SBATCH --ntasks=1               # each task in the job array will have a single task associated with it
#SBATCH --array=1-11            # job array id, adjusted for the total number of commands
#SBATCH --mem-per-cpu=2G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/rem_resp_out_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/rem_resp_err_%A_%a.txt    # Standard Error Log File (for Job Arrays)

source /home/kshitkar/miniforge3/bin/activate
conda activate seismic_cal

commands=()

stations=("ILL11")
juldays=(161 162 171 172 182 183 184 196 207 223 232)
for station in "${stations[@]}"; do
    for julday in "${juldays[@]}"; do
        commands+=("python remove_sr.py --station $station --julday $julday")
    done
done
# Get the command to run for this task
command_to_run=${commands[$SLURM_ARRAY_TASK_ID-1]}

# Print and run the command
echo "Running: $command_to_run"
srun $command_to_run

