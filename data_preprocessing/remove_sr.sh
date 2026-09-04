#!/bin/bash
#SBATCH -t 1:00:00               # time limit: (HH:MM:SS)
#SBATCH --job-name=rem_resp     # job name
#SBATCH --ntasks=1               # each task in the job array will have a single task associated with it
#SBATCH --array=1-135%15            # job array id, adjusted for the total number of commands
#SBATCH --mem-per-cpu=8G         # Memory Request (per CPU; can use on GLIC)
#SBATCH --mail-type=all
#SBATCH --mail-user=kshitkar@gfz-potsdam.de
#SBATCH --chdir=/storage/vast-gfz-hpc-01/home/kshitkar/Impact_Force_Inversion/
#SBATCH --output=./logs/out/rem_resp_out_%A_%a.txt   # Standard Output Log File (for Job Arrays)
#SBATCH --error=./logs/err/rem_resp_err_%A_%a.txt    # Standard Error Log File (for Job Arrays)

source /home/kshitkar/miniforge3/bin/activate
conda activate xlstm_env

# Configuration
STATIONS=("ILL11" "ILL12" "ILL13")
commands=()

# Helper to populate command array
add_jobs() {
    local year=$1
    shift
    local juldays=("$@")
    for station in "${STATIONS[@]}"; do
        for julday in "${juldays[@]}"; do
            commands+=("python ./data_preprocessing/remove_sr.py --station $station --julday $julday --year $year")
        done
    done
}

add_jobs 2019 161 162 171 172 182 183 184 196 207 223 232
add_jobs 2020 156 159 160 162 168 169 181 210 229 230 243 276 277
add_jobs 2021 131 136 141 142 156 173 175 187 194 197 219 262
add_jobs 2022 156 181 185 195 221
add_jobs 2023 153 161 193 194

# Get the command to run for this task
command_to_run=${commands[$SLURM_ARRAY_TASK_ID-1]}

if [ -z "$command_to_run" ]; then
    echo "Error: Job index $SLURM_ARRAY_TASK_ID exceeds command list size (${#commands[@]})"
    exit 1
fi

# Print and run the command
echo "Running: $command_to_run"
srun $command_to_run
