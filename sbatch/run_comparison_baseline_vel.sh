# First pair
jid1=$(sbatch ./comparison_baseline_vel/train_xlstm.sh | awk '{print $4}')
sbatch --dependency=afterok:$jid1 ./comparison_baseline_vel/train_xlstm_single.sh

# Second pair
jid2=$(sbatch ./comparison_baseline_vel/train_lstm.sh | awk '{print $4}')
sbatch --dependency=afterok:$jid2 ./comparison_baseline_vel/train_lstm_single.sh
