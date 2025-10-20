# First pair
jid1=$(sbatch ./comparison_baseline/train_xlstm.sh | awk '{print $4}')
sbatch --dependency=afterok:$jid1 ./comparison_baseline/train_xlstm_single.sh

# Second pair
jid2=$(sbatch ./comparison_baseline/train_lstm.sh | awk '{print $4}')
sbatch --dependency=afterok:$jid2 ./comparison_baseline/train_lstm_single.sh

# Third pair
jid3=$(sbatch ./comparison_baseline/train_linreg.sh | awk '{print $4}')
sbatch --dependency=afterok:$jid3 ./comparison_baseline/train_linreg_single.sh
