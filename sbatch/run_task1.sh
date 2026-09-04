# First pair
jid1=$(sbatch --constraint=grid-redundant ./task1/train_xlstm.sh | awk '{print $4}')
sbatch --constraint=grid-redundant --dependency=afterok:$jid1 ./task1/train_xlstm_single.sh

# # Second pair
jid2=$(sbatch --constraint=grid-redundant ./task1/train_lstm.sh | awk '{print $4}')
sbatch --constraint=grid-redundant --dependency=afterok:$jid2 ./task1/train_lstm_single.sh
