# First pair
jid1=$(sbatch --constraint=grid-redundant ./cb_traintest/train_xlstm.sh | awk '{print $4}')
sbatch --constraint=grid-redundant --dependency=afterok:$jid1 ./cb_traintest/train_xlstm_single.sh

# # Second pair
# jid2=$(sbatch ./cb_traintest/train_lstm.sh | awk '{print $4}')
# sbatch --dependency=afterok:$jid2 ./cb_traintest/train_lstm_single.sh
