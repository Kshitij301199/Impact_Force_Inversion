# First pair
jid1=$(sbatch --constraint=grid-redundant ./lr_test/train_xlstm.sh | awk '{print $4}')
sbatch --constraint=grid-redundant --dependency=afterok:$jid1 ./lr_test/train_xlstm_single.sh

# # Second pair
jid2=$(sbatch --constraint=grid-redundant ./lr_test/train_lstm.sh | awk '{print $4}')
sbatch --constraint=grid-redundant --dependency=afterok:$jid2 ./lr_test/train_lstm_single.sh

