import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3000, embedding_size=1024, hidden_size=128, num_layers=2):
        super(LSTMRegressor, self).__init__()
        
        self.embedding = nn.Linear(input_size, embedding_size)
        
        # LSTM input size becomes embedding + 1 (for prev_targets)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
            prev_targets (torch.Tensor): Previous target values, shape (batch_size, 20, 1)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.embedding(x)  # (batch_size, 20, 1024)

        lstm_out, (hn, cn) = self.lstm(x)
        
        final_hidden_state = hn[-1]  # (batch_size, hidden_size)
        output = self.fc(final_hidden_state)  # (batch_size, 1)
        
        return F.softplus(output)
