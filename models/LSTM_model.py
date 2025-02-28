import torch.nn as nn
import torch.nn.functional as F

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3000, embedding_size=1024, hidden_size=128, num_layers=2):
        """
        Args:
            input_size (int): The size of each timestep input (3000).
            embedding_size (int): The size of the embedding (1024).
            hidden_size (int): The number of features in the hidden state of the LSTM.
            num_layers (int): Number of recurrent layers in the LSTM.
        """
        super(LSTMRegressor, self).__init__()
        
        # Embedding layer to reduce input size from 3000 to 1024
        self.embedding = nn.Linear(input_size, embedding_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for the final output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Pass through the embedding layer
        x = self.embedding(x)  # Shape: (batch_size, 20, 1024)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out shape: (batch_size, 20, hidden_size)
        
        # Take the hidden state of the last timestep
        final_hidden_state = hn[-1]  # shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer to get a single output
        output = self.fc(final_hidden_state)  # shape: (batch_size, 1)
        
        return F.softplus(output)


