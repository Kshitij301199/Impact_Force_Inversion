import torch
import torch.nn as nn
import torch.nn.functional as F

# class LSTMRegressor(nn.Module):
#     """LSTM-based regression model for time series prediction."""
#     def __init__(self, input_size=3000, embedding_size=1024, hidden_size=128, num_layers=2):
#         """Initialize the LSTMRegressor model.
#         Args:
#             input_size (int): Size of the input features (default: 3000).
#             embedding_size (int): Size of the embedding layer (default: 1024).
#             hidden_size (int): Number of features in the hidden state of LSTM (default: 128).
#             num_layers (int): Number of recurrent layers in LSTM (default: 2).
#         """
#         super(LSTMRegressor, self).__init__()
        
#         self.embedding = nn.Linear(input_size, embedding_size)
        
#         # LSTM input size becomes embedding + 1 (for prev_targets)
#         self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
#         self.fc = nn.Linear(hidden_size, 1)
#         # self.fc2 = nn.Linear(2, 1)

#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, 1)
#         """
#         x = self.embedding(x)  # (batch_size, 20, 1024)

#         lstm_out, (hn, cn) = self.lstm(x)
#         # print(f"LSTM output shape: {lstm_out.shape}, Hidden state shape: {hn.shape}, Cell state shape: {cn.shape}")
        
#         final_hidden_state = hn[-1]  # (batch_size, hidden_size)
#         output = self.fc(final_hidden_state)  # (batch_size, 1)
#         output = F.softplus(output)
#         return output
    
class LSTMRegressor(nn.Module):
    """
    LSTM-based regession model for time series inversion with positional encoding and final output.
    """
    def __init__(self, input_size, context_length=60, embedding_size=256, hidden_size=256, num_layers=2):
        """
        Initialize the LSTMRegressor_v2 model.
        Args:
            input_size (int): Size of the input features (default: 3000).
            embedding_size (int): Size of the embedding layer (default: 1024).
            hidden_size (int): Number of features in the hidden state of LSTM (default: 128).
            num_layers (int): Number of recurrent layers in LSTM (default: 2).
        """
        super(LSTMRegressor, self).__init__()
        
        self.embedding = nn.Linear(input_size, embedding_size)

        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, context_length, embedding_size))

        # LSTM input size becomes embedding + 1 (for prev_targets)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Output MLP head after pooling
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 48),
            nn.Linear(48, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.embedding(x)  # (batch_size, 20, embedding_size)
        x = x + self.positional_embedding  # Add positional encoding

        lstm_out, (hn, cn) = self.lstm(x)
        # print(f"LSTM output shape: {lstm_out.shape}, Hidden state shape: {hn.shape}, Cell state shape: {cn.shape}")
        # Use the final hidden state for regression
        final_hidden_state = hn[-1]  # (batch_size, hidden_size) taking the output of the last layer
        # print(f"Final hidden state shape: {final_hidden_state.shape}")

        output = self.fc(final_hidden_state)  # (batch_size, 1)
        # print(f"Output shape after MLP head: {output.shape}")
        output = F.softplus(output)
        return output
        
class LSTMRegressor_v2(nn.Module):
    """
    LSTM-based regession model for time series inversion with positional encoding and max pooling.
    """
    def __init__(self, input_size, context_length=60, embedding_size=256, hidden_size=256, num_layers=2):
        """
        Initialize the LSTMRegressor_v2 model.
        Args:
            input_size (int): Size of the input features (default: 3000).
            embedding_size (int): Size of the embedding layer (default: 1024).
            hidden_size (int): Number of features in the hidden state of LSTM (default: 128).
            num_layers (int): Number of recurrent layers in LSTM (default: 2).
        """
        super(LSTMRegressor_v2, self).__init__()
        
        self.embedding = nn.Linear(input_size, embedding_size)

        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, context_length, embedding_size))

        # LSTM input size becomes embedding + 1 (for prev_targets)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Output MLP head after pooling
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.Linear(48, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.embedding(x)  # (batch_size, 20, embedding_size)
        x = x + self.positional_embedding  # Add positional encoding

        lstm_out, (hn, cn) = self.lstm(x)
        # print(f"LSTM output shape: {lstm_out.shape}, Hidden state shape: {hn.shape}, Cell state shape: {cn.shape}")
        # Apply max pooling over the time dimension
        x, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_size)
        # print(f"Pooled output shape: {x.shape}")

        output = self.fc(x)  # (batch_size, 1)
        # print(f"Output shape after MLP head: {output.shape}")
        output = F.softplus(output)
        return output

    def get_embeddings(self, x):
            """
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 20, input_size)
            Returns:
                torch.Tensor: Embedding tensor of shape (batch_size, 20, embedding_dim)
            """
            x = self.embedding(x)
            return x
        
    def get_lstm_embeddings(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, input_size)
        Returns:
            torch.Tensor: Embedding tensor from the last xLSTM layer, shape (batch_size, 20, embedding_dim)
        """
        x = self.embedding(x) + self.positional_embedding
        lstm_out, (hn, cn) = self.lstm(x)
        x, _ = torch.max(lstm_out, dim=1)  # (batch_size, embedding_dim)
        # x = x[:, -1, :]  # (batch_size, embedding_dim)
        return x