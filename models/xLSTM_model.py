import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class xLSTMRegressor(nn.Module):
    """ xLSTM-based regression model for time series prediction. """
    def __init__(self,
                 input_size,
                 conv1d_kernel_size=4, 
                 qkv_proj_blocksize=4, 
                 num_heads=4,
                 context_length=10, 
                 num_blocks=2, 
                 embedding_dim=512, 
                 slstm_at=[1],
                 num_reps: int = 1):
        super(xLSTMRegressor, self).__init__()
        """Initialize the xLSTMRegressor model.
        Args:
            input_size (int): Size of the input features.
            conv1d_kernel_size (int): Kernel size for Conv1D in mLSTM layers.
            qkv_proj_blocksize (int): Block size for QKV projection in mLSTM layers.
            num_heads (int): Number of attention heads.
            context_length (int): Length of the input sequences.
            num_blocks (int): Number of xLSTM blocks.
            embedding_dim (int): Dimension of the embedding layer.
            slstm_at (list): List of block indices where sLSTM layers are applied.
            num_reps (int): Number of times to repeat the xLSTM block stack.
        """
        
        self.embedding = nn.Linear(input_size, embedding_dim)
        backend = "vanilla"
        
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, 
                    qkv_proj_blocksize=qkv_proj_blocksize, 
                    num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,  # <-- since we're concatenating prev_targets
            slstm_at=slstm_at,
        )

        self.xlstm_layers = nn.ModuleList([xLSTMBlockStack(self.cfg) for _ in range(num_reps)])
        self.fc = nn.Linear(embedding_dim, 1)
        # self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
            prev_targets (torch.Tensor): Previous target values, shape (batch_size, 20, 1)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.embedding(x)  # (batch_size, 20, embedding_dim)

        for xlstm in self.xlstm_layers:
            x = xlstm(x)  # (batch_size, 20, embedding_dim + 1)

        x = x[:, -1, :]  # (batch_size, embedding_dim + 1)
        output = self.fc(x)  # (batch_size, 1)
        
        return F.softplus(output)
    
class xLSTMRegressor_v2(nn.Module):
    """ xLSTM-based regression model for time series prediction. 
    Added positional encoding and more MLP layers and max pooling. """
    def __init__(self,
                 input_size,
                 conv1d_kernel_size=4, 
                 qkv_proj_blocksize=4, 
                 num_heads=4,
                 context_length=20,
                 num_blocks=2, 
                 embedding_dim=512, 
                 slstm_at=[1],
                 num_reps: int = 1):
        super(xLSTMRegressor_v2, self).__init__()
        """Initialize the xLSTMRegressor_v2 model.
        Args:
            input_size (int): Size of the input features.
            conv1d_kernel_size (int): Kernel size for Conv1D in mLSTM layers.
            qkv_proj_blocksize (int): Block size for QKV projection in mLSTM layers.
            num_heads (int): Number of attention heads.
            context_length (int): Length of the input sequences.
            num_blocks (int): Number of xLSTM blocks.
            embedding_dim (int): Dimension of the embedding layer.
            slstm_at (list): List of block indices where sLSTM layers are applied.
            num_reps (int): Number of times to repeat the xLSTM block stack.
        """

        self.embedding = nn.Linear(input_size, embedding_dim)

        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(torch.randn(1, context_length, embedding_dim))

        backend = "vanilla"
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, 
                    qkv_proj_blocksize=qkv_proj_blocksize, 
                    num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=slstm_at,
        )

        self.xlstm_layers = nn.ModuleList([xLSTMBlockStack(self.cfg) for _ in range(num_reps)])

        # Output MLP head after pooling
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 48),
            nn.Linear(48, 1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, input_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        x = self.embedding(x)  # (batch_size, 20, embedding_dim)
        x = x + self.positional_embedding  # Add learnable positional encoding

        for xlstm in self.xlstm_layers:
            x = xlstm(x)  # (batch_size, 20, embedding_dim)

        # Max pooling over time
        x, _ = torch.max(x, dim=1)  # (batch_size, embedding_dim)

        output = self.fc(x)  # (batch_size, 1)
        output = F.softplus(output)
        return output  # raw output, optionally apply softplus externally if needed
    
    def get_embeddings(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, input_size)
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, 20, embedding_dim)
        """
        x = self.embedding(x)
        return x
    
    def get_xlstm_embeddings(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, input_size)
        Returns:
            torch.Tensor: Embedding tensor from the last xLSTM layer, shape (batch_size, 20, embedding_dim)
        """
        x = self.embedding(x) + self.positional_embedding
        for xlstm in self.xlstm_layers:
            x = xlstm(x)
        return x

# class xLSTMRegressor_conv1d(nn.Module):
#     """xLSTM regressor that uses a 1D convolution applied over the input feature dimension
#     (input_size) for each time step. Embedding dimension for the xLSTM layers is 92 by default.
#     """
#     def __init__(self,
#             input_size,
#             temporal_conv_kernel_size: int = 3,
#             conv1d_kernel_size: int = 4,
#             qkv_proj_blocksize: int = 4,
#             num_heads: int = 4,
#             context_length: int = 20,
#             num_blocks: int = 2,
#             embedding_dim: int = 96,
#             slstm_at=[1],
#             num_reps: int = 1):
#         super(xLSTMRegressor_conv1d, self).__init__()

#         # Convolution across the input feature dimension for each time step.
#         # We'll treat each time step's feature vector as a 1D signal of length `input_size`.
#         padding = (temporal_conv_kernel_size - 1) // 2
#         # Use in_channels=1 because we feed each feature vector as a single-channel signal.
#         self.conv_embedding = nn.Conv1d(in_channels=1,
#                         out_channels=embedding_dim,
#                         kernel_size=temporal_conv_kernel_size,
#                         padding=padding)

#         backend = "vanilla"
#         self.cfg = xLSTMBlockStackConfig(
#             mlstm_block=mLSTMBlockConfig(
#             mlstm=mLSTMLayerConfig(
#                 conv1d_kernel_size=conv1d_kernel_size,
#                 qkv_proj_blocksize=qkv_proj_blocksize,
#                 num_heads=num_heads
#             )
#             ),
#             slstm_block=sLSTMBlockConfig(
#             slstm=sLSTMLayerConfig(
#                 backend=backend,
#                 num_heads=num_heads,
#                 conv1d_kernel_size=conv1d_kernel_size,
#                 bias_init="powerlaw_blockdependent",
#             ),
#             feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
#             ),
#             context_length=context_length,
#             num_blocks=num_blocks,
#             embedding_dim=embedding_dim,
#             slstm_at=slstm_at,
#         )

#         self.xlstm_layers = nn.ModuleList([xLSTMBlockStack(self.cfg) for _ in range(num_reps)])
#         self.fc = nn.Linear(embedding_dim, 1)

#     def forward(self, x):
#         # x: (batch, seq_len, input_size)
#         b, seq_len, in_size = x.shape
#         # Reshape to treat each time-step feature vector as a single-channel 1D signal:
#         x = x.contiguous().view(b * seq_len, 1, in_size)   # -> (b*seq_len, 1, input_size)
#         x = self.conv_embedding(x)                         # -> (b*seq_len, embedding_dim, L_out)
#         # Pool across the feature-length dimension to produce one embedding vector per time step
#         x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)        # -> (b*seq_len, embedding_dim)
#         x = x.view(b, seq_len, -1)                         # -> (batch, seq_len, embedding_dim)

#         for xlstm in self.xlstm_layers:
#             x = xlstm(x)

#         x = x[:, -1, :]                      # take last time step
#         out = self.fc(x)
#         return F.softplus(out)