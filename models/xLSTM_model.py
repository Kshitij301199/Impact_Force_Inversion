import torch.nn as nn

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
    def __init__(self,
                 input_size,
                 conv1d_kernel_size=4, 
                 qkv_proj_blocksize=4, 
                 num_heads=4,
                 context_length=10, 
                 num_blocks=2, 
                 embedding_dim=512, 
                 slstm_at=[1],
                 num_reps: int = 1):  # num_reps controls the number of xLSTM layers
        super(xLSTMRegressor, self).__init__()
        
        # Embedding layer to reduce input size from 3000 to embedding_dim
        self.embedding = nn.Linear(input_size, embedding_dim)
        backend = "vanilla"
        
        # xLSTM configuration
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, qkv_proj_blocksize=qkv_proj_blocksize, num_heads=num_heads
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

        # Stack multiple xLSTM layers based on num_reps
        self.xlstm_layers = nn.ModuleList([xLSTMBlockStack(self.cfg) for _ in range(num_reps)])
        
        # Fully connected layer for the final output
        self.fc = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, 3000)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Pass through the embedding layer
        x = self.embedding(x)  # Shape: (batch_size, 20, embedding_dim)
        
        # Pass through each xLSTM layer
        for xlstm in self.xlstm_layers:
            x = xlstm(x)  # Shape: (batch_size, 20, embedding_dim)

        # Take the hidden state of the last timestep
        x = x[:, -1, :]  # Shape: (batch_size, embedding_dim)
        
        # Pass through the fully connected layer to get a single output
        output = self.fc(x)  # shape: (batch_size, 1)
        
        return output
