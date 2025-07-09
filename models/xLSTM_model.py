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
