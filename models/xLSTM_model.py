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
    
class xLSTMRegressor_v2(nn.Module):
    """ Added positional encoding and more MLP layers """
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

        # x = x.mean(dim=1)  # Mean pooling over sequence (batch_size, embedding_dim)
        x = x[:, -1, :]  # (batch_size, embedding_dim + 1)

        output = self.fc(x)  # (batch_size, 1)
        return output  # raw output, optionally apply softplus externally if needed

class xLSTMRegressor_v3(nn.Module):
    """ Added attention pooling """
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
        super(xLSTMRegressor_v3, self).__init__()

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

        self.attn_pool = nn.Sequential(
            nn.Linear(embedding_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

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

        # Attention pooling
        attn_scores = self.attn_pool(x)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        x = (x * attn_weights).sum(dim=1)  # (B, D)

        output = self.fc(x)  # (B, 1)
        return output  # raw output, optionally apply softplus externally if needed
    
