import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedOrthogonalConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 3 * in_channels
        self.kernel_size = kernel_size

        # Define 3 orthogonal 1x5 kernels
        base_kernels = torch.tensor([
            [1, 0, 0, 0, 0],
            [0, 1, -1, 1, -1],
            [1, -2, 0, 2, -1],
        ], dtype=torch.float32)

        # Normalize to ensure orthogonality
        base_kernels = F.normalize(base_kernels, dim=1)

        # Repeat for each input channel
        kernels = base_kernels[:, None, :]  # [3, 1, 5]
        kernels = kernels.repeat(in_channels, 1, 1)  # [3*in_channels, 1, 5]

        self.register_buffer('weight', kernels)

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        B, C, L = x.shape
        x = x.view(B * C, 1, L)  # [B*C, 1, L]
        out = F.conv1d(x, self.weight, padding=self.kernel_size // 2, groups=1)  # [B*C*3, 1, L]
        out = out.view(B, 3 * C, L)  # [B, 3C, L]
        return out

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)
        
       
        self.fixed_conv = FixedOrthogonalConv1D(self.channels, kernel_size=5)
        self.pred = nn.Linear(self.seq_len, self.pred_len)

    # x: [batch_size, seq_len, channels] 
    def forward(self, x):
        
        x = x.permute(0, 2, 1) # [batch_size, channels, seq_len]

        # Compute mean for normalization
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean  # Normalize input

        x = self.fixed_conv(x)  # [B, 3C, L]
        x = self.pred(x)        # [B, 3C, pred_len]

        x = x.view(x.size(0), self.channels, 3, self.pred_len)  # [B, C, 3, pred_len]
        x = x.sum(dim=2)  # Combine: [B, C, pred_len]

        x = x + seq_mean  # Add back mean
        x = x.permute(0, 2, 1)  # [B, pred_len, C]

        return x  # [batch_size, pred_len, channels]