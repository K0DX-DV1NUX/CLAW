import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct, idct
import math

class FFN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, dropout=0.0):
        super(FFN, self).__init__()
        if hidden_features is None:
            hidden_features = in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LowRank(nn.Module):
    """
    Implements a low-rank approximation layer using two smaller weight matrices (A and B).
    This reduces the number of parameters compared to a full-rank layer.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRank, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias

        # Initialize weight matrices A (in_features x rank) and B (rank x out_features)
        wA = torch.empty(self.in_features, rank)
        wB = torch.empty(self.rank, self.out_features)
        self.A = nn.Parameter(nn.init.kaiming_uniform_(wA))
        self.B = nn.Parameter(nn.init.kaiming_uniform_(wB))

        # Initialize bias if required
        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb))

    def forward(self, x):
        # Apply low-rank transformation: X * A * B
        out = x @ self.A
        out = out @ self.B
        if self.bias:
            out += self.b  # Add bias if enabled
        return out

# === Trend Extraction via Moving Average ===
class AvgPooling(nn.Module):
    """
    Extracts trend from the time series using average pooling,
    with front-padding to preserve alignment.
    """
    def __init__(self, kernel_size: int, stride: int):
        super(AvgPooling, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, 
                                stride=stride, 
                                padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        # Front-pad with the first value to preserve shape
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1)
        x = torch.cat([front, x], dim=-1)
        return self.avg(x)  # [batch_size, channels, seq_len]    
    
  
# === StationaryWaveletTransform ===
class SWTExtractor(nn.Module):
    """
    
    """

    def __init__(self, kernel_size: int, features: int, channels: int):
        super(SWTExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.features = features # Number of input features
        self.channels = channels # Number of channels to expand each feature to.

        # Depthwise convolution to expand each feature into multiple channels
        self.conv_DW = nn.Conv1d(in_channels=features,
                                out_channels=features * channels,
                                kernel_size=self.kernel_size,
                                stride=1,
                                groups=features,
                                padding="same",
                                bias=False)

        # Pointwise convolution to reduce the number of channels back to features
        self.conv_PW = nn.Conv1d(in_channels= features * channels,
                                out_channels=features,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=features,
                                bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        batch_size, _, _ = x.shape

        # Depthwise convolution to expand features into multiple channels
        out = self.conv_DW(x)

        # Reshape and apply softmax across channels
        out = out.reshape(batch_size, self.features, self.channels, -1)
        out = F.softmax(out, dim=2)

        # Reshape back and apply pointwise convolution
        out = out.reshape(batch_size, self.features * self.channels, -1)
        out = self.conv_PW(out)
        
        return out


    def orthogonality_regularizer(self) -> torch.Tensor:
        """
        Encourages each kernel to have unit norm (orthonormal-like behavior).
        """
        weight = self.conv_DW.weight.reshape(self.features, self.channels,-1)

        reg= torch.tensor(0.0, device=weight.device)
        for i in range(self.features):
            kernel = weight[i]
            norm = kernel @ kernel.T
            identity = torch.eye(self.channels, device=norm.device)
            reg += torch.norm(norm - identity, p='fro') ** 2
        
        return reg


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)

        encoder_depth = 3
        trend_kernel_size = 50 #configs.kernel_size
        num_seasons = configs.seasons
        rank = configs.rank

        self.avg_pooling = AvgPooling(kernel_size=trend_kernel_size, stride=1)

        # SWT extractor for trend and seasonal components
        self.swt_trend = SWTExtractor(kernel_size=16, features=self.channels, channels=8)
        self.swt_season = SWTExtractor(kernel_size=16, features=self.channels, channels=8)

        ## Prediction Layers
        self.pred_trend = nn.Linear(self.seq_len, self.pred_len)
        self.pred_season = nn.Linear(self.seq_len, self.pred_len)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)  # -> [B, C, L]
        seq_mean = torch.mean(x, dim=-1, keepdim=True)
        
        trend_input = self.avg_pooling(x)  # [B, C, L]
        season_input = x - trend_input  # Remove trend from input


        #trend_input += self.swt_trend(trend_input)
        season_input += self.swt_season(season_input)


        out_trend = self.pred_trend(trend_input)
        out_season = self.pred_season(season_input)

        out = out_trend + out_season  # Combine trend and season predictions

        out = out + seq_mean
        return out.permute(0,2,1)

    def custom_regularizer(self) -> torch.Tensor:
        reg_loss = torch.tensor(0.0)
        reg_loss += self.swt_trend.orthogonality_regularizer()
        reg_loss += self.swt_season.orthogonality_regularizer()

        return reg_loss

    # def reconstruction_loss(self) -> torch.Tensor:
    #     """
    #     Computes the reconstruction loss based on the trend and seasonal components.
    #     """
    #     if self.season_before is None or self.trend_before is None:
    #         return torch.tensor(0.0, device=self.trend_after.device)

    #     # Calculate the reconstruction loss
    #     trend_loss = F.mse_loss(self.trend_after, self.trend_before)
    #     season_loss = F.mse_loss(self.season_after, self.season_before)

    #     return trend_loss + season_loss