import torch
import torch.nn as nn
import torch.nn.functional as F

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
class TrendExtractor(nn.Module):
    """
    Extracts trend from the time series using average pooling,
    with front-padding to preserve alignment.
    """
    def __init__(self, kernel_size: int, stride: int):
        super(TrendExtractor, self).__init__()
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
    
  
# === Seasonal Pattern Extraction via Depthwise Convolution ===
class SeasonalExtractor(nn.Module):
    """
    
    """

    def __init__(self, kernel_size: int, channels: int):
        super(SeasonalExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.channels = channels

        self.season = nn.Conv1d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                groups=channels,
                                padding="same",
                                bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        return self.season(x)


    def orthogonality_regularizer(self) -> torch.Tensor:
        """
        Encourages each kernel to have unit norm (orthonormal-like behavior).
        """
        kernel = self.season.weight  # shape: [C, 1, K]
        flat_kernels = kernel.view(self.channels, -1)  # [C, K]
        norms = flat_kernels @ flat_kernels.T  # [C, C] Gram matrix
        identity = torch.eye(self.channels, device=norms.device)
        return ((norms - identity)**2).mean()
        # return torch.norm(norms-identity, p="fro")**2



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)
        self.gating_type = "soft"

        encoder_depth = configs.decomposer_depth
        trend_kernel_size = configs.kernel_size
        num_seasons = configs.seasons
        rank = configs.rank

        self.trend_extractor = TrendExtractor(kernel_size=trend_kernel_size)
        self.seasonal_extractor = SeasonalExtractor(kernel_size=9, channels=self.channels)

        # Gating to combine multiple seasonal filters
        self.gating_type = configs.gating_type  # "soft" or "hard"
        self.gate = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, kernel_size=1),
            nn.Sigmoid() if self.gating_type == "soft" else nn.Softmax(dim=1)
        )

        # Final prediction layer: maps denoised seq to future horizon
        self.pred_trend = nn.Linear(self.seq_len, self.pred_len)
        self.pred_seasonal = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)  # -> [B, C, L]
        seq_mean = torch.mean(x, dim=-1, keepdim=True)
        
        trend = self.trend_extractor(x)
        seasonal_input = x - trend

        seasonal_components = self.seasonal_extractor(seasonal_input)  # [B, C, L]

        gate_weights = self.gate(seasonal_components)  # [B, C, L]
        if self.gating_type == "hard":
            # Convert to hard gate: select max filter per position
            max_gate = F.one_hot(gate_weights.argmax(dim=1), num_classes=self.channels)  # [B, L, C]
            gate_weights = max_gate.permute(0, 2, 1).float()  # [B, C, L]

        # Apply gate to seasonal components (elementwise)
        seasonal = seasonal_components * gate_weights  # [B, C, L]

        # Combine across filters/channels
        seasonal = seasonal.sum(dim=1, keepdim=True)  # [B, 1, L]
        trend = trend.sum(dim=1, keepdim=True)        # [B, 1, L]

        out_season = self.pred_seasonal(seasonal)
        out_trend = self.pred_trend(seasonal)

    def symmetry_regularizer(self) -> torch.Tensor:
        return self.encoder.symmetry_regularizer()