import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.fft import dct, idct
import math
from layers.lowrank import ComplexLowRank

class SWTExtractor(nn.Module):

    def __init__(self, kernel_size: int, in_features: int, channels: int =2):
        super(SWTExtractor, self).__init__()
        self.kernel_size = kernel_size # Kerne size of the filters.
        self.channels = channels # Number of filters.
        self.in_features = in_features # Number of input features.

        # Wavelet Signal Refinement Block.
        self.conv_DW = nn.Conv1d(in_channels=self.in_features,
                                out_channels=self.in_features * self.channels,
                                kernel_size=self.kernel_size,
                                stride=1,
                                groups=self.in_features,
                                padding="same",
                                bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        batch_size, _, _ = x.shape
        
        out = self.conv_DW(x)
        out = F.tanh(out)

        if self.channels > 1:
            coeff = out[:, (self.channels - 1)::self.channels, :]
        else:
            coeff = out

        return coeff

class Model(nn.Module):
    """
    Implements the CLAW architecture that comprises of WSR blocks for iterative signal refinement followed by Complex-valued low-rank linear layer for prediction .
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length.
        self.pred_len = configs.pred_len  # Prediction horizon.
        self.channels = configs.enc_in  # Number of Input features.

        self.individual = configs.individual  # Use separate model per channel.

        self.rank = configs.rank  # Rank for complex-valued low-rank linear layer.
        self.no_of_filters = configs.filters # Number of filters/channels in WSR block.
        self.filter_size = configs.filter_size  # Size of the filter/kernel.
        self.extractor_depth = configs.extractor_depth  # Number of WSR blocks.

        
        # Initialize the WSR blocks.
        self.swt = nn.ModuleList([SWTExtractor(kernel_size=self.filter_size, 
                                in_features=self.channels,
                                channels=self.no_of_filters) for _ in range(self.extractor_depth)])


        if self.individual:
            self.pred = nn.ModuleList(
                    [ComplexLowRank(in_features=self.seq_len//2 + 1,
                                    out_features=self.pred_len//2 + 1,
                                    bias=True,
                                    rank=self.rank) for _ in range(self.channels)]
                                    )
        else:
            self.pred = ComplexLowRank(in_features=self.seq_len//2 + 1,
                                    out_features=self.pred_len//2 + 1,
                                    bias=True,
                                    rank=self.rank)


    def forward(self, x):
        """
        x: Input tensor of shape [Batch, Input length, Channel]
        Returns: Output tensor of shape [Batch, Output length, Channel]
        """
        batch_size, _, _ = x.shape

        # Transpose input to [Batch, Channel, Input length]
        x = x.permute(0, 2, 1)
        
        # Compute mean for normalization
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean  # Normalize input
        
        # Apply WSR to clean the signal.
        for _ in range(self.extractor_depth):
            coefficients = self.swt[_](x)
            x -= coefficients

        # Convert to frequency domain using FFT.
        x = torch.fft.rfft(x, dim=-1, norm="backward")

        # Perform final prediction.
        if self.individual:
            out = torch.empty(batch_size, self.channels, self.pred_len//2+1, device=x.device)
            for i in range(self.channels):
                pred = self.pred[i](x[:, i, :].view(batch_size, 1, -1))
                out[:, i, :] = pred.view(batch_size, -1)
        else:
            out = self.pred(x)
       
        # Convert back to time domain using inverse FFT.
        out = torch.fft.irfft(out, dim=-1, norm="backward", n=self.pred_len)

        # De-normalize the output by adding back the mean.
        out = out + seq_mean
      
        return out.permute(0, 2, 1)  # Return output as [Batch, Output length, Channel]