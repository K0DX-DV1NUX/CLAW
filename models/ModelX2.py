import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.fft import dct, idct
import math


class SWTExtractor(nn.Module):

    def __init__(self, kernel_size: int, in_features: int):
        super(SWTExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.channels = 2 # Number of input features
        self.in_features = in_features # Number of channels to expand each feature to.

        # Depthwise convolution to expand each feature into multiple channels
        self.conv_DW = nn.Conv1d(in_channels=self.in_features,
                                out_channels=self.in_features * self.channels,
                                kernel_size=self.kernel_size,
                                stride=1,
                                groups=self.in_features,
                                padding="same",
                                bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        batch_size, _, _ = x.shape
        
        out = self.conv_DW(x) #/ math.sqrt(self.kernel_size)
        out = F.tanh(out)

        coef1 = out[:, ::2, :]
        coef2 = out[:, 1::2, :]

        return coef1, coef2
    
    def orthogonality_regularizer(self) -> torch.Tensor:
        """
        Encourages each kernel to have unit norm (orthonormal-like behavior).
        """
        weight = self.conv_DW.weight.reshape(self.in_features, self.channels,-1)

        reg= torch.tensor(0.0, device=weight.device)
        for i in range(self.in_features):
            kernel = weight[i]
            norm = kernel @ kernel.T
            identity = torch.eye(self.channels, device=norm.device)
            reg += torch.norm(norm - identity, p='fro') ** 2
        
        return reg
    
    def variance_regularizer(self) -> torch.Tensor:
        weight = self.conv_DW.weight.reshape(self.in_features, self.channels, -1)
        coef1_weights = weight[:, ::2, :]
        coef2_weights = weight[:, 1::2, :]

        coef2_var = coef2_weights.var(dim=-1).mean()
        coef1_var = coef1_weights.var(dim=-1).mean()

        return  -coef1_var

    def covariance_regularizer(self) -> torch.Tensor:
        reg = torch.Tensor([0.0])
        weight = self.conv_DW.weight.reshape(self.in_features, self.channels, -1)
        for i in range(self.in_features):
            filter_weight = weight[i]
            covar_mat = filter_weight.cov()

            covariance = (covar_mat - torch.diag(torch.diag(covar_mat))).triu().flatten()


            # Positive Covariance test
            reg -= covariance.pow(2).sum()

            # Negative Covariance Test
            # identity = torch.ones(covariance.shape)
            # negative_covar = identity - covariance.pow(2)
            # reg += negative_covar.sum()

        return reg


class NyquistLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=30, bias=True):
        super(NyquistLinear, self).__init__()
        self.bias = bias
        
        Pr = torch.empty(in_features, rank)
        Pi = torch.empty(in_features, rank)

        nn.init.uniform_(Pr, a=-math.sqrt(1 / in_features), b=math.sqrt(1 / in_features))
        nn.init.uniform_(Pi, a=-math.sqrt(1 / in_features), b=math.sqrt(1 / in_features))

        Qr = torch.empty(rank, out_features)
        Qi = torch.empty(rank, out_features)

        nn.init.uniform_(Qr, a=-math.sqrt(1 / out_features), b=math.sqrt(1 / out_features))
        nn.init.uniform_(Qi, a=-math.sqrt(1 / out_features), b=math.sqrt(1 / out_features))

        P = torch.complex(Pr, Pi)
        Q = torch.complex(Qr, Qi)

        self.P = nn.Parameter(P, requires_grad=True)
        self.Q = nn.Parameter(Q, requires_grad=True)


        if bias:
            br = torch.empty(out_features)
            bi = torch.empty(out_features)
            nn.init.uniform_(br, a=-math.sqrt(1 / in_features), b=math.sqrt(1 / in_features))
            nn.init.uniform_(bi, a=-math.sqrt(1 / in_features), b=math.sqrt(1 / in_features))

            b = torch.complex(br, bi)
            self.b = nn.Parameter(b, requires_grad=True)
        else:
            self.register_parameter('b', None)


    def forward(self, x):

        out = x @ self.P @ self.Q

        if self.b is not None:
            out = out + self.b

        mask = torch.complex(torch.zeros(out.shape), torch.zeros(out.shape))
        mask[:,:,0].imag = out[:,:,0].imag

        out = out - mask

        return out


class Model(nn.Module):
    """
    Implements the HADL framework with optional Haar wavelet transformation, Discrete Cosine Transform (DCT) and Low Rank Approximation.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)
        self.rank = 30  # Rank for low-rank coef1imation
        self.bias = 1  # Whether to include bias
        self.individual = 0  # Use separate models per channel
        self.enable_Haar = 1  # Enable Haar transformation
        self.enable_DCT = 0  # Enable Discrete Cosine Transform
        self.enable_iDCT = 0  # Enable Inverse Discrete Cosine Transform
        self.enable_lowrank = 0  # Enable low-rank coef1imation
        self.patch_size = 16
        self.depth = 4

        self.dwt = nn.ModuleList([SWTExtractor(8, in_features=self.channels) for _ in range(self.depth)])


        self.pred = NyquistLinear(in_features=self.seq_len//2 + 1, 
                                out_features=self.pred_len//2 + 1, 
                                bias=self.bias,
                                rank=15)
        

    def forward(self, x):
        """
        Forward pass of the model.
        x: Input tensor of shape [Batch, Input length, Channel]
        Returns: Output tensor of shape [Batch, Output length, Channel]
        """
        batch_size, _, _ = x.shape

        # Transpose input to [Batch, Channel, Input length]
        x = x.permute(0, 2, 1)
        
        # Compute mean for normalization
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean  # Normalize input
        
        for i in range(self.depth):
            coef1, coef2 = self.dwt[i](x)
            x -= coef2


        x = torch.fft.rfft(x, dim=-1, norm="backward")

        out = self.pred(x)
       
        out = torch.fft.irfft(out, dim=-1, norm="backward", n=self.pred_len)

        # De-normalize the output by adding back the mean
        out = out + seq_mean
      
        return out.permute(0, 2, 1)  # Return output as [Batch, Output length, Channel]
    

    def regularizer(self):

        reg = torch.tensor([0.0])
        # for i in range(self.depth):
        #     reg += self.dwt[i].covariance_regularizer()
        return  1.0 * reg