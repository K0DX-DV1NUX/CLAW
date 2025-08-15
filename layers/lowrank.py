import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLowRank(nn.Module):
    """
    Implements a complex low-rank approximation layer using two smaller weight matrices (P and Q). This reduces the number of parameters compared to a full-rank complex valued linear layer.
    """
    def __init__(self, in_features, out_features, rank=30, bias=True):
        super(ComplexLowRank, self).__init__()
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

        return out

class LowRank(nn.Module):
    """
    Implements a low-rank approximation layer using two smaller weight matrices (A and B). This reduces the number of parameters compared to a full-rank layer.
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
        self.A = nn.Parameter(nn.init.uniform_(wA, a= (-1 / math.sqrt(self.in_features)), b=(1 / math.sqrt(self.in_features))))
        self.B = nn.Parameter(nn.init.uniform_(wB, a= (-1 / math.sqrt(self.out_features)), b=(1 / math.sqrt(self.out_features))))

        # Initialize bias if required
        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb, a= (-1 / math.sqrt(self.out_features)), b=(1 / math.sqrt(self.out_features))))

    def forward(self, x):
        # Apply low-rank transformation: X * A * B
        out = x @ self.A @ self.B
        if self.bias:
            out += self.b  # Add bias if enabled
        return out