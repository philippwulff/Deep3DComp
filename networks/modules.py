import torch.nn as nn
import torch

class Sine(nn.Module):
    """
    From SIREN:
    https://github.com/vsitzmann/siren/blob/master/modules.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class Encoding3D(nn.Module):
    """
    Implementation of Gaussian random Fourier features following 
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" from Tancik et al.
    https://github.com/tancik/fourier-feature-networks
    """
    def __init__(self, encoding_features, sigma) -> None:
        super(Encoding3D, self).__init__()
        # m
        self.encoding_features = encoding_features
        # Standard deviation.
        self.sigma = sigma
        # B ∈ Rm×d ~ N(0,σ2)
        self.B = nn.Parameter(torch.normal(0, sigma**2, (self.encoding_features, 3), requires_grad=False))
        # self.B *= self.sigma
    
    def forward(self, input):
        # Sec. 4, second paragraph: 
        # gamma = [a1 cos(2pi * b1^T @ v), a1 sin(2pi * b1^T @ v), ...]
        # Output shape is 2*m = 2*encoding_features
        input_proj = 2 * torch.pi * input @ self.B.T#.to(input.device)
        output = torch.concatenate((torch.sin(input_proj), torch.cos(input_proj)), dim=-1)
        return output
    

