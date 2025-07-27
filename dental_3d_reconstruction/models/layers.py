"""
Custom neural network layers for dental 3D reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock3D(nn.Module):
    """3D Residual Block for volumetric processing."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionGate3D(nn.Module):
    """3D Attention Gate for focusing on relevant features."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate3D, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SelfAttention3D(nn.Module):
    """3D Self-Attention mechanism for long-range dependencies."""
    
    def __init__(self, in_channels: int):
        super(SelfAttention3D, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, D, H, W = x.size()
        
        # Generate query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, D * H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, D * H * W)
        
        # Attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        
        out = self.gamma * out + x
        return out


class SpectralNormalization(nn.Module):
    """Spectral Normalization for stable GAN training."""
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super(SpectralNormalization, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height,-1).data, v.data))
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class PixelShuffle3D(nn.Module):
    """3D Pixel Shuffle for upsampling 3D tensors."""
    
    def __init__(self, upscale_factor: int):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, in_depth, in_height, in_width = x.size()
        nOut = channels // (self.upscale_factor ** 3)
        
        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor
        
        input_view = x.contiguous().view(batch_size, nOut, self.upscale_factor, self.upscale_factor, self.upscale_factor, in_depth, in_height, in_width)
        
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4)
        return output.contiguous().view(batch_size, nOut, out_depth, out_height, out_width)
