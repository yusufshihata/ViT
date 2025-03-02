import torch
import torch.nn as nn
from einops import repeat
from typing import Tuple

class PatchEmbedding(nn.Module):
    def __init__(self, patch_res: int, img_shape: Tuple[int, int, int], embed_size: int):
        super(PatchEmbedding, self).__init__()
        self.patch_res: int = patch_res
        self.num_channels: int = img_shape[0]
        self.patch_size: int = (img_shape[1] * img_shape[2]) // self.patch_res**2
        self.embed_size: int = embed_size

        self.proj: nn.Module = nn.Conv2d(in_channels=self.num_channels, out_channels=self.embed_size, kernel_size=self.patch_res, stride=self.patch_res)
        self.positional_embedding: nn.Parameter = nn.Parameter(torch.randn(1, self.patch_size + 1, self.embed_size))
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, self.embed_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positional_embedding
        return x

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
    
    def forward(self, x):
        pass

class HeadAttention(nn.Module):
    def __init__(self):
        super(HeadAttention, self).__init__()
    
    def forward(self, x):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
    
    def forward(self, x):
        pass

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
    
    def forward(self, x):
        pass

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
    
    def forward(self, x):
        pass
