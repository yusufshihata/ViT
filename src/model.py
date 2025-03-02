import torch
import torch.nn as nn
from einops import repeat
from typing import Tuple, Optional

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
    def __init__(self, embed_size: int):
        super(FeedForward, self).__init__()
        self.embed_size = embed_size
        self.linear1 = nn.Linear(embed_size, embed_size * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(embed_size * 4, embed_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

class SelfAttentionHead(nn.Module):
    def __init__(self, embed_size: int):
        super(SelfAttentionHead, self).__init__()
        self.embed_size: int = embed_size
        self.qkv_proj: nn.Module = nn.Linear(embed_size, embed_size * 3)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        attention = Q @ K.transpose(-2, -1) / self.embed_size**0.5
        attention = self.softmax(attention)
        attention = attention @ V
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(embed_size) for _ in range(num_heads)])
        self.linear = nn.Linear(embed_size * num_heads, embed_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = [head(x) for head in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, embed_size: int, module: str = "linear", num_heads: Optional[int] = None):
        super(ResidualBlock, self).__init__()
        if module == "linear":
            self.layers = nn.ModuleList([
                nn.LayerNorm(embed_size),
                FeedForward(embed_size)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.LayerNorm(embed_size),
                MultiHeadAttention(embed_size, num_heads)
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers[0](x)
        out = self.layers[1](out)
        return x + out

class ViTLayer(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super(ViTLayer, self)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(embed_size, "multihead", num_heads),
            ResidualBlock(embed_size)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.residual_blocks:
            x = block(x)
        return x

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
    
    def forward(self, x):
        pass
