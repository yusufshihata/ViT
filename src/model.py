import torch
import torch.nn as nn
from einops import repeat
from typing import Tuple, Optional

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, img_shape: Tuple[int, int, int], embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_channels = img_shape[0]
        self.img_height, self.img_width = img_shape[1], img_shape[2]
        
        self.num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels=self.num_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.register_buffer(
            'positional_embedding', 
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.positional_embedding_initialized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        x = self.proj(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        seq_len = x.size(1)
        
        if not self.positional_embedding_initialized or self.positional_embedding.size(1) != seq_len + 1:
            new_pos_embed = torch.randn(1, seq_len + 1, self.embed_dim, device=x.device)
            
            nn.init.trunc_normal_(new_pos_embed, std=0.02)
            
            self.register_buffer('positional_embedding', new_pos_embed, persistent=True)
            self.positional_embedding_initialized = True
        
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.positional_embedding
        
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(
        self, 
        img_shape: Tuple[int, int, int], 
        patch_size: int, 
        embed_dim: int, 
        num_heads: int, 
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, img_shape, embed_dim)
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        
        x = self.transformer(x)
        
        x = self.norm(x)
        
        return x[:, 0]

class ViTClassifier(nn.Module):
    def __init__(self, vit_model: ViT, embed_dim: int, num_classes: int = 10):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embed_vector = self.vit(x)
        return self.classifier(embed_vector)
