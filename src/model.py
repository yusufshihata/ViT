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
        
        # Calculate number of patches based on image dimensions
        self.num_patches = (self.img_height // patch_size) * (self.img_width // patch_size)
        self.embed_dim = embed_dim

        # Projection layer: maps each patch to the embedding dimension
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Pre-initialize positional embedding with expected size
        self.register_buffer(
            'positional_embedding', 
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.positional_embedding_initialized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]  # Batch size
        
        # Project patches
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        
        # Flatten and transpose: [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Get actual sequence length after projecting
        seq_len = x.size(1)
        
        # Dynamically initialize positional embeddings if needed
        if not self.positional_embedding_initialized or self.positional_embedding.size(1) != seq_len + 1:
            # Create new positional embedding with the right size
            new_pos_embed = torch.randn(1, seq_len + 1, self.embed_dim, device=x.device)
            
            # Initialize it with random values
            nn.init.trunc_normal_(new_pos_embed, std=0.02)
            
            # Replace the buffer with the new tensor
            self.register_buffer('positional_embedding', new_pos_embed, persistent=True)
            self.positional_embedding_initialized = True
        
        # Expand class token to batch size and prepend to sequence
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding (already on the same device as x)
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
        B, N, C = x.shape  # batch, sequence length, embedding dimension
        
        # Project to query, key, value and reshape to multi-head format
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
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
        # First residual block: attention
        x = x + self.attn(self.norm1(x))
        # Second residual block: MLP
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, num_layers: int, patch_res: int, img_shape: Tuple[int, int, int]):
        super(ViT, self).__init__()
        self.embedding = PatchEmbedding(patch_res, img_shape, embed_size)
        self.layers = nn.ModuleList([ViTLayer(embed_size, num_heads) for _ in range(num_layers)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x[:, 0]

class ViTClassifier(nn.Module):
    def __init__(self, vit_model: ViT, embed_dim: int, num_classes: int = 10):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embed_vector = self.vit(x)
        return self.classifier(embed_vector)
