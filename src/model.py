import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from typing import Tuple

class PatchEmbedding(nn.Module):
    def __init__(self, patch_res: int, img_shape: Tuple[int, int, int], latent_size: int):
        super(PatchEmbedding, self).__init__()
        self.patch_res = patch_res
        self.num_channels = img_shape[0]
        self.patch_size = (img_shape[1] * img_shape[2]) // self.patch_res**2
        self.flattened_size = self.patch_size * self.patch_res * self.patch_res * self.num_channels
        self.latent_size = latent_size

        self.linear_proj = nn.Linear(self.flattened_size, self.latent_size)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = img.view(self.patch_size, self.patch_res, self.patch_res, self.num_channels)
        img = rearrange(img, 'n h w c -> n (h w) c')
        img = img.flatten()
        img = self.linear_proj(img)
        return img