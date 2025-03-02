import unittest
import torch
from src.model import PatchEmbedding, SelfAttentionHead

class ModelsTest(unittest.TestCase):
    def test_patch_embedding_forward(self):
        batch_size = 10
        patch_res = 16
        img_shape = (3, 224, 224)
        patch_size = ((img_shape[1] * img_shape[2]) // patch_res**2) + 1
        latent_size = 512
        img = torch.randn(batch_size, *img_shape)
        patch_embedding = PatchEmbedding(patch_res, img_shape, latent_size)
        output = patch_embedding(img)
        self.assertEqual(output.shape, torch.Size([batch_size, patch_size, latent_size]))
    
    def test_self_attention_head(self):
        batch_size = 10
        patch_res = 16
        img_shape = (3, 224, 224)
        patch_size = ((img_shape[1] * img_shape[2]) // patch_res**2) + 1
        latent_size = 512
        img = torch.randn(batch_size, *img_shape)
        patch_embedding = PatchEmbedding(patch_res, img_shape, latent_size)
        output = patch_embedding(img)
        self_attention_head = SelfAttentionHead(latent_size)
        output = self_attention_head(output)
        self.assertEqual(output.shape, torch.Size([batch_size, patch_size, latent_size]))        

if __name__ == "__main__":
    unittest.main()
