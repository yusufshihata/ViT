import unittest
import torch
from src.model import PatchEmbedding

class PatchEmbeddingTest(unittest.TestCase):
    def test_forward(self):
        patch_res = 16
        img_shape = (3, 224, 224)
        patch_size = ((img_shape[1] * img_shape[2]) // patch_res**2) + 1
        latent_size = 512
        img = torch.randn(1, *img_shape)
        patch_embedding = PatchEmbedding(patch_res, img_shape, latent_size)
        output = patch_embedding(img)
        self.assertEqual(output.shape, torch.Size([1, patch_size, latent_size]))

if __name__ == "__main__":
    unittest.main()
