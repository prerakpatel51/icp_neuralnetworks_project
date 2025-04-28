# Use this code to generate random 3D data samples with different patterns.
# It can be used in case when we dont have real data to train the model.


# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from typing import Tuple
# import logging

# logger = logging.getLogger(__name__)

# class Random3DDataset(Dataset):
#     def __init__(self, num_samples: int = 1000, shape: Tuple[int, int, int, int] = (1, 8, 360, 516)):
#         self.num_samples = num_samples
#         self.shape = shape
#         logger.info(f"Initialized Random3DDataset with {num_samples} samples of shape {shape}")

#     def __len__(self) -> int:
#         return self.num_samples
    
#     def __getitem__(self, idx: int) -> torch.Tensor:
#         try:
#             vol = torch.rand(*self.shape) * 0.2  # Base with low-amplitude noise
#             pattern_type = idx % 4
            
#             if pattern_type == 0:
#                 # Sinusoidal patterns
#                 for z in range(self.shape[1]):
#                     freq = 0.1 + 0.3 * torch.rand(1).item()
#                     phase = 2 * np.pi * torch.rand(1).item()
#                     x_pattern = torch.sin(freq * torch.linspace(0, 10, self.shape[3]) + phase)
#                     y_pattern = torch.cos(freq * torch.linspace(0, 8, self.shape[2]) + phase)
#                     vol[0,z] += x_pattern * y_pattern[:, None]
                    
#             elif pattern_type == 1:
#                 # Radial patterns
#                 center_x = 0.3 + 0.4 * torch.rand(1).item()
#                 center_y = 0.2 + 0.6 * torch.rand(1).item()
#                 for z in range(self.shape[1]):
#                     radius_var = 0.5 + 0.5 * (z / self.shape[1])
#                     x = torch.linspace(-1, 1, self.shape[3])
#                     y = torch.linspace(-1, 1, self.shape[2])
#                     xx, yy = torch.meshgrid(x, y, indexing='xy')
#                     rr = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
#                     vol[0,z] += torch.exp(-rr * (3 + 5 * radius_var))
                    
#             elif pattern_type == 2:
#                 # Checkerboard pattern
#                 freq = 5 + 10 * torch.rand(1).item()
#                 for z in range(self.shape[1]):
#                     x = torch.linspace(0, freq * np.pi, self.shape[3])
#                     y = torch.linspace(0, freq * np.pi, self.shape[2])
#                     xx, yy = torch.meshgrid(x, y, indexing='xy')
#                     vol[0,z] += 0.5 * (torch.sin(xx) * torch.sin(yy)).abs()
                    
#             else:
#                 # Random blobs
#                 num_blobs = 3 + int(3 * torch.rand(1).item())
#                 for _ in range(num_blobs):
#                     blob_z = int(self.shape[1] * torch.rand(1).item())
#                     blob_x = int(self.shape[3] * torch.rand(1).item())
#                     blob_y = int(self.shape[2] * torch.rand(1).item())
#                     sigma = 10 + 50 * torch.rand(1).item()
#                     x = torch.arange(self.shape[3]).float()
#                     y = torch.arange(self.shape[2]).float()
#                     gauss = torch.exp(-((x - blob_x)**2 + (y[:, None] - blob_y)**2) / (2 * sigma**2))
#                     vol[0,blob_z] += gauss
            
#             vol += 0.1 * torch.randn(*self.shape)
#             vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            
#             return vol
            
#         except Exception as e:
#             logger.error(f"Error generating sample {idx}: {str(e)}")
#             raise