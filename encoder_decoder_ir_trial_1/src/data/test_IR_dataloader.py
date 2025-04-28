from data_provider import IMERGDataModule
import numpy as np
import os
import matplotlib.pyplot as plt

# event id for which the data was downloaded
event_id = 'WA'

# location of the h5 file that was generated after downloading the data
h5_dataset_location = '/home1/ppatel2025/ppworktp/encoder/encoder_decoder_imerg_trial_3/src/data/dataset/WA_dataset.h5'

# as of now, we do not have IR data, so we set it None
ir_h5_dataset_location = '/home1/ppatel2025/ppworktp/encoder/encoder_decoder_ir_trial_1/src/data/dataset/WA_IR.h5'

# this string is used to determine the kind of dataloader we need to use
# for processing individual events, we reccommend the user to keep this fixed
dataset_type = 'wa_ir'


data_provider =  IMERGDataModule(
        forecast_steps = 12,
        history_steps = 12,
        imerg_filename = h5_dataset_location,
        ir_filename = ir_h5_dataset_location,
        batch_size = 6,
        image_shape = (360, 516),
        normalize_data=False,
        dataset = dataset_type,
        production_mode = False,
        )


train_data_loader = data_provider.train_dataloader()
test_data_loader = data_provider.test_dataloader()
val_data_loader = data_provider.val_dataloader()

import os
import matplotlib.pyplot as plt
import torch

# # Create a directory to save images
# output_dir = "sample_timesteps_visualization"
# os.makedirs(output_dir, exist_ok=True)

# # Get a batch from the dataloader
# batch = next(iter(train_data_loader))
# input_seq = batch[2]  # Shape: (6, 16, 1, 360, 516)

# # Select a sample (e.g., first in the batch)
# sample_idx = 0  
# sample = input_seq[sample_idx]  # Shape: (16, 1, 360, 516)

# # Loop through each timestep and save
# for timestep in range(sample.shape[0]):
#     # Extract image (squeeze removes channel dim if 1)
#     image = sample[timestep].squeeze().cpu().numpy()  # Shape: (360, 516)
    
#     # Normalize to [0, 1] if needed (skip if already normalized)
#     if image.min() < 0 or image.max() > 1:
#         image = (image - image.min()) / (image.max() - image.min())
    
#     # Plot and save
#     plt.figure(figsize=(10, 8))
#     plt.imshow(image, cmap='viridis')  # Use 'gray' for grayscale
#     plt.colorbar()
#     plt.title(f"Sample {sample_idx}, Timestep {timestep}")
#     plt.axis('off')
    
#     # Save as PNG
#     save_path = os.path.join(output_dir, f"sample_{sample_idx}_timestep_{timestep}.png")
#     plt.savefig(save_path, bbox_inches='tight', dpi=100)
#     plt.close()

# print(f"Saved {sample.shape[0]} images to {output_dir}")


import os
import matplotlib.pyplot as plt
import torch

# Create output directory
output_dir = "raw_timesteps_visualization"
os.makedirs(output_dir, exist_ok=True)

# Get a batch from the dataloader
batch = next(iter(train_data_loader))
input_seq = batch[2]  # Shape: (6, 16, 1, 360, 516)

# Select a sample (e.g., first in the batch)
sample_idx = 0  
sample = input_seq[sample_idx]  # Shape: (16, 1, 360, 516)

# Loop through each timestep and save
for timestep in range(sample.shape[0]):
    # Extract image (remove channel dim if 1)
    image = sample[timestep].squeeze().cpu().numpy()  # Shape: (360, 516)
    
    # Plot raw values (no normalization)
    plt.figure(figsize=(10, 8))
    img_plot = plt.imshow(image, cmap='viridis')  # Use 'gray' for grayscale
    
    # Add a colorbar to see actual values
    plt.colorbar(img_plot, label="Raw Pixel Value")
    plt.title(f"Sample {sample_idx}, Timestep {timestep} (No Normalization)")
    plt.axis('off')
    
    # Save as PNG
    save_path = os.path.join(output_dir, f"sample_{sample_idx}_timestep_{timestep}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

print(f"Saved {sample.shape[0]} raw images to {output_dir}")