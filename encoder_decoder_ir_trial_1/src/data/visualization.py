import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import imageio
import shutil
from datetime import datetime, timedelta
import json
from tqdm import tqdm
from src.models.vae import SimpleVAE3D

logger = logging.getLogger(__name__)

class VAEVisualizer:
    """Comprehensive visualization handler for 3D VAE training and evaluation."""
    
    def __init__(
        self,
        experiment_dir: str,
        max_experiments: int = 3,
        config: Optional[Dict] = None
    ):
        """
        Initialize visualizer with experiment directory.
        
        Args:
            experiment_dir: Path to the current experiment directory
            config: Optional configuration dictionary
        """
        self.experiment_dir = experiment_dir
        self.config = config or {}
        self.visualization_dir = os.path.join(experiment_dir, "visualizations")
         
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Setup visualization tracking
        self.visualization_log = os.path.join(self.visualization_dir, "visualization_log.json")
        self._init_visualization_log()
        
        logger.info(f"Visualizer initialized. Output directory: {self.visualization_dir}")

    def _init_visualization_log(self) -> None:
        """Initialize or load visualization tracking log."""
        if not os.path.exists(self.visualization_log):
            self.completed_visualizations = {
                'samples': [],
                'reconstructions': [],
                'animations': []
            }
            self._save_visualization_log()
        else:
            with open(self.visualization_log, 'r') as f:
                self.completed_visualizations = json.load(f)

    def _save_visualization_log(self) -> None:
        """Save the current state of visualization tracking."""
        with open(self.visualization_log, 'w') as f:
            json.dump(self.completed_visualizations, f, indent=2)

    def visualize_samples_from_loader(
        self,
        dataloader: DataLoader,
        num_samples: int = 5,
        save: bool = True,
        show: bool = False,
        prefix: str = "",
        force: bool = False
    ) -> Optional[str]:
        """
        Visualize samples from a DataLoader.
        
        Args:
            dataloader: DataLoader to visualize samples from
            num_samples: Number of samples to display
            save: Whether to save the visualization
            show: Whether to display the visualization
            prefix: Prefix for filename
            force: Force recreation even if exists
            
        Returns:
            Path to saved visualization if saved, else None
        """
        try:
            output_name = f"{prefix}dataset_samples.png" if prefix else "dataset_samples.png"
            output_path = os.path.join(self.visualization_dir, output_name)
            
            if not force and output_name in self.completed_visualizations['samples']:
                logger.info(f"Skipping existing samples visualization: {output_name}")
                return output_path if os.path.exists(output_path) else None
            
            # Get a batch from the dataloader
            batch = next(iter(dataloader))
           
            
            
            batch = batch[2]  # Assume first element is the data
            batch=batch.permute(0, 2, 1, 3, 4)  # Rearrange dimensions if needed
            
            
            ################
            
            
            
            ##################
            # Ensure we don't request more samples than available
            actual_samples = min(num_samples, batch.shape[0])
            
            fig, axes = plt.subplots(actual_samples, 16, figsize=(20, actual_samples*2.5))
            if actual_samples == 1:
                axes = axes[np.newaxis, :]
            
            for i in range(actual_samples):
                sample = batch[i].squeeze()
                
                for slice_idx in range(min(16, sample.shape[0])):
                    img = sample[slice_idx].numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    
                    axes[i, slice_idx].imshow(img, cmap='viridis')
                    axes[i, slice_idx].axis('off')
                    axes[i, slice_idx].set_title(f"Sample {i+1}\nSlice {slice_idx+1}")
            
            plt.tight_layout()
            
            if save:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                self.completed_visualizations['samples'].append(output_name)
                self._save_visualization_log()
                logger.info(f"Saved dataset samples to {output_path}")
            
            if show:
                plt.show()
            
            plt.close()
            return output_path if save else None
            
        except Exception as e:
            logger.error(f"Error visualizing samples: {str(e)}")
            raise

    def visualize_reconstructions(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        num_samples: int = 5,
        save: bool = True,
        show: bool = False,
        epoch: Optional[int] = None,
        force: bool = False
    ) -> Optional[str]:
        """
        Visualize original and reconstructed samples side-by-side.
        
        Args:
            model: Trained VAE model
            dataloader: DataLoader for the dataset
            device: Device to run model on
            num_samples: Number of samples to display
            save: Whether to save the visualization
            show: Whether to display the visualization
            epoch: Optional epoch number for naming
            force: Force recreation even if exists
            
        Returns:
            Path to saved visualization if saved, else None
        """
        try:
            fname = f"reconstructions_epoch{epoch:03d}.png" if epoch else "reconstructions.png"
            output_path = os.path.join(self.visualization_dir, fname)
            
            if not force and fname in self.completed_visualizations['reconstructions']:
                logger.info(f"Skipping existing reconstructions visualization: {fname}")
                return output_path if os.path.exists(output_path) else None
            
            model.eval()
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                batch = batch[2]
            batch = batch.to(device)
            batch = batch.permute(0, 2, 1, 3, 4)
            
            # Ensure we don't request more samples than available
            actual_samples = min(num_samples, batch.shape[0])
            batch = batch[:actual_samples]
            
            with torch.no_grad():
                reconstructions, _, _ = model(batch)
            
            batch = batch.cpu()
            reconstructions = reconstructions.cpu()
            
            fig, axes = plt.subplots(actual_samples, 2, figsize=(10, actual_samples*5))
            if actual_samples == 1:
                axes = axes[np.newaxis, :]
            
            for i in range(actual_samples):
                mid_slice = batch.shape[2] // 2
                original = batch[i, 0, mid_slice].numpy()
                recon = reconstructions[i, 0, mid_slice].numpy()
                
                original = (original - original.min()) / (original.max() - original.min() + 1e-8)
                recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
                
                axes[i, 0].imshow(original, cmap='viridis')
                axes[i, 0].set_title(f"Original {i+1}")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(recon, cmap='viridis')
                axes[i, 1].set_title(f"Reconstructed {i+1}")
                axes[i, 1].axis('off')
            
            plt.tight_layout()
            
            if save:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                self.completed_visualizations['reconstructions'].append(fname)
                self._save_visualization_log()
                logger.info(f"Saved reconstructions to {output_path}")
            
            if show:
                plt.show()
            
            plt.close()
            return output_path if save else None
            
        except Exception as e:
            logger.error(f"Error visualizing reconstructions: {str(e)}")
            raise

    def create_slice_animation(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        sample_idx: int = 0,
        fps: int = 5,
        prefix: str = "",
        force: bool = False
    ) -> Optional[str]:
        """
        Create an animation showing all slices of a 3D volume.
        
        Args:
            volume: 3D numpy array or tensor (z, y, x)
            sample_idx: Index of the sample (for naming)
            fps: Frames per second for animation
            prefix: Prefix for filename
            force: Force recreation even if exists
            
        Returns:
            Path to saved animation if saved, else None
        """
        try:
            if isinstance(volume, torch.Tensor):
                volume = volume.squeeze().cpu().numpy()
                
            output_name = f"{prefix}volume_{sample_idx}_animation.gif"
            output_path = os.path.join(self.visualization_dir, output_name)
            
            if not force and output_name in self.completed_visualizations['animations']:
                logger.info(f"Skipping existing animation: {output_name}")
                return output_path if os.path.exists(output_path) else None
            
            # Normalize volume
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
            
            # Create temp directory for frames
            temp_dir = os.path.join(self.visualization_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate frames with progress bar
            frames = []
            for z in tqdm(range(volume.shape[0]), desc="Generating animation frames"):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(volume[z], cmap='viridis')
                ax.set_title(f"Slice {z+1}/{volume.shape[0]}")
                ax.axis('off')
                
                frame_path = os.path.join(temp_dir, f"frame_{z:03d}.png")
                plt.savefig(frame_path, bbox_inches='tight', dpi=100)
                frames.append(imageio.imread(frame_path))
                plt.close()
            
            # Save animation
            imageio.mimsave(output_path, frames, fps=fps)
            self.completed_visualizations['animations'].append(output_name)
            self._save_visualization_log()
            logger.info(f"Saved animation to {output_path}")
            
            # Clean up temp files
            for frame in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, frame))
            os.rmdir(temp_dir)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating slice animation: {str(e)}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

################################################################################################################################################







    def save_full_dataset_reconstructions(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        force: bool = False
    ) -> str:
        """
        Save original and reconstructed samples for the entire dataset in organized directory structure.
        
        Args:
            model: Trained VAE model
            dataloader: DataLoader for the dataset
            device: Device to run model on
            force: Force recreation even if exists
            
        Returns:
            Path to the root directory containing the samples
        """
        try:
            # Create root directory
            root_dir = os.path.join(self.visualization_dir, "dataset_reconstructions")
            os.makedirs(root_dir, exist_ok=True)
            
            # Check if already exists
            if not force and os.path.exists(root_dir) and len(os.listdir(root_dir)) > 0:
                logger.info(f"Reconstructions already exist at {root_dir}. Use force=True to regenerate.")
                return root_dir
            
            model.eval()
            
            # Process entire dataset
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing dataset")):
                if isinstance(batch, (list, tuple)):
                    batch = batch[2]
                
                batch = batch.to(device)
                batch = batch.permute(0, 2, 1, 3, 4)
                
                with torch.no_grad():
                    reconstructions, _, _ = model(batch)
                
                # Convert to numpy and move to CPU
                originals = batch.cpu().numpy()
                reconstructions = reconstructions.cpu().numpy()
                
                # Save each sample in the batch
                for sample_idx in range(originals.shape[0]):
                    sample_dir = os.path.join(root_dir, f"sample_{batch_idx * dataloader.batch_size + sample_idx:04d}")
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Create subdirectories
                    orig_dir = os.path.join(sample_dir, "original")
                    recon_dir = os.path.join(sample_dir, "reconstructed")
                    os.makedirs(orig_dir, exist_ok=True)
                    os.makedirs(recon_dir, exist_ok=True)
                    
                    # Save each slice
                    for slice_idx in range(originals.shape[2]):
                        np.save(
                            os.path.join(orig_dir, f"slice_{slice_idx:03d}.npy"),
                            originals[sample_idx, :, slice_idx]
                        )
                        np.save(
                            os.path.join(recon_dir, f"slice_{slice_idx:03d}.npy"),
                            reconstructions[sample_idx, :, slice_idx]
                        )
            
            logger.info(f"Saved all dataset reconstructions to {root_dir}")
            return root_dir
            
        except Exception as e:
            logger.error(f"Error saving full dataset reconstructions: {str(e)}")
            raise







################################################################################################################################################


    def run_full_visualization_pipeline(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        epoch: Optional[int] = None,
        force: bool = False
    ) -> Dict[str, str]:
        """
        Run complete visualization pipeline using DataLoader.
        
        Args:
            model: Trained VAE model
            dataloader: DataLoader for visualization
            device: Device to run model on
            epoch: Optional epoch number for naming
            force: Force recreation of all visualizations
            
        Returns:
            Dictionary of paths to saved visualizations
        """
        results = {}
        
        # Get a batch to work with
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)):
            batch = batch[2]
        batch = batch.permute(0, 2, 1, 3, 4)
        
        # Determine actual number of samples available
        actual_samples = min(5, batch.shape[0])
        
        # 1. Dataset samples
        results['samples'] = self.visualize_samples_from_loader(
            dataloader,
            num_samples=actual_samples,
            prefix=f"epoch{epoch}_" if epoch else "",
            force=force
        )
        
        # 2. Reconstructions
        results['reconstructions'] = self.visualize_reconstructions(
            model,
            dataloader,
            device=device,
            num_samples=actual_samples,
            epoch=epoch,
            force=force
        )
        
        # 3. Slice animation
        sample = batch[0].squeeze()
        results['animation'] = self.create_slice_animation(
            sample,
            sample_idx=0,
            prefix=f"epoch{epoch}_" if epoch else "",
            force=force
        )
        
        # 4. Full dataset reconstructions
        results['full_reconstructions'] = self.save_full_dataset_reconstructions(
            model,
            dataloader,
            device=device,
            force=force
        )
        logger.info(f"Completed full visualization pipeline. Results: {results}")
        
        
        
        return results

    def load_and_visualize(
        self,
        model_path: str,
        dataloader: DataLoader,
        device: str = "cuda",
        force: bool = False
    ) -> Dict[str, str]:
        """
        Load a saved model and run complete visualization pipeline using DataLoader.
        
        Args:
            model_path: Path to saved model checkpoint
            dataloader: DataLoader for visualization
            device: Device to run model on
            force: Force recreation of all visualizations
            
        Returns:
            Dictionary of paths to saved visualizations
        """
        try:
            # Load model
            model = self._load_model(model_path, device)
            
            # Run full visualization pipeline
            return self.run_full_visualization_pipeline(
                model,
                dataloader,
                device=device,
                force=force
            )
            
        except Exception as e:
            logger.error(f"Error in load_and_visualize: {str(e)}")
            raise
        
        
        
        
############################################################        
        
    def _load_model(
    self,
    model_path: str,
    device: str = "cuda"
) -> torch.nn.Module:
        """Internal method to load a saved model."""
        try:
            # Initialize model
            model = SimpleVAE3D(
                input_channels=self.config.get('data', {}).get('input_shape', [1])[0],
                latent_dim=self.config.get('model', {}).get('latent_dim', 32)
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if the checkpoint contains "model_state_dict"
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # Assume the checkpoint is the state_dict itself
                state_dict = checkpoint
            
            # Handle DataParallel if needed
            if all(k.startswith('module.') for k in state_dict.keys()):
                model = torch.nn.DataParallel(model, device_ids=[0, 1])
                # Remove 'module.' prefix if present
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            # Load the state_dict into the model
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            logger.info(f"Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise        
            
            
        
        
        
##########################################################

    # def _load_model(
    #     self,
    #     model_path: str,
    #     device: str = "cuda"
    # ) -> torch.nn.Module:
    #     """Internal method to load a saved model."""
    #     try:
    #         # Initialize model
    #         model = SimpleVAE3D(
    #             input_channels=self.config.get('data', {}).get('input_shape', [1])[0],
    #             latent_dim=self.config.get('model', {}).get('latent_dim', 32)
    #         )
            
    #         # Load checkpoint
    #         checkpoint = torch.load(model_path, map_location=device)
            
    #         # Extract model_state_dict
    #         if "model_state_dict" in checkpoint:
    #             state_dict = checkpoint["model_state_dict"]
    #         else:
    #             raise KeyError("The checkpoint does not contain 'model_state_dict'")
            
    #         # Handle DataParallel if needed
    #         if all(k.startswith('module.') for k in state_dict.keys()):
    #             model = torch.nn.DataParallel(model, device_ids=[0, 1])
    #             # Remove 'module.' prefix if present
    #             state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
    #         # Load the state_dict into the model
    #         model.load_state_dict(state_dict)
    #         model.to(device)
    #         model.eval()
            
    #         logger.info(f"Successfully loaded model from {model_path}")
    #         return model
            
    #     except Exception as e:
    #         logger.error(f"Error loading model: {str(e)}")
    #         raise




########################################################
    # def _load_model(
    #     self,
    #     model_path: str,
    #     device: str = "cuda"
    # ) -> torch.nn.Module:
    #     """Internal method to load a saved model."""
    #     try:
    #         # Initialize model
    #         model = SimpleVAE3D(
    #             input_channels=self.config.get('data', {}).get('input_shape', [1])[0],
    #             latent_dim=self.config.get('model', {}).get('latent_dim', 32)
    #         )
            
    #         # Load state dict
    #         state_dict = torch.load(model_path, map_location=device)
            
    #         # Handle DataParallel if needed
    #         if all(k.startswith('module.') for k in state_dict.keys()):
    #             model = torch.nn.DataParallel(model, device_ids=[0,1])
            
    #         model.load_state_dict(state_dict)
    #         model.to(device)
    #         model.eval()
            
    #         logger.info(f"Successfully loaded model from {model_path}")
    #         return model
            
    #     except Exception as e:
    #         logger.error(f"Error loading model: {str(e)}")
    #         raise