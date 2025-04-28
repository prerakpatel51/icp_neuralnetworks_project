import os
import torch
import argparse
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.models.vae import SimpleVAE3D
from src.data.data_provider import IMERGDataModule
from src.data.visualization import VAEVisualizer
from torch.utils.data import Subset
from torch.utils.data import DataLoader


# from src.data.dataset import Random3DDataset
def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run post-training visualizations for 3D VAE')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file used for training')
    parser.add_argument('--output_dir', type=str, default='outputs/post_training_analysis',
                       help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=4,  # Changed to match batch size
                       help='Number of samples to use for visualization')
    parser.add_argument('--force', action='store_true',
                       help='Force recreation of existing visualizations')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)
    
    # Load config
    config = load_config(args.config)
    
    # Initialize data provider
    h5_dataset_location = '/home1/ppatel2025/ppworktp/encoder/encoder_decoder_imerg_trial_2/src/data/dataset/WA_dataset.h5'
    ir_h5_dataset_location = None
    dataset_type = 'wa_expanded'

    data_provider = IMERGDataModule(
        forecast_steps=12,
        history_steps=12,
        imerg_filename=h5_dataset_location,
        ir_filename=ir_h5_dataset_location,
        batch_size=10,  # Make sure this matches num_samples
        image_shape=(360, 516),
        normalize_data=False,
        dataset=dataset_type
    )


    
    # # Get the train dataloader
    # train_dataloader = data_provider.train_dataloader()
    
     
     
     
    #   # # Create a subset of the training dataset
    # train_dataset = data_provider.train_dataset  # Assuming `train_dataset` is accessible from `data_provider`
    # train_subset = Subset(train_dataset, range(200))  # Use the first 100 samples or create as per your requirement
    # # Create DataLoader for the subset
    # train_dataloader = DataLoader(
    #         train_subset,
    #         batch_size=config['data']['batch_size'],
    #         shuffle=False,
    #         num_workers=config['data']['num_workers'],
    #         pin_memory=config['data']['pin_memory'],
    #         persistent_workers=config['data']['persistent_workers']
    #     )


    test_data_loader = data_provider.test_dataloader()
    # train_data_loader = data_provider.train_dataloader()
    # val_data_loader = data_provider.val_dataloader()
        
     
     
     
     
     
     
     
    # # # Create dataset
    # dataset = Random3DDataset(
    #     num_samples=args.num_samples,
    #     shape=config['data']['input_shape']
    # )
    
    # # Create dataloader
    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=6,
    #     shuffle=True
    # )
    
    
    # Initialize visualizer
    visualizer = VAEVisualizer(
        experiment_dir=args.output_dir,
        config=config
    )
    
    # Run all visualizations
    try:
        print("\n" + "="*50)
        print("Running post-training visualizations")
        print(f"Model: {args.model_path}")
        print(f"Output directory: {args.output_dir}")
        print("="*50 + "\n")
        
        results = visualizer.load_and_visualize(
            model_path=args.model_path,
            dataloader=test_data_loader,
            device="cuda",
            force=args.force
        )
        
        print("\nVisualization results:")
        for name, path in results.items():
            if path:
                print(f"- {name}: {path}")
        
        print("\nPost-training visualization completed successfully!")
        
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()