import argparse
from src.utils.config import load_config
from src.utils.logger import setup_logging
# from src.data.dataset import Random3DDataset
from src.models.vae import SimpleVAE3D
from src.training.trainer import VAETrainer
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import logging
from src.data.data_provider import IMERGDataModule
from torch.utils.data import Subset
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='3D VAE Training Script')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Ensure output directory exists
        os.makedirs(config['experiment']['output_dir'], exist_ok=True)
        
        # Setup logging
        setup_logging(config['experiment']['output_dir'])
        
        # Initialize components
        # dataset = Random3DDataset(
        #     num_samples=config['data']['num_samples'],
        #     shape=config['data']['input_shape']
        # )
        
        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=config['data']['batch_size'],
        #     shuffle=True,
        #     num_workers=config['data']['num_workers'],
        #     pin_memory=config['data']['pin_memory'],
        #     persistent_workers=config['data']['persistent_workers']
        # )
        ######################################################################################################################################
        
        
        h5_dataset_location='/home1/ppatel2025/ppworktp/encoder/encoder_decoder_imerg_trial_2/src/data/dataset/WA_dataset.h5'
            # as of now, we do not have IR data, so we set it None
        ir_h5_dataset_location = None

            # this string is used to determine the kind of dataloader we need to use
            # for processing individual events, we reccommend the user to keep this fixed
        dataset_type = 'wa_expanded'


        data_provider =  IMERGDataModule(
                    forecast_steps = 12,
                    history_steps = 12,
                    imerg_filename = h5_dataset_location,
                    ir_filename = ir_h5_dataset_location,
                    batch_size = 8,
                    image_shape = (360, 516),
                    normalize_data=False,
                    dataset = dataset_type)


        test_data_loader = data_provider.test_dataloader()
        train_data_loader = data_provider.train_dataloader()
        val_data_loader = data_provider.val_dataloader()
        
        
                
                # Check the size of each dataset
        train_dataset_size = len(data_provider.train_dataset)
        val_dataset_size = len(data_provider.val_dataset)
        test_dataset_size = len(data_provider.test_dataset)

        print(f"Training dataset size: {train_dataset_size}")
        print(f"Validation dataset size: {val_dataset_size}")
        print(f"Test dataset size: {test_dataset_size}")
                
                
        
        
        
        
        # code copied as akshay suggested to use it.
##########################################################

     
        # # Create a subset of the training dataset
        # train_dataset = data_provider.train_dataset  # Assuming `train_dataset` is accessible from `data_provider`
        # train_subset = Subset(train_dataset, range(1000))  # Use the first 100 samples or create as per your requirement
        # # Create DataLoader for the subset
        # train_data_loader = DataLoader(
        #     train_subset,
        #     batch_size=config['data']['batch_size'],
        #     shuffle=False,
        #     num_workers=config['data']['num_workers'],
        #     pin_memory=config['data']['pin_memory'],
        #     persistent_workers=config['data']['persistent_workers']
        # )

            # Example batch from train_data_loader:
        # for batch in train_data_loader:
        #     input_seq, output_seq = batch
        ## To see the shape of the input and output sequences
        ## we need to permute the input.
        
        #     # input_seq=input_seq.permute(0, 2, 1, 3, 4)
        #     print("Input shape:", input_seq.shape)
            
        #     # torch.Size([32, 8, 1, 360, 516])
        #     # print("Output shape:", output_seq.shape) # torch.Size([32, 12, 1, 360, 516])
        #     break
                
        
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ####################################################################################################################################
        model = SimpleVAE3D(
            input_channels=config['data']['input_shape'][0],
            latent_dim=config['model']['latent_dim']
        )
        
      
        # # # Train
        trainer = VAETrainer(config, model, train_data_loader,val_data_loader)
        history = trainer.train()
        
        
    # # Option 2: Resume training
    #     trainer = VAETrainer(
    #     config, 
    #     model, 
    #     train_data_loader,
    #     val_data_loader, 
        
    #   # use you own path here dont just uncomment and use!!
    #     resume_checkpoint="/home1/ppatel2025/ppworktp/encoder/encoder_decoder_imerg_trial_2/outputs/vae_3d_experiment_20250414_093807/checkpoints/checkpoint_epoch_149.pth"
    #     )
   #  change yout checkpoint path according to your system here
        # history = trainer.train()        
    #     # Save final model
        final_model_path = os.path.join(trainer.exp_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model to {final_model_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
