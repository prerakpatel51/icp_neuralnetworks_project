🌟 IMERG & IR Encoders
3D Convolutional Autoencoders for Satellite Data

This repository contains PyTorch implementations of encoders/decoders for IMERG (precipitation) and IR (infrared) satellite data.
There are two folders for each encoders.
The one with the trial are the most recent ones with changes and others are saved just for reference. example trial2 is the latest one use latest one only 
The structure of each folder is as follows:
```
|___configs
|___src
| |___data
| | |_____pycache__
| | |___Dataset (keep it inside this folder)
| |___training
| | |_____pycache__
| |___models
| | |_____pycache__
| |_____pycache__
| |___utils
| | |_____pycache__
|___outputs
| |___vae_3d_experiment_20250406_102423
| | |___checkpoints
| |___post_training_analysis
| | |___visualizations
| |___vae_3d_experiment_20250406_101630
| | |___checkpoints
|___train.py
|___visualization.py
|___live_loss.py
|___result_visualizer.ipynb

```
Here’s a polished bullet-point version of your features section with improved readability and visual appeal:

---

## ✨ Key Features  
- **⏱️ Time Logging**  
  - Track duration of each training phase (data loading, forward/backward pass, validation).  
  - Logs formatted as `HH:MM:SS` for easy profiling.  

- **⚡ Gradient Clipping**  
  - Prevents exploding gradients with `torch.nn.utils.clip_grad_norm_`.  
  - Configurable threshold in `train_config.yaml`.  

- **🚀 Mixed Precision Training**  
  - Uses `torch.cuda.amp` for faster forward passes.  
  - Reduces GPU memory usage without sacrificing accuracy.  

- **💾 Model Checkpointing**  
  - Auto-saves models every *N* epochs (configurable).  
  - Includes optimizer state for seamless resumption.  

- **♻️ Resumable Training**  
  - Recovery from crashes 
  - Preserves epoch count and learning rate scheduling.  

- **📊 Visualization Tools**  
  - Side-by-side original/reconstructed slices (GIF/PNG).  
  
---

### Example Config Snippet  
```yaml
training:
  gradient_clip: 1.0          # Clip threshold
  use_amp: True               # Enable mixed precision
  checkpoint_every: 2         # Save every 2 epochs
```

---

**
🚀 Quick Start
 Copy paste this if you dont want to know anything
after connecting to ssh do

```
srun --partition=gpu1  --mem=250GB --cpus-per-task=24 --time=04:00:00 --pty bash
git clone https://github.com/prerakpatel51/ppworktp.git
cd ppworktp/encoder/encoder_decoder_imerg_trial_2
conda activate
pip install -r requirements.txt
# To start training
python train.py —config configs/train_config.yaml #(you can specify if you want to have multiple configs)
#  To start visualization
python visualization.py —model “path/to/model”
```

Make sure you are connected to AI Panther or a GPU cluster via ssh.
you can use this kind of configuration for the gpu in ai panther  `srun --partition=gpu1  --mem=250GB --cpus-per-task=24 --time=04:00:00 --pty bash`
It will automatically give you access to partition 1 with 250GB memory and 24 cpus per task thats the max capacity for 4 hours which can be changed according to the use.
Anyhow i got problem in using all the 4 gpus for dataparallel in model training. It supports only two gpu training at a time so i used device_ids=[0,1] for the training specifically and its hard coded.


Install dependencies:

```pip install -r requirements.txt```\
Start training:

Learning rate can be configured in `train_config.yaml` 

```python train.py --config configs/train_config.yaml```\
\
Checkpoints saved every 2 epochs (configurable in train_config.yaml).

To resume training, use the Option 2 block in `train.py`.


While training the loss can be seen live using 
```python live_loss.py /path/to/loss.csv ``` 
# The loss file is seen in output and inside experiments. (It will be created when you run training) 

Visualize results:
```python visualization.py --model /path/to/checkpoint.pt```


You will be provided a  access to notebook for visualization i will try to tune models using different parameters and upload the results in drive you can see it using collab.

Outputs:
Outputs will be available on the outputs folder with each time you run the training it will make a new directory with timestamp in its name. It consists of the check points models and best models and final model. It will automatically figure out the best model among others and make it available for you.

The output of the visualizations will be seen inside `output/post-training-visualization`
It will try to show the resonstruction, and samples of input and the animation of sample input (GIF)
It also consists of the timely logs.

Let me define the architecture for the model of IR model here which is mostly similar to imerg.


```
(B, C, D, H, W) = (1, 1, 16, 360, 516)
```
The data must be in the src/data/dataset folder inorder to work.
---

### 🔷 **Encoder Shape Progression**

| Step | Layer | Description | Output Shape |
|------|-------|-------------|--------------|
| Input | - | Original input | **(1, 1, 16, 360, 516)** |
| 1 | Conv3d (stride=(1,1,2)) | Channels ↑ to 64, W ↓ | **(1, 64, 16, 360, 258)** |
| 2 | GroupNorm | - | (1, 64, 16, 360, 258) |
| 3 | Conv3d | 3x3 conv | (1, 64, 16, 360, 258) |
| 4 | GroupNorm | - | (1, 64, 16, 360, 258) |
| 5 | Conv3d | 3x3 conv | (1, 64, 16, 360, 258) |
| 6 | SELU | - | (1, 64, 16, 360, 258) |
| 7 | Conv3d (1,3,3) | Only H and W touched | (1, 64, 16, 360, 256) |
| 8–10 | GroupNorm, Conv3d ×2 | Repeats | (1, 64, 16, 360, 256) |
| 11 | SELU | - | (1, 64, 16, 360, 256) |
| 12 | Conv3d (stride=2) | Downsample all dims | **(1, 64, 8, 180, 128)** |
| 13 | GroupNorm | - | (1, 64, 8, 180, 128) |
| 14 | Conv3d | Refinement | (1, 64, 8, 180, 128) |
| 16 | Conv3d (1,3,1) stride=(1,2,1) | Downsample H | **(1, 64, 8, 90, 128)** |
| 16 | AdaptiveAvgPool3d → (4,80,128) | Force shape | **(1, 64, 4, 80, 128)** |
| 17 | GroupNorm | - | (1, 64, 4, 80, 128) |
| 18 | AdaptiveAvgPool3d → (2,64,64) | Final latent spatial | **(1, 64, 2, 64, 64)** |
| 19 | GroupNorm | - | (1, 64, 2, 64, 64) |
| 20 | Conv3d → 32 channels | Reduce dim | **(1, 32, 2, 64, 64)** |
| 21 | SELU | - | (1, 32, 2, 64, 64) |
| 22–23 | μ, logσ² convs | No shape change | **(1, 32, 2, 64, 64)** |

🧠 **Latent representation shape:** `(1, 32, 2, 64, 64)`

The loss function has a additional parameter Beta which can be tweeked to increase or decrease the importance of kl_loss.
---

### 🔶 **Decoder Shape Progression**

| Step | Layer | Description | Output Shape |
|------|-------|-------------|--------------|
| Input | - | Latent z | **(1, 32, 2, 64, 64)** |
| 1–3 | Conv3d → 64, GN, SELU | Channels ↑ | **(1, 64, 2, 64, 64)** |
| 4 | Upsample → (2,64,64) | Same shape | (1, 64, 2, 64, 64) |
| 5–6 | Conv3d x2 | Refinement | (1, 64, 2, 64, 64) |
| 7 | Upsample → (4,80,128) | Increase spatial size | **(1, 64, 4, 80, 128)** |
| 8 | ConvTranspose3d (1,3,1) stride=(1,2,1) | Double H | **(1, 64, 4, 160, 128)** |
| 9–13 | Conv3d, GN, SELU, etc | Smoothing | (1, 64, 4, 160, 128) |
| 14 | ConvTranspose3d (kernel=3, stride=2) | Upsample all dims | **(1, 64, 8, 320, 256)** |
| 16–17 | Conv3d, GN, SELU | Refinement | (1, 64, 8, 320, 256) |
| 18–20 | Conv3d x3 | Final tuning | (1, 64, 8, 320, 256) |
| 21 | Upsample → (16, 360, 516) | Final upsample | **(1, 64, 16, 360, 516)** |
| 22–23 | Conv3d, GN | Final conv layers | (1, 64, 16, 360, 516) |
| 24–25 | SELU, Conv3d (→ 1 channel) | Output image | **(1, 1, 16, 360, 516)** |

---

✅ **Final output shape matches input: (1, 1, 16, 360, 516)**



No need to worry about trial_1,trial2,trial3 they are just raw code or the base code.
