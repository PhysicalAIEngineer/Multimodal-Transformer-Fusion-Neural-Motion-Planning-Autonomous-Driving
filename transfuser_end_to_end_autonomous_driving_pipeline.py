# Imports the kagglehub library:  Used to download datasets directly from Kaggle within a notebook environment
import kagglehub
# Downloads the dataset "Helper Scripts"
Helper_Scripts = kagglehub.dataset_download("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Helper Scripts")
# Downloads the CARLA dataset (train + validation)
DataSet = kagglehub.dataset_download("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/DataSet")
# Prints a confirmation message indicating that dataset download is complete
print('Data source import complete.')

"""
1 Imports & Data Loading
"""

!pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
!pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11/index.html
!pip install mmdet==2.25.0 -f https://download.openmmlab.com/mmdet/dist/cpu/torch1.11/index.html

# Provides functions to interact with the operating system file handling, directory paths, environment variables
import os
#  Allows access to system-specific parameters and functions
import sys
#  Used for generating random numbers e.g., sampling, shuffling data
import random
# NumPy for numerical operations, arrays, matrix computations
import numpy as np
# Provides progress bars for loops useful for training or data loading
from tqdm import tqdm
# Used for plotting graphs and visualizing images
import matplotlib.pyplot as plt
# Adds custom directory to python path enables importing modules from ''
sys.path.append("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Helper Scripts")
# PyTorch main library for deep learning tensors, GPU support, training models
import torch
# # Neural network module of PyTorch provides layers (Conv, Linear, etc.) used to build deep learning models
import torch.nn as nn

# # Imports configuration class:
# - Contains hyperparameters and settings (image size, sequence length, sensor configs, etc.)
from config import GlobalConfig
# Imports dataset loader class:
# - Handles loading and preprocessing of CARLA autonomous driving dataset
from data import CARLA_Data
# Creates an instance of the configuration object with default settings
config = GlobalConfig()
# Initializes training dataset:
# - root → path to training data
# - config → configuration settings for preprocessing
# - load_raw_lidar=True → loads raw LiDAR point cloud (XYZI) instead of only processed BEV
train_set = CARLA_Data(root="F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/DataSet/train",
                       config=config, load_raw_lidar=True)
# Initializes validation dataset:
# - Same settings as training set but uses validation split
val_set = CARLA_Data(root="F:\Complete Project Stores\Thesis Work\3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning\DataSet\val",
                     config=config, load_raw_lidar=True)
# Prints number of samples in both datasets
print(f"There are {len(train_set)} samples in training set, {len(val_set)} samples in validation set ")

# Imports DataLoader:
# - Used to load dataset in batches
# - Supports shuffling, parallel loading (multi-workers), etc
from torch.utils.data import DataLoader
# Creates a PyTorch random number generator on CPU
g_cuda = torch.Generator(device='cpu')
# Sets seed of the generator using PyTorch’s initial seed
g_cuda.manual_seed(torch.initial_seed())
# Defines a function to initialize random seeds for each DataLoader worker
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers,need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    # Sets NumPy random seed for this worker
    np.random.seed(worker_seed)
    # Sets Python's built-in random seed for this worker
    random.seed(worker_seed)

# Creates DataLoader for training dataset:
# - train_set → dataset object
# - shuffle=True → randomizes data order each epoch (important for training)
# - batch_size=4 → loads 4 samples per batch
# - worker_init_fn=seed_worker → ensures each worker has unique random seed
# - generator=g_cuda → controls randomness for reproducibility
# - num_workers=4 → uses 4 parallel processes to load data faster
dataloader_train = DataLoader(train_set, shuffle=True, batch_size=4,
                              worker_init_fn=seed_worker, generator=g_cuda,
                              num_workers=4)
# Creates DataLoader for validation dataset:
# - val_set → dataset object
# - shuffle=False → keeps order fixed (important for evaluation consistency)
# - batch_size=4 → loads 4 samples per batch
# - worker_init_fn=seed_worker → ensures proper seeding
# - generator=g_cuda → reproducibility
# - num_workers=4 → parallel data loading
dataloader_val   = DataLoader(val_set, shuffle=False, batch_size=4,
                              worker_init_fn=seed_worker, generator=g_cuda,
                              num_workers=4)

# Creates an iterator from the training DataLoader and retrieves the first batch
sample_data = next(iter(dataloader_train))
# Prints the type of the batch
print(f"sample data is of type {type(sample_data)} and has following keys")
#  Iterates over each key-value pair in the batch dictionary
# k → key (e.g., 'rgb', 'lidar', 'depth', 'semantic', 'waypoints', etc.)
# v → corresponding tensor/data
for k,v in sample_data.items():
    #  Prints: key name, shape of the tensor as a list
    print(k, list(v.shape))

# Visualizes the first RGB image in the batch
plt.imshow(sample_data['rgb'][0].permute(1,2,0))
# Renders the image on the screen
plt.show()
# Ouput Image : 1. RGB Image

# Creates a figure with 1 row and 4 columns of subplot figsize controls overall display size
fig, ax  = plt.subplots(1,4, figsize=(18,6))
# Displays first RGB image in batch
ax[0].imshow(sample_data['rgb'][0].permute(1,2,0))
# Title for RGB image
ax[0].set_title("Original RGB Image")
# Displays BEV (Bird’s Eye View) map:
# - Represents top-down spatial environment (LiDAR/processed map)
ax[1].imshow(sample_data['bev'][0])
# Title for BEV image
ax[1].set_title("BEV")
# Displays depth map:
# - Encodes distance of each pixel from camera
ax[2].imshow(sample_data['depth'][0])
# Title for depth visualization
ax[2].set_title("Depth")
# Displays semantic segmentation map:
# - Each pixel represents a class (road, car, pedestrian, etc.)
ax[3].imshow(sample_data['semantic'][0])
# Title for semantic map
ax[3].set_title("Semantics")
"""
2. Visualize RGB, Bird Eye View, Depth And Semantics  
Images Output
Image presents a multi-modal representation of a driving scene used in an autonomous driving pipeline, showing how different sensor modalities capture complementary information about the environment. On the left, the Original
RGB image provides the raw visual view from the vehicle’s front camera, capturing textures, colors, and objects like road, buildings, and surroundings. Next to it, the BEV (Bird’s Eye View) map offers a top-down spatial
representation of the scene, highlighting drivable lanes and road structure in a simplified geometric form, which is especially useful for planning and navigation. The Depth map encodes distance information, where color
variations represent how far objects are from the vehicle, enabling understanding of 3D structure. Finally, the Semantic segmentation map assigns class labels to each pixel (e.g., road, sky, obstacles), providing high-level
scene understanding. Together, these four views illustrate how autonomous systems combine appearance (RGB), geometry (depth), structure (BEV), and semantics

## 
2. TransFuser Architecture: Image Encoder
Begin with the image branch. Use 'RegNet32' backbone for both extracting image and LiDAR features. We'll make use of the `timm` library, for getting models, pretrained on large datasets, like Imagenet. We need 2 main inputs
- Name of architecture (eg: resnet18, resnet32). In our case, use regnety032
- pretrained or not (boolean)
"""

# Imports timm provides many pretrained CNN architectures (ResNet, EfficientNet, etc.)
import timm
# Defines a custom CNN class for processing RGB images
class ImageCNN(nn.Module):
    def __init__(self, architecture, normalize):
        # Initializes parent nn.Module
        super().__init__()
        # Stores whether input images should be normalized before passing to network
        self.normalize = normalize
        # Loads a pretrained model from timm based on given architecture name pretained = True -> uses ImageNet pretained weight
        self.features = timm.create_model(architecture, pretrained=True)
        # Removes fully connected (classification) layer only need features extractor, not classifier
        self.features.fc = None
        # Maps initial convolution layer from timm model (architecture-specific)
        self.features.conv1 = self.features.stem.conv
        # Batch normalization layer from model stem
        self.features.bn1  = self.features.stem.bn
        #  Removes activation layer with empty module possibly to cutomize forward pass later
        self.features.act1 = nn.Sequential()
        # Removes max pooling layer helps preserve spatial resolution
        self.features.maxpool =  nn.Sequential()
        # Maps internal stages (s1–s4) to ResNet-like naming (layer1–layer4)
        # Makes architecture compatible with standard CNN pipelines
        self.features.layer1 =self.features.s1
        self.features.layer2 =self.features.s2
        self.features.layer3 =self.features.s3
        self.features.layer4 =self.features.s4
        # Adds global average pooling: Converts feature map → single value per channel (C × 1 × 1)
        self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Removes classification head final output will be features embedding instend of class predictions
        self.features.head = nn.Sequential()


#  Creates an instance of the ImageCNN class:
# - architecture='regnety_032' → uses RegNetY-032 model from timm as backbone
# - normalize=True → enables input normalization (handled inside the model)
# This encoder will extract feature maps from RGB images
image_encoder = ImageCNN(architecture='regnety_032', normalize=True)
# Prints the full architecture of the image encoder:
# - Shows all layers (conv, batch norm, blocks, pooling, etc.)
# - Helps verify that classification head is removed
# - Useful for debugging and understanding model structure
print(image_encoder)

# Iterates over all parameters of the model:
# - name → name of the layer/parameter (e.g., conv1.weight, layer1.0.conv1.weight)
# - param → the actual tensor (weights or biases)
for name, param in model.named_parameters():
    # Prints:
    # - parameter name
    # - shape of the parameter tensor
    print(name, param.shape)

# Selects one RGB sample from the batch:
# - sample_data['rgb'] → shape (B, C, H, W)
# - [0:1] → keeps batch dimension (shape becomes (1, C, H, W))
test_rgb_input = sample_data['rgb'][0:1, :, :, :]
# normalize using imagenet values
def normalize_imagenet(x):
    # Creates a copy to avoid modifying the original tensor
    x = x.clone()
    # Normalizes Red channel using ImageNet mean/std
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    # Normalizes Green channel using ImageNet mean/std
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    # Normalizes Blue channel using ImageNet mean/std
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    # Return normalized tensor
    return x
# Applies normalization to the selected RGB input convert tensor to float types
test_rgb_input = normalize_imagenet(test_rgb_input).float()

# Passes input image through first convolution layer:
# - Extracts low-level features (edges, textures)
# - Reduces spatial resolution slightly and increases channel depth
image_features = image_encoder.features.conv1(test_rgb_input)
# Applies batch normalization:
# - Stabilizes feature distribution
# - Helps faster and more stable training
image_features = image_encoder.features.bn1(image_features)
# Applies activation function (ReLU or identity depending on modification)
# - Introduces non-linearity into the network
image_features = image_encoder.features.act1(image_features)
# Applies max pooling (here replaced with identity/no-op)
# - Normally reduces spatial size, but here it preserves resolution
image_features = image_encoder.features.maxpool(image_features)
# Prints shape after initial "stem" block
print(f"Image features shape after 1st stem = {image_features.shape}")
# Passes features through first residual/block layer:
# - Learns more complex patterns than initial conv
# - Usually keeps resolution but increases feature richness
image_features = image_encoder.features.layer1(image_features)
# Prints shape after layer1
print(f"Layer1 Image features shape = {image_features.shape}")

# Randomly selects 8 channel indices from feature map:
# - image_features.shape[1] → number of channels (C)
# - Each channel represents a learned feature (edges, textures, patterns)
indices = np.random.randint(0, image_features.shape[1], size=8)
# Imports OpenCV for image resizing
import cv2
# Creates a 2x4 grid of subplots (total 8 plots)
fig, axes = plt.subplots(2,4, figsize= (18, 6))
# Flattens 2D axes array into 1D list for easier indexing
axes = axes.flatten()
# Loops over selected feature channels
for i in range(len(indices)):
    # Extracts one feature map:
    feature_map = image_features[0, indices[i]].detach().cpu().numpy()
    # Resizes feature map for better visualization
    feature_map = cv2.resize(feature_map, (1200, 320), interpolation = cv2.INTER_LINEAR)
    # Displays feature map using "turbo" colormap
    axes[i].imshow(feature_map, cmap="turbo")
    # Removes axis ticks for cleaner visualization
    axes[i].axis('off')
# Adds title describing visualization
plt.suptitle('RGB image Regnety032 Layer1 features');
# Adjusts layout to prevent overlap between subplots
plt.tight_layout();
"""
3. RegNetY-032 CNN encoder processing an RGB Driving Scene
Ouput Images
Image visualizes intermediate feature maps from the first layer (Layer1) of a RegNetY-032 CNN encoder processing an RGB driving scene. Instead of showing the original image, each small panel represents a single channel (feature
map) learned by the network, highlighting different patterns extracted from the scene. The color variations (from dark blue to bright yellow) indicate the activation strength, where brighter regions correspond to areas the
network considers important. Some feature maps clearly emphasize edges and boundaries such as lane markings, road borders, and object contours (cars, buildings), while others capture texture and spatial patterns like trees
fences, or road surfaces. A few maps focus on specific regions of interest, such as the center of the road or distant objects, showing how different filters specialize in different aspects of the scene. Overall
this visualization demonstrates how early CNN layers transform raw RGB input into multi-channel abstract representations, extracting low-level features (edges, gradients) that will later evolve into higher-level semantic
understanding in deeper layers
"""

""" 
3. TransFuser Architecture: Lidar Encoder
LiDAR is a 3D point cloud, but in our case,  turn that into a Bird Eye View histogram and use the same RegNet32 encoder on it. 
"""
# Retrieves the number of valid LiDAR points for the first sample in the batch
# - Some datasets pad LiDAR arrays, so this tells how many points are actually valid
num_lidar_points = sample_data['num_raw_lidar_points'][0]
# Extracts the actual LiDAR point cloud:
# - sample_data['raw_lidar'][0] → all (possibly padded) points for first sample
# - [:num_lidar_points] → keeps only valid points (removes padding)
# Result shape → (N, 4) where N = number of points, 4 = (x, y, z, intensity)
lidar_data = sample_data['raw_lidar'][0][:num_lidar_points]
# Prints shape of LiDAR data → (N, 4)
print(lidar_data.shape)
# Prints the first LiDAR point:
# - Contains (x, y, z, intensity) values
# - Represents position and reflectance of a single point in space
print(lidar_data[0])

# Imports Plotly for interactive 3D visualization
import plotly.graph_objects as go
# Imports a helper function to create 3D scatter plot data for point clouds
from utils import plot_pc_data3d
# Defines scene configuration for 3D plot
PCD_SCENE=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False,),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.1),
)
# Creates 3D scatter plot data:
# - x, y, z → coordinates of LiDAR points
# - Each point represents a position in 3D space
pc_plots = plot_pc_data3d(x=lidar_data[:,0], y=lidar_data[:,1], z=lidar_data[:,2])
#  Defines layout of the plot:
# - Dark theme for better visibility
# - Title centered (title_x=0.5)
# - Uses custom scene settings
layout = dict(template="plotly_dark", title="Raw Point cloud", scene=PCD_SCENE, title_x=0.5)
# Creates Plotly figure using point cloud data and layout
fig = go.Figure(data=pc_plots, layout=layout)
#  Displays interactive 3D visualization
fig.show()
"""
4. 3D Visualization LiDAR Point Clound Visualization Bird EYE View
Ouput Images
3D LiDAR point cloud visualization projected into a Bird’s Eye View (BEV), capturing the spatial structure of the environment around the ego vehicle. The point cloud appears as a collection of colored points arranged in
concentric arc-like patterns, which are characteristic of LiDAR scanning geometry as the sensor emits beams in circular sweeps. The dense red region near the bottom center represents points closest to the ego vehicle, where the
sensor receives the strongest and most frequent returns. As the distance increases, the points transition into cooler colors (blue/purple), indicating lower density or farther objects. The vertical and linear structures visible
in the upper and side regions likely correspond to environmental boundaries such as walls, fences, buildings, or roadside objects, forming clear geometric outlines in the BEV space. The symmetry and curvature of the arcs
highlight how LiDAR captures depth and distance in discrete layers. Overall, this visualization provides a geometric understanding of the scene, emphasizing spatial relationships, object boundaries, and free space, which are
crucial for tasks like mapping, obstacle detection, localization, and path planning
"""

# Extracts LiDAR BEV histogram features for the first sample in the batch:
lidars_pc = sample_data['lidar'][0]
# # Creates a figure with 2 subplots (side-by-side)
fig, ax  = plt.subplots(1,2, figsize=(10,5))
# Displays first channel:
# - lidars_pc[0] → BEV histogram for above-ground points
# - Represents objects like vehicles, poles, obstacles
ax[0].imshow(lidars_pc[0]);
# Title for first subplot
ax[0].set_title("BEV Histogram feature for Above ground points");
#  Displays second channel:
# - lidars_pc[1] → BEV histogram for below-ground points
# - Represents road surface and terrain structure
ax[1].imshow(lidars_pc[1]);
# Title for second subplot
ax[1].set_title("BEV Histogram feature for Below ground points");
"""
5. BEV Histogram Features For Below Ground Points
Ouput Images
Image presents a two-channel Bird’s Eye View (BEV) histogram representation of LiDAR data, where the scene is decomposed based on height into above-ground and below-ground point distributions. The left panel (above ground
points) highlights sparse but meaningful clusters corresponding to objects in the environment, such as vehicles, poles, or vertical structures. These appear as scattered bright spots and thin vertical patterns, indicating
localized high-density regions where LiDAR returns from elevated objects occur. In contrast, the right panel (Below ground points) shows a much denser and more structured pattern, representing the road surface and ground plane.
The characteristic concentric arc patterns originate from the LiDAR sensor’s scanning mechanism, capturing the geometry of the ground as distance increases from the ego vehicle (located near the bottom center). The intensity
variation reflects point density, with brighter regions indicating stronger or more frequent returns. Together, these two representations separate obstacle information (above ground) from drivable surface geometry (below ground
which crucial for perform object detection, free-space estimation, and safe path planning using structured BEV features
"""

# Extracts the target point BEV map for the first sample in the batch
tgt_pt_in_lidar = sample_data['target_point_image'][0]
# Displays the first (and usually only) channel of the target point map:
plt.imshow(tgt_pt_in_lidar[0]);
"""
6. Target Point Image
Image Output
Image represents a Bird’s Eye View (BEV) target point map, where the goal or desired waypoint for the ego vehicle is encoded spatially in a grid. The background is almost entirely dark (low values), indicating no significant
activation across most of the scene, while a small bright circular region near the top center highlights the target location. This bright spot acts as a goal indicator, showing where the vehicle should navigate in the BEV
coordinate frame. Its position relative to the grid reflects the direction and distance of the target from the ego vehicle (typically located near the bottom center of the map). Such representations are commonly used in
autonomous driving models to provide explicit spatial guidance, allowing the network to learn goal-conditioned planning. In essence, this image encodes a single, focused navigation cue, helping the model understand where to go
within the surrounding environment.
"""
# Defines a CNN-based encoder for LiDAR BEV inputs
class LidarEncoder(nn.Module):
    def __init__(self, architecture, in_channels=3):
        # Initializes parent nn.Module
        super().__init__()
        # Loads a CNN backbone from timm (e.g., RegNet)
        # pretrained=False → no ImageNet weights LiDAR data is different modality
        self._model = timm.create_model(architecture, pretrained=False)
        # Removes final classification layer only need feature extractor
        self._model.fc = None
        # First convolution layer (input layer)
        self._model.conv1 = self._model.stem.conv
        # Batch normalization layer
        self._model.bn1  = self._model.stem.bn
        # Replaces activation with identity
        self._model.act1 = nn.Sequential()
        # Removes max pooling preserves spatial resolution
        self._model.maxpool =  nn.Sequential()
        # Maps internal stages (s1–s4) to standard naming (layer1–layer4) makes architectures consistent with image encoders
        self._model.layer1 = self._model.s1
        self._model.layer2 = self._model.s2
        self._model.layer3 = self._model.s3
        self._model.layer4 = self._model.s4
        # Global average pooling → reduces spatial dimensions to 1x1
        self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Removes classification head
        self._model.head = nn.Sequential()
        # Stores original conv1 layer temporarily
        _tmp = self._model.conv1
        # Checks whether original conv layer had bias
        use_bias = (_tmp.bias != None)
        #  Replaces first conv layer:
        # - Adjusts input channels (e.g., LiDAR BEV may have 2 or 3 channels)
        # - Keeps same output channels and parameters as original
        self._model.conv1 = nn.Conv2d(in_channels,
                                      out_channels=_tmp.out_channels,
                                      kernel_size=_tmp.kernel_size,
                                      stride=_tmp.stride, padding=_tmp.padding,
                                      bias=use_bias)
        # Need to delete the old conv_layer to avoid unused parameters
        del self._model.stem.conv
        # Clears GPU memory
        torch.cuda.empty_cache()
        # Deletes temporary variable
        del _tmp

# Creates an instance of the LidarEncoder:
# - architecture='regnety_032' → uses RegNetY-032 backbone from timm
# - Model is adapted to process LiDAR BEV inputs (instead of RGB images)
# - First convolution layer is modified to match LiDAR input channels
lidar_encoder = LidarEncoder(architecture='regnety_032')
# Prints the full architecture of the LiDAR encoder:
# - Shows all layers (conv, batch norm, blocks, pooling, etc.)
# - Confirms that classification head is removed
print(lidar_encoder)

# Prepares LiDAR input for the network:
# - lidars_pc → BEV LiDAR features (C, H, W)
# - tgt_pt_in_lidar → target point map (C, H, W)
# - torch.cat(..., dim=0) → concatenates along channel dimension
# - combines LiDAR features + goal information
# - unsqueeze(0) → adds batch dimension → (1, C, H, W)
# - .float() → converts to float tensor for model input
test_lidar_input = torch.cat((lidars_pc,
                              tgt_pt_in_lidar), dim=0).unsqueeze(0).float()

# Passes LiDAR input through first convolution layer:
# - Extracts low-level spatial features from BEV input
lidar_features = image_encoder.features.conv1(test_lidar_input)
# Applies batch normalization: stabilizes feature distribution
lidar_features = image_encoder.features.bn1(lidar_features)
# Applies activation function or identity depending on model modification
lidar_features = image_encoder.features.act1(lidar_features)
# Applies max pooling (or identity here) usually reduces spatial size, but here may preserve resolution
lidar_features = image_encoder.features.maxpool(lidar_features)
# Prints shape after initial "stem" processing helps verify dimensions
print(f"Lidar features shape after 1st stem = {lidar_features.shape}")
# Passes features through first CNN block (layer1) learns higher-level patterns (shapes, structures)
lidar_features = image_encoder.features.layer1(lidar_features)
# Prints shape after layer1
print(f"Layer1 Lidar features shape = {lidar_features.shape}")

# Randomly selects 8 channel indices from LiDAR feature map:
# - lidar_features.shape[1] → number of channels (C)
# - Each channel encodes a learned spatial feature from BEV input
indices = np.random.randint(0, lidar_features.shape[1], size=8)
# Creates a 2x4 grid of subplots (8 visualizations)
fig, axes = plt.subplots(2,4, figsize= (10, 5))
# Flattens axes array for easy iteration
axes = axes.flatten()
# Loop over selected feature channels
for i in range(len(indices)):
    # Extracts one feature map:
    feature_map = lidar_features[0, indices[i]].detach().cpu().numpy()
    # Resizes feature map for better visualization
    feature_map = cv2.resize(feature_map, (512, 512),
                             interpolation = cv2.INTER_LINEAR)
    # Displays feature map using "turbo" colormap:
    axes[i].imshow(feature_map, cmap="turbo")
    # Removes axis ticks for cleaner look
    axes[i].axis('off')
# Adds overall title describing visualization
plt.suptitle('Lidar Regnety032 Layer1 features');
# Adjusts subplot spacing to avoid overlap
plt.tight_layout();

"""
7. First Convolutional Layers (Layer1) Of RegNetY-032 Network Processing LiDAR BEV Input
Image Output:
Image visualizes intermediate feature maps from the first convolutional layer (Layer1) of a RegNetY-032 network processing LiDAR BEV input, similar to how CNNs extract features from images but applied to spatial LiDAR data. Each
small panel corresponds to a different channel (feature map), highlighting specific learned patterns from the LiDAR representation. The color intensity (ranging from dark blue to bright yellow/green) indicates the activation
strength, where brighter regions correspond to important structures detected by the network. Many of these feature maps clearly capture geometric patterns of the environment, such as the characteristic arc-shaped LiDAR scan
lines, vertical boundaries like walls or fences, and clustered regions that may correspond to obstacles or objects. Some channels emphasize fine-grained textures and point density, while others highlight strong edges and spatial
transitions in the BEV space. A few maps appear more uniform or sparse, indicating filters that are less activated for this particular scene. Overall, this visualization demonstrates how early CNN layers transform raw LiDAR BEV
inputs into multi-channel spatial features, encoding both structural geometry and object-related information, which are essential for downstream tasks like object detection, scene understanding, and navigation.
"""

"""
4. TransFuser Architecture: Deep Fusion Transformer
At this point, able to extract features from Layer 1, 2, 3, 4 of each encoder. now going to fuse these features together. In Deep Fusion, we usually do that in the Bird Eye View space. It's an elegant way to
fuse the perspective data of front facing camera and Top down view of point cloud. In the Transfuser paper, they use a `GPT (General Purpose Transformer)` module, which correlates different sensor data via `Self-attention`
mechanism. 
"""

# Imports math utilities
import math
# Imports PyTorch functional API
import torch.nn.functional as F

class SelfAttention(nn.Module):
    # A vanilla multi-head masked self-attention layer with a projection at the end.
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        # Initializes parent nn.Module
        super().__init__()
        # Ensures embedding dimension is divisible by number of heads
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        #  Linear layers to project input embeddings into: Key(K), Query(Q), Value(V)
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        # Dropout layers:
        # - attn_drop → applied on attention weights
        # - resid_drop → applied after output projection
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        # Final linear layer to combine outputs from all heads
        self.proj = nn.Linear(n_embd, n_embd)
        # Stores number of attention heads
        self.n_head = n_head
        # Stores attention weightsuseful for visualization/debugging
        self.attn_map = None

    def forward(self, x):
        # B → batch size
        # T → sequence length (tokens)
        # C → embedding dimension
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # self-attention computation
        # Computes attention scores:
        # - q @ k^T → similarity between tokens
        # - scaled by sqrt(head size) for stability
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Converts scores into probabilities (attention weights)
        att = F.softmax(att, dim=-1)
        # Stores attention map for analysis/visualization
        self.attn_map = att.detach().cpu().numpy()
        # Applies dropout to attention weights
        att = self.attn_drop(att)
        # Applies attention to values: wighted sum of values vectors
        y = att @ v
        #  Reassembles all heads: (B, nh, T, hs) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        # Final linear projection + dropout
        y = self.resid_drop(self.proj(y))
        # Returns transformed output
        return y

# Creates a random input tensor:
# - Shape → (B, T, C)
#   B = 1 (batch size)
#   T = 174 (sequence length / number of tokens)
#   C = 72 (embedding dimension)
# Used to test the self-attention module
test_attention_input = torch.randn((1, 174, 72))
# Initializes SelfAttention module:
# - n_embd=72 → embedding dimension
# - n_head=4 → 4 attention heads (each head size = 72/4 = 18)
# - attn_pdrop=0.1 → dropout on attention weights
# - resid_pdrop=0.1 → dropout after projection
self_attention_module = SelfAttention(n_embd=72, n_head=4,
                                      attn_pdrop=0.1, resid_pdrop=0.1)
# Passes input through self-attention:
# - Computes Q, K, V
# - Applies scaled dot-product attention
# - Combines multi-head outputs
# - Returns transformed features
self_attention_output = self_attention_module(test_attention_input)
# Prints output shape
print(f"Self Attention output shape = {self_attention_output.shape}")


# Defines a Transformer block (Attention + Feedforward + Residual connections)
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        # Initializes parent nn.Module
        super().__init__()
        # # Layer Normalization:
        # - ln1 → applied before attention
        # - ln2 → applied before MLP
        # Helps stabilize training and improve convergence
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # # Multi-head self-attention module: captures relationship between token & features
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            # Expands feature dimension (hidden layer)
            nn.Linear(n_embd, block_exp * n_embd),
            # Non-linear activation (ReLU used instead of GELU)
            nn.ReLU(True),
            # Projects back to original embedding dimension
            nn.Linear(block_exp * n_embd, n_embd),
            # Dropout for regularization
            nn.Dropout(resid_pdrop),
        )
    # Forward pass through Transformer block
    def forward(self, x):
        # Apply LayerNorm → Self-Attention → Residual connection Residual (skip connection) helps gradient flow and stability
        x = x + self.attn(self.ln1(x))
        # Apply LayerNorm → MLP → Residual connection Adds non-linear transformation to features
        x = x + self.mlp(self.ln2(x))
        # Returns transformed features
        return x

# Creates a random input tensor for testing the Transformer block:
# - Shape → (B, T, C)
#   B = 1 (batch size)
#   T = 174 (sequence length / tokens)
#   C = 72 (embedding dimension)
test_block_input = torch.randn((1, 174, 72))
# Initializes Transformer Block:
# - n_embd=72 → embedding dimension
# - n_head=4 → 4 attention heads
# - block_exp=4 → MLP expands to 4× embedding size (72 → 288 → 72)
# - attn_pdrop=0.1 → dropout in attention
# - resid_pdrop=0.1 → dropout after MLP/output
block = Block(n_embd=72, n_head=4, block_exp=4,
              attn_pdrop=0.1, resid_pdrop=0.1)
# Passes input through Transformer block:
# - LayerNorm → Self-Attention → Residual connection
# - LayerNorm → MLP → Residual connection
# - Learns both global dependencies and non-linear transformations
block_output = block(test_block_input)
# Prints output shape : Same as input shape (Transformer preserves dimensions)
print(f"Transformer block output shape = {block_output.shape}")

class GPT(nn.Module):
    # Defines a Transformer-based fusion module Used to fuse image and LiDAR features
    def __init__(self, n_embd, n_head, block_exp, n_layer,
                    img_vert_anchors, img_horz_anchors,
                    lidar_vert_anchors, lidar_horz_anchors,
                    embd_pdrop, attn_pdrop, resid_pdrop):
        # Initializes parent nn.Module
        super().__init__()
        # Embedding dimension (feature size)
        self.n_embd = n_embd
        #  Number of spatial tokens (anchors) for image and LiDAR features
        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        # Sequence length (only current frame used)
        self.seq_len = 1
        # positional embedding parameter (learnable), image + lidar
        # Learnable positional embeddings:
        # - 1. Encodes spatial position of each token
        # - 2. Covers both image tokens + LiDAR tokens
        self.pos_emb = nn.Parameter(torch.zeros(1,
                                                self.seq_len * img_vert_anchors * img_horz_anchors + \
                                                self.seq_len * lidar_vert_anchors * lidar_horz_anchors,
                                                n_embd))
        # Dropout applied to embeddings
        self.drop = nn.Dropout(embd_pdrop)
        # Stack of Transformer blocks:
        # - Performs multi-head attention + MLP repeatedly
        # - Enables cross-modal fusion (image ↔ LiDAR)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop,
                                            resid_pdrop) for _ in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        # Stores sequence length
        self.block_size = self.seq_len
    # Forward pass: fuse image + LiDAR features
    def forward(self, image_tensor, lidar_tensor):
        # Batch size
        bz = lidar_tensor.shape[0]
        #  Extract spatial dimensions
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]
        # Ensures only single timestep is used
        assert self.seq_len == 1
        # Reshape image features into tokens convert image features map -> sequence of tokens: (B, C, H, W) → (B, H*W, C)
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)
        # Reshape LiDAR features into tokens
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)
        # Concatenate tokens = Combines image + LiDAR tokens into one sequence
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)
        #  Add positional encoding + dropout
        x = self.drop(self.pos_emb + token_embeddings)
        # Transformer processing applies multiple transformer blocks: learn relationship across all token (image -> LiDAR)
        x = self.blocks(x)
        # Final normalization
        x = self.ln_f(x)
        # Reshape back to combined sequence
        x = x.view(bz, self.seq_len*self.img_vert_anchors*self.img_horz_anchors + self.seq_len*self.lidar_vert_anchors*self.lidar_horz_anchors, self.n_embd)
        # Split back into image and LiDAR features extracts image tokens and reshape back to feature map
        image_tensor_out = x[:, :self.seq_len*self.img_vert_anchors*self.img_horz_anchors, :].contiguous().view(bz * self.seq_len, -1, img_h, img_w)
        # Split back into image and LiDAR features extracts LiDAR tokens and reshape back to BEV feature map
        lidar_tensor_out = x[:, self.seq_len*self.img_vert_anchors*self.img_horz_anchors:, :].contiguous().view(bz * self.seq_len, -1, lidar_h, lidar_w)
        # Returns fused image and LiDAR features
        return image_tensor_out, lidar_tensor_out

# Defines adaptive average pooling layer for image features:
# - Output size → (5, 22)
# - Converts any input feature map into fixed spatial resolution
# - Useful for creating consistent token sizes for Transformer
avgpool_img = nn.AdaptiveAvgPool2d((5, 22))
# Defines adaptive average pooling for LiDAR features:
# - Output size → (8, 8)
# - Standardizes BEV feature resolution
avgpool_lidar = nn.AdaptiveAvgPool2d((8, 8))
# Applies pooling to image features:
# - image_features → (B, C, H, W)
# - Output → (B, C, 5, 22)
# - Reduces spatial size while preserving important information
image_layer1_feature = avgpool_img(image_features)
# Applies pooling to LiDAR features:
# - lidar_features → (B, C, H, W)
# - Output → (B, C, 8, 8)
lidar_layer1_feature = avgpool_lidar(lidar_features)

# Initializes GPT-based Transformer fusion module:
# - n_embd=72 → embedding dimension (feature size)
# - n_head=4 → 4 attention heads
# - block_exp=4 → MLP expansion factor (hidden size = 4×)
# - n_layer=4 → number of Transformer blocks stacked
# - img_vert_anchors=5, img_horz_anchors=22 → image tokens (5×22 grid)
# - lidar_vert_anchors=8, lidar_horz_anchors=8 → LiDAR tokens (8×8 grid)
# - dropout params → regularization
transformer_module = GPT(n_embd=72, n_head=4, block_exp=4, n_layer=4,
                         img_vert_anchors =5, img_horz_anchors=22,
                         lidar_vert_anchors=8, lidar_horz_anchors=8,
                         embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1)
# Passes pooled image + LiDAR features into Transformer:
# - Converts both into tokens
# - Applies multi-head self-attention across combined sequence
# - Enables cross-modal fusion (image ↔ LiDAR)
# - Outputs fused feature maps for both modalities
fused_image_features, fused_lidar_features = transformer_module(image_layer1_feature, lidar_layer1_feature)
# Prints shape of fused image features
print(f"Transformer output Fused Image features = {fused_image_features.shape}")
# Prints shape of fused LiDAR features
print(f"Transformer output Fused Lidar features = {fused_lidar_features.shape}")

"""
NOTE:
- Lidar layer1 output shape = (1, 72, 64, 64)
- Fused Lidar layer1 transformer output shape = (1, 72, 8, 8)
- Image layer1 output shape = (1, 72, 40, 176)
- Fused Image layer1 transformer output shape = (1, 72, 5, 22)
Apart from fusing lidar and camera images using Transformer blocks, can add `Residual` connections b/w the original layer1 output and the fused output for each modality. This is similar to **Multi-scale fusion using FPN
Idea is to upsample the fused output using bilinear interpolation and add it to original output
"""

# Upsamples fused image features back to original spatial resolution:
# - fused_image_features → (B, C, 5, 22)
# - target size → same as original image_features (H, W)
# - bilinear interpolation → smooth resizing
image_features_layer1 = F.interpolate(fused_image_features,
                                      size=(image_features.shape[2],
                                            image_features.shape[3]),
                                       mode='bilinear',
                                      align_corners=False)
# Upsamples fused LiDAR features:
# - fused_lidar_features → (B, C, 8, 8)
# - resized to match original LiDAR feature map size
lidar_features_layer1 = F.interpolate(fused_lidar_features,
                                      size=(lidar_features.shape[2],
                                            lidar_features.shape[3]),
                                      mode='bilinear',
                                      align_corners=False)
# Residual fusion for image branch:
# - Adds Transformer-fused features back to original CNN features
# - Combines local (CNN) + global (Transformer) information
image_features = image_features + image_features_layer1
# Residual fusion for LiDAR branch:
# - Enhances LiDAR features with cross-modal context
lidar_features = lidar_features + lidar_features_layer1
# Prints final image feature shape
print(f"Image features shape after layer1 fusion = {image_features.shape}")
# Prints final LiDAR feature shape
print(f"Lidar features shape after layer1 fusion = {lidar_features.shape}")

"""
The above feature extraction and fusion b/w lidar and image data is repeated 4 times (4 layers) in the same manner. The output shapes at each layer are as follows:
| Layer | Lidar feature shape | Image feature shape|
| --- | --- | --- |
| Layer1 | (1, 72, 64, 64) | (1, 72, 40, 176)|
| Layer2 | (1, 216, 32, 32) | (1, 216, 20, 88)|
| Layer3 | (1, 576, 16, 16) | (1, 576, 10, 44)|
| Layer4 | (1, 1512, 8, 8) | (1, 1512, 5, 22)|

Small final piece of fusion module, that represents all that the network has learnt, from the sensor inputs. We need to combine the lidar and image feature maps from the last layer, to be passed on to the planning module. 
use the following steps:
- Use 1x1 convolutions to reduce the channel dimension
- Reduce the spatial dimension using AdaptiveAveragePooling
"""

# Creates dummy outputs from final CNN layer (layer4):
# - Image → (B=1, C=1512, H=5, W=22)
# - LiDAR → (B=1, C=1512, H=8, W=8)
# These simulate deep feature maps before final projection
test_layer4_image_output = torch.randn(1, 1512, 5, 22)
test_layer4_lidar_output = torch.randn(1, 1512, 8, 8)
# Defines 1x1 convolution layers:
# - Reduces channel dimension from 1512 → 512
# - Keeps spatial dimensions unchanged
# - Acts as feature compression/projection layer
final_image_conv = nn.Conv2d(1512, 512, kernel_size=1, padding=0)
final_lidar_conv = nn.Conv2d(1512, 512, kernel_size=1, padding=0)
# Applies channel reduction: Output → (B, 512, H, W)
image_features = final_image_conv(test_layer4_image_output)
lidar_features = final_lidar_conv(test_layer4_lidar_output)
#  Applies global average pooling on image features: Converts (B, C, H, W) → (B, C, 1, 1)
image_features = image_encoder.features.global_pool(image_features)
# Flattens to vector: (B, C, 1, 1) → (B, C)
image_features = torch.flatten(image_features, 1)
# Applies global average pooling to LiDAR features
lidar_features = lidar_encoder._model.global_pool(lidar_features)
# Flattens LiDAR features to (B, C)
lidar_features = torch.flatten(lidar_features, 1)
# Fuses both modalities:
# - Element-wise addition
# - Combines semantic (image) + geometric (LiDAR) information
fused_features = image_features + lidar_features
# Prints shapes of individual feature vectors → (B, 512)
print(f"Lidar features shape = {lidar_features.shape}, Image features shape = {image_features.shape}")
# Prints shape of fused feature vector → (B, 512)
print(f"Fused features shape = {fused_features.shape}")

# Visualizes the fused feature vector as a 2D image:
# - fused_features → shape (1, 512)
plt.imshow(fused_features.detach().cpu().numpy().reshape(16,32))
# Displays the visualization
plt.show()

"""
8. Visualization Of  Fused Feature Vector Obtained After Combining Multi-Modal Sensor Information (Typically RGB + LiDAR)
Image Output
Image represents a visualization of a fused feature vector obtained after combining multi-modal sensor information (typically RGB + LiDAR), reshaped into a 2D grid for interpretability. Although it appears like a small heatmap,
it actually corresponds to a high-dimensional feature embedding (e.g., 512-dimensional vector reshaped into 16×32) produced by the fusion network. Each cell in the grid represents the activation value of a learned feature, where
color variations (from dark blue to bright yellow) indicate the strength of that feature’s response. The scattered pattern of activations shows that the model has encoded diverse and distributed information, capturing both
semantic cues from images (like objects and textures) and geometric cues from LiDAR (like depth and structure). There is no explicit spatial meaning in this layout; instead, it reflects the latent representation learned by the
network, where each dimension contributes to decision-making. Such fused feature vectors serve as a compact, information-rich summary of the environment, which is then used by downstream modules (e.g., MLP + GRU) for tasks like
trajectory prediction, planning, or control
"""

"""
5. TransFuser Architecture: Deep Planning
Goal of planning module is to predict the future waypoints, to be followed by the vehicle.
Input
- BEV Fused Features, extracted from Perception module (512,)
- Waypoints, predicted from previous timestamp (for first timestamp, we input zero values) (2,)
- Target Waypoint (2,)
Output
- Future Waypoints in vehicle coordinate frame
Use a simple MLP + GRU network, to predict the future points, one by one.
"""
# MLP compresses fused features into a smaller latent vector used as input to planning & trajectory decoders
proj1 = nn.Sequential(
                    # Reduces feature size from 512 → 256
                    nn.Linear(512, 256),
                    #  Adds non-linearity (ReLU activation)
                    nn.ReLU(inplace=True),
                    # Further reduces dimension 256 → 128
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    # Final reduction 128 → 64 (compact representation)
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                )
# Defines GRUCell (recurrent unit):
# - input_size=4 → input consists of:
#     (x, y) previous waypoint + (x, y) target point
# - hidden_size=64 → hidden state size matches MLP output
# Used for sequential trajectory prediction (step-by-step)
decoder = nn.GRUCell(input_size=4, hidden_size=64)
# Final layer:
# - Maps hidden state (64) → 3 values
# - Typically represents:
#     (x, y, confidence) OR (x, y, speed) OR (dx, dy, other signal)
# Used to predict next waypoint or control signal
output = nn.Linear(64, 3)

# Creates a random fused feature vector:
# - Shape → (B, 512)
# - B = 1 (batch size)
# - Represents output from perception + fusion module
test_planning_input = torch.randn(1,512)
# Creates a random target/goal point:
# - Shape → (B, 2)
# - Represents (x, y) coordinates of destination in ego frame
target_point = torch.randn(1,2)
# Extracts batch size from input
bs = test_planning_input.shape[0]
# Defines number of future steps to predict GRU decoders wil generate 4 future waypoits sequentially
num_steps = 4

# Passes fused features through MLP:
# - Input → (B, 512)
# - Output → (B, 64)
# This becomes the initial hidden state for GRU
z = proj1(test_planning_input)
# Initializes list to store predicted waypoints
output_wp = list()
# initial input variable to GRU
# Initial previous waypoint:
# - Shape → (B, 2)
# - Starts at (0,0) in ego frame
# - Acts as starting position for trajectory
x = torch.zeros(size=(bs, 2), dtype=z.dtype)
# autoregressive generation of output waypoints loop over future timesteps
for _ in range(num_steps):
    # Concatenates: previous waypoints(x) , target point(goal)
    x_in = torch.cat([x, target_point], dim=1)
    # Updates GRU hidden state :
    # - x_in → input (current context)
    # - z → previous hidden state
    z = decoder(x_in, z)
    # Projects hidden state to output:
    # - Shape → (B, 3)
    # - Typically contains delta movement + extra info
    dx = output(z)
    # Updates waypoint:
    # - dx[:, :2] → predicted offset (Δx, Δy)
    # - Adds to previous position (x)
    # → accumulates trajectory step-by-step
    x = dx[:,:2] + x
    # Stores predicted waypoint (x, y)
    output_wp.append(x[:,:2])
# Stacks all predicted waypoints:
# - Shape → (B, num_steps, 2)
# - Represents full predicted trajectory
pred_wp = torch.stack(output_wp, dim=1)
# Prints shape of predicted waypoints
print(pred_wp.shape)
# Prints actual predicted waypoint values
print(pred_wp)

# Creates dummy ground-truth trajectory: represents actual future path to ego vehicle
test_ego_waypoint = torch.randn(1, num_steps, 2)
# Compute L1 loss (Mean Absolute Error): measures how far predictions are from true trajectory
loss_wp = torch.mean(torch.abs(pred_wp - test_ego_waypoint))
# prints scalar loss value
print(f"L1 loss b/w predicted and actual trajectory = {loss_wp.item() : 0.4f}")

"""
Auxillary tasks
The above modules form the crux of the `Transfuser` model, which is executed during inference. Experiments have shown model convergence and performance to improve, with addition of auxilary tasks, during training. In this model, authors have used
- 3D Object detection
- Depth estimation
- Semantic Segmentation
- HD Map prediction (in BEV space)
as auxillary tasks. These many tasks were possible, due to training data being generated from CARLA simulator. All these might not be possible in real life datasets, due to practical difficulties, but it definitely helps to have such tasks
NOTE:
- The last layer image and lidar feature maps are used as input for each of these tasks (akin to different heads in Hydranet)
- The losses from each of auxilary tasks are combined with main loss (L1 on waypoint prediction) using weighted average. The weights are hyperparameters for the model
"""

# Selects computation device:
# - Uses GPU (cuda:0) if available
# - Otherwise falls back to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Imports the main model:
# - LidarCenterNet → multi-modal network for autonomous driving
# - Combines LiDAR + image features for perception + planning
from model import LidarCenterNet
# # Initializes the model:
# - config → hyperparameters/settings
# - device → CPU/GPU
# - backbone → defines architecture details
# - image_architecture → CNN backbone for RGB branch
# - lidar_architecture → CNN backbone for LiDAR branch
model = LidarCenterNet(config, device, config.backbone,
                       image_architecture='regnety_032',
                       lidar_architecture='regnety_032')
# Moves model to selected device (GPU/CPU)
model.to(device);
# Filters only trainable parameters:  p.requires_grad=True → parameters that will be updated during training
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# Computes total number of trainable parameters
params = sum([np.prod(p.size()) for p in model_parameters])
# Prints total number of trainable parameters
print ('Total trainable parameters: ', params)

"""
Training utilities
"""
# Defines training + validation loop function
# - model → neural network
# - num_epochs → number of training iterations over dataset
# - optimizer → updates model weights
# - dataloaders → provide training/validation batches
def train_validate_model(model, num_epochs, model_name,
                         optimizer,device, dataloader_train,
                         dataloader_valid,lr_scheduler = None,
                         output_path = '.'):
    # Stores training
    train_results = []
    # Stores validation metrices
    val_results = []
    # min_val_loss used for model checkpointing
    min_val_loss = np.Inf
    # move model to device
    model.to(device)
    # Loop over epochs
    for epoch in range(num_epochs):
        # Training
        model.train()
        # Sets model to training mode (enables dropout, batchnorm updates)
        epoch_detailed_train_losses  = {key: 0.0 for key in config.detailed_losses}
        # Initializes dictionary to track different loss components
        epoch_detailed_train_losses['weighted_loss'] = 0.0
        # Wraps dataloader with progress bar
        with tqdm(dataloader_train, unit="batch") as tepoch:
            #  Updates progress bar with current epoch
            for batch_idx, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                # Load data to gpu, according to type moves interger type input (class labels) to device
                for k in ['rgb', 'depth', 'lidar', 'label', 'ego_waypoint',
                          'target_point', 'target_point_image', 'speed']:
                    # Moves float-type inputs to device
                    data[k] = data[k].to(device, torch.float32)
                # Moves integer-type inputs (class labels) to device
                for k in ['semantic', 'bev']:
                    data[k] = data[k].to(device, torch.long)
                # Forward pass, store losses returns dictionary of different loss components
                losses, _ = model(data)
                # Intialize total loss
                loss = torch.tensor(0.0).to(device, dtype=torch.float32)
                # Computes weighted sum of individual losses
                for key, value in losses.items():
                    # Computes weighted sum of individual losses
                    loss += detailed_weights[key] * value
                    # Accumulates per-loss contribution
                    epoch_detailed_train_losses[key] += float(detailed_weights[key] * value.item())
                # Accumulates total weighted loss
                epoch_detailed_train_losses['weighted_loss'] += float(loss.item())
                # Backward pass previous gradients
                optimizer.zero_grad(set_to_none=True)
                # Computes gradients via backpropagation
                loss.backward()
                # Updates model parameters
                optimizer.step()
                # log losses display current loss in progress bar
                tepoch.set_postfix(loss=loss.item())
                # Early break for debugging/demo → only 3 batches per epoch
                if batch_idx == 2:
                    break
            # Average losses across batches
            for k,v in epoch_detailed_train_losses.items():
                # Computes average loss per epoch
                epoch_detailed_train_losses[k] = v / len(dataloader_train)
        # Stores epoch training results
        train_results.append(epoch_detailed_train_losses)
    # Returns training and validation metrics
    return train_results, val_results

"""
Training
"""
# Imports PyTorch optimization module
import torch.optim as optim
# Defines optimizer:
# - AdamW → variant of Adam with weight decay better generalization
# - model.parameters() → all trainable parameters of model
# - lr=1e-4 → learning rate step size for updates
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
# Sets number of training epochs
N_EPOCHS = 3
#  Creates dictionary mapping each loss component → its weight used to compute weighted total loss during training
detailed_weights = {key: config.detailed_losses_weights[idx] for idx, key in enumerate(config.detailed_losses)}
# Calls training function:
# - Trains model for N_EPOCHS
# - Returns training and validation loss history
train_results, val_results = train_validate_model(model, num_epochs=N_EPOCHS,
                                                  model_name='Transfuser_regnet032',
                                                  optimizer=optimizer,
                                                  device = device,
                                                  dataloader_train=dataloader_train,
                                                  dataloader_valid = dataloader_val)
# Imports pandas for data handling
import pandas as pd
# Converts training results (list of dicts) → pandas DataFrame
train_results = pd.DataFrame(train_results)
# Saves training results to file:
# - Filename includes model name + number of epochs
train_results.to_csv(f'Transfuser_regnet032y_{N_EPOCHS}.pth')

#  What it contains:
# - Each row → one epoch
# - Each column → a specific loss component (e.g., waypoint loss, detection loss, etc.)
# - Also includes 'weighted_loss' (total combined loss)
train_results

# Imports Plotly for interactive plotting
import plotly.graph_objects as go
# Initializes an empty Plotly figure
fig = go.Figure()
# Iterates over all loss components:
    # - config.detailed_losses → individual loss terms (e.g., waypoint, detection, etc.)
    # - 'weighted_loss' → total combined loss
for key in config.detailed_losses + ['weighted_loss']:
    fig.add_trace(go.Scatter(
        # X-axis → epoch indices (0, 1, 2, ...)
        x=np.arange(len(train_results)),
        # Y-axis → corresponding loss values from DataFrame
        y=train_results[key],
        # Plots line graph
        mode='lines',
        # Sets trace name: removes "loss_" prefix for display
        name=f'{key.replace("loss_", "")}_TR'))
# Configures plot layout:
# - Title → describes experiment/model
# - Size → large for better visualization
# - Axis labels → Epoch vs Loss
fig.update_layout(title='Transfuser with regnet032y backbone',
                  width=1200, height=600,
                  xaxis_title='Epoch',
                  yaxis_title='Loss components')
# Displays interactive plot:
fig.show()
"""