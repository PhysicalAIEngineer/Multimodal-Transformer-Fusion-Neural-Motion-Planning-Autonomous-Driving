# Imports kagglehub library used to download datasets directly from kaggle inside notebook
import kagglehub
#  Downloads Helper Script scripts dataset
Helper_Scripts = kagglehub.dataset_download("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Helper Scripts")
# Downloads demo dataset
DataSet = kagglehub.dataset_download("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/DataSet")
# Downloads pretrained model weights
carla_transfuser_regnet032_path = kagglehub.dataset_download('carla-transfuser-regnet032')
# Prints confirmation message after all downloads finish
print('Data source import complete.')

!pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
!pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.11/index.html
!pip install mmdet==2.25.0 -f https://download.openmmlab.com/mmdet/dist/cpu/torch1.11/index.html
!pip install kaleido
!pip install timm==0.5.4 natsort

import os
import cv2
import sys
import random
import scipy as sp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=5)
sys.path.append("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Helper Scripts")

# imports torch
import torch
# Imports configuration class:
# - Contains model/data settings (image size, sequence length, etc.)
from config import GlobalConfig
# Imports dataset loader for CARLA:
# - Handles loading RGB, LiDAR, labels, and preprocessing
from data import CARLA_Data
# Path to demo dataset specific CARLA town + scenario
root_dir = "F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/DataSet/train/routes_30mshortroutes_Town01_Scenario7junction_route0_11_23_20_22_17"
# Creates configuration object with default parameters
config = GlobalConfig()
# Sets prediction horizon model will predict 6 future waypoints
config.pred_len = 6
# Initializes demo dataset:
# - root → dataset path
# - config → preprocessing + settings
# - routeKey='route13' → selects specific driving route
# - load_raw_lidar=True → loads raw LiDAR point clouds
demo_set = CARLA_Data(root=root_dir, config=config, routeKey='route13', load_raw_lidar=True)
# Prints number of samples in demo dataset
print(f"There are {len(demo_set)} samples in Demo dataset")

"""
Put the images into a PyTorch DataLoader.
"""
# Imports DataLoader:
# - Used to load dataset in batches efficiently
from torch.utils.data import DataLoader
# Creates DataLoader for demo dataset:
# - demo_set → dataset object (CARLA demo data)
# - shuffle=False → keeps data order fixed (important for inference/demo)
# - batch_size=2 → loads 2 samples per batch
# - num_workers=4 → uses 4 parallel workers for faster data loading
dataloader_demo = DataLoader(demo_set, shuffle=False, batch_size=2, num_workers=4)

# Creates an iterator over the demo DataLoader and retrieves the first batch
# - iter(dataloader_demo) → creates iterator
# - next(...) → fetches first batch of data
sample_data = next(iter(dataloader_demo))
# Prints type of batch (typically dict) indicates that multiple data modalities are included
print(f"sample data is of type {type(sample_data)} and has following keys")
# Iterates over each key-value pair in the batch
for k,v in sample_data.items():
    # Prints:
    # - key name (e.g., 'rgb', 'lidar', 'depth', 'semantic', etc.)
    # - shape of corresponding tensor
    print(k, list(v.shape))

# Displays the first RGB image from the demo batch:
# - sample_data['rgb'] → shape (B, C, H, W)
# - [0] → selects first sample in batch
# - permute(1, 2, 0) → converts (C, H, W) → (H, W, C)
plt.imshow(sample_data['rgb'][0].permute(1, 2, 0))
# Renders the image on screen
plt.show()

"""
2. Load Trained TransFuser.
"""

# Selects computation device:
# - Uses GPU (cuda:0) if available
# - Otherwise falls back to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Import the tansfuser based model combines RGB + LiDAR for perception and planning
from model import LidarCenterNet
# Initializes model for inference:
# - estimate_loss=False → disables loss computation only forward predictions
model = LidarCenterNet(config, device, config.backbone,
                       image_architecture='regnety_032',
                       lidar_architecture='regnety_032',
                       estimate_loss = False)
# Moves model to selected device (GPU/CPU)
model.to(device);
# Enables debug mode may print extra logs or intermediate outputs
model.config.debug = True
# Sets model to evaluation mode:
# - Disables dropout
# - Uses running statistics in batch normalization
model.eval();
# # Loads pretrained model weights from file
checkpt = torch.load("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Pretrained Model/transfuser_regnet032_seed1_39.pth", map_location=device)
# Loads weights into model restores trained parameters for inference
model.load_state_dict(checkpt)

# Prints the complete architecture of the model
# - All layers and modules inside LidarCenterNet (image encoder, LiDAR encoder, Transformer fusion, detection heads, planning modules, etc.)
# - Structure of backbone (e.g., RegNetY-032)
# - Details of each submodule (Conv layers, Linear layers, GRU, etc.)
print(model)

""" 
3. Activation Map Visualization
3.1 GradCAM
- CAM (Class activation maps) are localisation maps, that highlight regions of input, that were important in predicting the output.
- GradCAM can be used to visualize the important features, based up to specific layer. It uses combination of gradients and activations on each layer to calculate the Activation maps
- Originally designed for CNN based models, numerous variants have come up, to support different models, including Transformers.
- [EigenCAM](https://arxiv.org/abs/2008.00299) is one of gradient-free methods that is used with models, involving non-differentiable steps (eg: Non Maximal suppression in Object Detection)
- e2e_cam.py is based on this [wonderful repo](https://github.com/jacobgil/pytorch-grad-cam/tree/master), that offers pytorch implementation for many CAM methods
- Use EigenCAM to visualize what different parts of network look for, before predicting final output
"""

# EigenCAM calculates the activation map using activations when passing model through input data
# Imports EigenCAM:
# - A visualization technique (similar to Grad-CAM but gradient-free)
# - Uses principal components (eigenvectors) of feature activations to generate class-agnostic activation maps
from e2e_cam import EigenCAM
# Iterates through all parameters of the model:
# - name → string name of the parameter (layer + weight/bias)
# - param → tensor containing parameter values
for name, param in model.named_parameters():
    # Prints:
    # - parameter name (e.g., "image_encoder.conv1.weight")
    # - shape of the tensor (e.g., [64, 3, 7, 7])
    print(name, param.shape)

# Defines list of target layers for visualization (EigenCAM):
# - First 4 → image encoder layers (RGB branch)
# - Next 4 → LiDAR encoder layers (BEV branch)
# These layers will be used to extract activation maps
# Why multiple layers?
# - Early layers → low-level features (edges, textures)
# - Deeper layers → high-level features (objects, semantics)
target_layers = [model._model.image_encoder.features.layer1,
                 model._model.image_encoder.features.layer2,
                 model._model.image_encoder.features.layer3,
                 model._model.image_encoder.features.layer4,
                 model._model.lidar_encoder._model.layer1,
                 model._model.lidar_encoder._model.layer2,
                 model._model.lidar_encoder._model.layer3,
                 model._model.lidar_encoder._model.layer4,
                ]
# Defines reference input type for each layer:
# - First 4 layers → correspond to RGB input
# - Last 4 layers → correspond to LiDAR input
shape_ref_keys = ['rgb', 'rgb', 'rgb', 'rgb', 'lidar', 'lidar', 'lidar', 'lidar']

# Initializes EigenCAM context:
    # - model → pretrained TransFuser model
    # - target_layers → layers from which activations will be extracted
    # Enables automatic hook registration for feature extraction
with EigenCAM(model=model, target_layers=target_layers) as cam:
    # Iterates over demo dataset with progress bar
    for data in tqdm(dataloader_demo):
        # load data to device, according to type
        for k in ['rgb', 'depth', 'lidar', 'label', 'ego_waypoint', \
                  'target_point', 'target_point_image', 'speed']:
            data[k] = data[k].to(device, torch.float32)
        # Moves integer-type inputs (labels/maps) to device
        for k in ['semantic', 'bev']:
            data[k] = data[k].to(device, torch.long)
        # get model predictions + activation maps
        # Runs forward pass through model and computes activation maps:
        # - grayscale_cam → list of CAM heatmaps for each target layer
        # - targets=[] → no specific class target (class-agnostic attention)
        # - key=shape_ref_keys → ensures proper resizing (RGB vs LiDAR)
        grayscale_cam = cam(input_data=data, targets=[], key=shape_ref_keys)
        # the actual model outputs can be got here
        # Retrieves model predictions:
        # - cam.outputs stores forward pass outputs
        # - [1] likely corresponds to main prediction (e.g., waypoints)
        outputs = cam.outputs[1]
        # stop after first batch
        break

# Prints basic information about CAM output:
# - type(grayscale_cam) → usually list
# - len(grayscale_cam) → number of activation maps
print(f"grayscale cam is of type {type(grayscale_cam)} and length {len(grayscale_cam)}")
# Prints shape of each activation map:
# - Each x → CAM heatmap for a specific layer
# - Shape typically → (B, H, W)
#   where:
#     B = batch size
#     H, W = spatial dimensions (after resizing)
print(f"grayscale_cam shapes = {[x.shape for x in grayscale_cam]}")

# Selects index of sample to visualize
i = 0
# # Extracts RGB image from batch:
# - data['rgb'] → shape (B, C, H, W)
# - [i] → selects i-th sample
# - permute(1, 2, 0) → converts (C, H, W) → (H, W, C)
rgb_image = data['rgb'][i].permute(1, 2, 0).detach().cpu().numpy()
# Normalizes pixel values: Converts range [0, 255] → [0, 1]
rgb_image = rgb_image / 255.0
# Displays RGB image
plt.imshow(rgb_image)
# Renders the image
plt.show()
# Output Images : 1. RGB Image

# Extracts CAM heatmaps for RGB branch:
# - grayscale_cam[0:4] → first 4 layers (image encoder)
# - x[i, 0] → selects i-th sample and first channel
# Result → list of 4 heatmaps (one per layer)
rgb_cam_images = [x[i,0] for x in grayscale_cam[0:4]]
# Creates 1 row × 4 columns of subplots
fig, ax  = plt.subplots(1,4, figsize=(16,3))
for idx in range(4):
    # Displays CAM heatmap: 'jet' colormap highlights intensity (blue → low, red → high)
    ax[idx].imshow(rgb_cam_images[idx], cmap = 'jet');
    # Titles each subplot according to layer number
    ax[idx].set_title(f'Layer{idx+1}')
    # Removes axis ticks for clean visualization
    ax[idx].axis('off')
# Adds overall title for all plots
plt.suptitle('Image Encoder Class Activation Maps')

"""
2. Class Activation Maps(CAMs) from different Layers
Images Output
Image presents Class Activation Maps (CAMs) from different layers (Layer1 to Layer4) of the image encoder, illustrating how the model’s attention evolves across depth while processing an RGB driving scene. In Layer1, the 
activation map is dense and highly textured, capturing low-level features such as edges, gradients, and fine details across the entire scene, including road boundaries, buildings, and background clutter. Moving to Layer2, the 
activations become more sparse and focused, highlighting salient regions like objects or important structures, while suppressing irrelevant background noise. In Layer3, the model begins to emphasize more structured and semantically 
meaningful patterns, such as lane regions, object groupings, and spatial layouts, indicating a transition from local feature extraction to contextual understanding. Finally, in Layer4, the activation map is highly abstract and 
concentrated, with strong responses in specific regions that are most critical for decision-making, such as drivable space, obstacles, or navigation-relevant areas. The color intensity (blue to red) reflects the importance of 
different regions, with red indicating high attention. Overall, this visualization demonstrates the hierarchical learning behavior of CNNs, where early layers capture low-level visual cues, and deeper layers encode high-level 
semantic understanding
"""

# Overlays a CAM (heatmap) on top of the original image
def show_cam_on_image(img, mask, colormap = cv2.COLORMAP_JET, image_weight = 0.5):
    # Converts normalized mask (0–1) → (0–255) and applies colormap produces colored heatmap (e.g: blue->red intensity)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # Converts OpenCV BGR format -> RGB format
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Normalizes heatmap values to [0, 1]
    heatmap = np.float32(heatmap) / 255
    # heatmap with original image:
    # - image_weight controls transparency
    #   higher → more original image visible
    #   lower → stronger heatmap
    cam = (1 - image_weight) * heatmap + image_weight * img
    # Normalizes result to keep values within valid range
    cam = cam / np.max(cam)
    # Converts final image back to uint8 format (0–255) for display
    return np.uint8(255 * cam)

# Creates a figure with 4 subplots (one for each CNN layer)
fig, ax  = plt.subplots(1,4, figsize=(16,3))
# Loops over 4 layers (Layer1 to Layer4)
for idx in range(4):
    # Overlays CAM heatmap on original RGB image:
    # - rgb_image → original image
    # - rgb_cam_images[idx] → activation map for that layer
    # - show_cam_on_image() → blends heatmap + image
    ax[idx].imshow(show_cam_on_image(rgb_image, rgb_cam_images[idx]), cmap = 'jet');
    # Sets title for each subplot (Layer1, Layer2, etc.)
    ax[idx].set_title(f'Layer{idx+1}')
    # Removes axis ticks for cleaner visualization
    ax[idx].axis('off')
# Adds overall title for the entire figure
plt.suptitle('Image Encoder Class Activation Maps')

"""
3. Class Activation Maps (CAMs) across four layers (Layer1–Layer4) of an Image Encoder
Output Images
Image shows Class Activation Maps (CAMs) across four layers (Layer1–Layer4) of an image encoder, overlaid on a driving scene, illustrating how the model’s attention becomes progressively more focused and semantically meaningful. 
In Layer1, the activations are widespread and noisy, capturing low-level visual features such as edges, textures, and color gradients across buildings, road surfaces, and surrounding structures. In Layer2, the attention begins to 
concentrate on salient regions, particularly along the road and nearby objects, while background details start to fade, indicating early feature selection. By Layer3, the model emphasizes more structured and context-aware regions, 
such as lane boundaries, road geometry, and object groupings, reflecting a deeper understanding of the scene layout. In Layer4, the activation becomes highly concentrated and abstract, with strong responses in specific areas that 
are most critical for driving decisions—such as drivable space, obstacles, and navigation-relevant zones—while irrelevant regions are largely suppressed. The color intensity (blue to red) represents the importance of different 
regions, where red indicates higher attention. Overall, the image demonstrates the hierarchical feature learning of the encoder: transitioning from low-level perception to high-level semantic reasoning, which is essential for 
autonomous driving systems to interpret and act on complex environments.
"""

""" 
3.2. LiDAR Activation
Visualize the LiDAR point cloud
"""
# Extracts LiDAR BEV data for visualization
lidar_data = data['lidar'][i].detach().cpu().numpy().transpose(1,2,0)
# Displays LiDAR BEV image each channel encodes spatial features (above/below ground)
plt.imshow(lidar_data)

"""
4. LiDAR-based Bird’s Eye View (BEV)
Ouput Images
Image shows a LiDAR-based Bird’s Eye View (BEV) or polar-style point cloud visualization, where the environment around the sensor is represented as concentric arcs and radial patterns. The dark background highlights sparse but 
structured point returns, forming semi-circular rings that correspond to LiDAR scan lines at increasing distances from the sensor. Near the bottom center, a dense bright region indicates the LiDAR origin or immediate surroundings,
where point density is highest due to proximity. As distance increases outward, the points become more spread out and less dense, reflecting the natural behavior of LiDAR sampling. The color variation—from green to red—likely 
encodes intensity, height, or distance, with brighter or warmer colors indicating stronger returns or closer objects. On the left side, irregular clustered points suggest vertical structures or obstacles, such as walls or roadside 
objects, while the right side appears relatively sparse, indicating open space. A distinct blue circular marker near the mid-right region highlights a specific point or region of interest, possibly representing a detected object, 
target location, or annotation. Overall, the image captures the geometric structure of the scene as perceived by LiDAR, emphasizing radial symmetry, obstacle distribution, and spatial awareness crucial for autonomous navigation.
"""

# Extracts CAM heatmaps for LiDAR branch:
# - grayscale_cam[4:] → last 4 layers (LiDAR encoder)
# - x[i, 0] → selects i-th sample and first channel
# Result → list of 4 LiDAR CAM maps
lidar_cam_images = [x[i,0] for x in grayscale_cam[4:]]
# Creates 1×4 subplot grid for visualization
fig, ax  = plt.subplots(1,4, figsize=(16,4))
# Loops over 4 layers (Layer1 to Layer4)
for idx in range(4):
    # Overlays CAM heatmap on LiDAR BEV image:
    # - lidar_data → BEV representation
    # - lidar_cam_images[idx] → activation map for that layer
    # - show_cam_on_image → blends heatmap with LiDAR map
    ax[idx].imshow(show_cam_on_image(lidar_data, lidar_cam_images[idx]), cmap = 'jet');
    # Titles each subplot
    ax[idx].set_title(f'Layer{idx+1}')
    # Removes axes for cleaner visualization
    ax[idx].axis('off')
# Adds overall title
plt.suptitle('Lidar Encoder Class Activation Maps')
# Displays the visualization
plt.show()

"""
5. LiDAR Encoder Class Activation Maps (CAMs) across four layers (Layer1–Layer4)
Output Images
Image presents LiDAR Encoder Class Activation Maps (CAMs) across four layers (Layer1–Layer4), visualized on a Bird’s Eye View (BEV) representation of the environment, illustrating how the model’s spatial attention evolves through 
the network. In Layer1, the activation is dense and spread across many regions, capturing low-level geometric patterns such as scan lines, point density variations, and basic obstacle outlines; the attention appears noisy and 
broadly distributed, reflecting early-stage feature extraction. In Layer2, the model begins to focus on more structured regions, with clearer emphasis on nearby obstacles and regions of interest, while irrelevant background areas 
start to fade. By Layer3, the attention becomes more semantically organized, highlighting vertical structures and obstacle boundaries with stronger, more localized responses, indicating that the model is learning meaningful spatial 
relationships in the scene. In Layer4, the activation is highly abstract and concentrated, with only a few critical regions strongly highlighted, such as key obstacles or navigationally important zones, while most of the scene is 
suppressed. The color gradient (blue → green → red) represents increasing importance, with red regions indicating where the model is focusing most strongly. The circular LiDAR scan pattern remains faintly visible underneath, 
grounding the activations in the original sensor geometry. Overall, this visualization demonstrates how the LiDAR encoder transitions from raw geometric perception to high-level spatial understanding, enabling the model to identify 
obstacles and important areas for autonomous navigation.
"""

"""
4. Waypoint Prediction
"""
# Prints the model outputs obtained during inference
# - Predictions from the model forward pass
# - Typically includes:
#     → predicted waypoints (trajectory)
#     → control signals (steering, throttle, brake) or other task-specific outputs depending on model design
print(outputs)

# Prints all keys in the output dictionary
print(outputs.keys())

# # Extracts predicted waypoints for i-th sample:
# - outputs['pred_wp'] → shape (B, num_steps, 2)
# - [i] → selects i-th sample from batch
pred_waypoints = outputs['pred_wp'][i]
# Prints shape of predicted waypoints
print(pred_waypoints.shape)
#  Prints actual waypoint values:
# - Each row represents a future position of ego vehicle
# - Coordinates are usually in local/ego reference frame
print(pred_waypoints)

# Flips Y-axis:
# - Converts from LiDAR coordinate system → vehicle coordinate system
# - LiDAR and vehicle frames have opposite lateral direction
pred_waypoints[:, 1] *= -1
# Shifts X-axis forward:
# - Adds offset (1.3 meters) to align LiDAR origin with vehicle center
# - Accounts for sensor placement relative to ego vehicle
pred_waypoints[:, 0] += 1.3
# Creates figure for plotting trajectory
plt.figure(figsize=(6,3))
# Plots predicted waypoints:
plt.plot(pred_waypoints[:,0], pred_waypoints[:,1],
         'r*', label='Predicted waypoints', alpha=0.4)
# Displays legend
plt.legend();
# X-axis represents forward motion
plt.xlabel('Vehicle Forward direction')
# Y-axis represents sideways motion
plt.ylabel('Vehicle Lateral direction')
# Plot title
plt.title('Predicted waypoints')
# Adds grid for better readability
plt.grid(True)
# Ensures equal scaling on both axes presents true geometry of trajectory
plt.axis('equal');
"""
6. Trajectory visualization of predicted waypoints for an autonomous vehicle
Output Images
Trajectory visualization of predicted waypoints for an autonomous vehicle, plotted in the vehicle’s coordinate frame. The horizontal axis represents the vehicle’s forward direction (X-axis), while the vertical axis represents the 
lateral direction (Y-axis). Two sets of points are displayed: blue markers indicating the ego (ground-truth or reference) waypoints and red markers representing the model’s predicted future waypoints. The ego waypoints are 
clustered closer to the origin, suggesting the current or near-future positions of the vehicle, while the predicted waypoints extend further along the forward direction, forming a trajectory that curves gradually toward the 
negative lateral side. This indicates that the model is planning a path that moves forward while slightly turning or shifting sideways, possibly to follow a lane or avoid an obstacle. The spacing between predicted points increases 
along the trajectory, reflecting progression over time steps. The grid and equal axis scaling help visualize the geometric consistency of the path. Overall, the plot demonstrates how the model forecasts a smooth and continuous 
driving trajectory, aligning reasonably with the initial ego waypoints while projecting future motion into the environment.
"""

""""
5. Auxiliary Tasks.
5.1. Bounding Boxes (LiDAR Branch)
"""
# Extracts raw LiDAR point cloud for i-th sample
lidar_pc = data['raw_lidar'][i].detach().cpu().numpy()
# Retrieves number of valid LiDAR points
num_points = data['num_raw_lidar_points'].detach().cpu().numpy()[i]
# Keeps only valid points removes padded / unused points
lidar_pc = lidar_pc[:num_points, :]
# Imports Plotly input/output module controls rendering behaviour of ploty figures
import plotly.io as pio
# # Sets default renderer to SVG (Scalable Vector Graphics) produces high-quanlity, resolution independent plots
pio.renderers.default = 'svg'
# Imports Plotly for creating interactive 3D visualizations
import plotly.graph_objects as go
# Imports helper function to generate 3D scatter plot data from point cloud
from utils import plot_pc_data3d
# Defines 3D scene configuration:
# - Hides all axes for clean visualization
# - aspectratio compresses Z-axis (height) since LiDAR height variation is small
PCD_SCENE=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False,),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.1),
)
# Creates 3D scatter plot data:
# - x, y, z → coordinates of LiDAR points
# - Each point represents a location in 3D space
pc_plots = plot_pc_data3d(x=lidar_pc[:,0], y=lidar_pc[:,1], z=lidar_pc[:,2])
# Defines layout:
# - Dark theme for better contrast
# - Title centered
# - Uses custom scene settings
layout = dict(template="plotly_dark",
              title="Raw Point cloud",
              scene=PCD_SCENE, title_x=0.5)
# Creates Plotly figure with point cloud data
fig = go.Figure(data=pc_plots, layout=layout)
# Hides legend for cleaner visualization
fig.update_layout(showlegend=False)
# Hides X-axis in 2D projection
fig.update_xaxes(visible=False)
# Hides Y-axis in 2D projection
fig.update_yaxes(visible=False)
# Displays 3D point cloud
fig.show()

"""
7. 3D visualization of a raw LiDAR point cloud
Output Images
Images shows a 3D visualization of a raw LiDAR point cloud, where the environment is represented as a collection of discrete points in space against a dark background. The points are sparsely distributed and arranged in curved, 
arc-like patterns, which reflect the rotational scanning mechanism of the LiDAR sensor. These arcs indicate different scan layers capturing the surroundings at varying angles and distances. The color gradient—ranging from yellow 
and orange to pink and purple—likely encodes depth, height, or intensity, helping distinguish spatial structure within the scene. The densest concentration of points appears near the center, representing nearby objects or ground 
surfaces, while more scattered points extend outward, indicating farther or less reflective surfaces. Some clusters form recognizable shapes that may correspond to obstacles, roadside structures, or terrain features, while large 
empty regions suggest open space with no returns. The perspective is slightly angled, giving a semi-top-down 3D view that emphasizes both horizontal spread and vertical variation. Overall, this visualization captures the raw 
geometric perception of the environment as sensed by LiDAR, providing essential spatial information used for tasks like obstacle detection, mapping, and autonomous navigation.
"""

# Extracts predicted BEV bounding boxes for i-th sample:
# - outputs['detections'] → shape (B, N, 7)
# - [i] → selects i-th sample
pred_bev_boxes = outputs['detections'][i]
# Prints all detected objects and their parameters
print(pred_bev_boxes)

# Extracts predicted waypoints for i-th sample:
# - outputs['pred_wp'] → shape (B, num_steps, 2)
# - [i] → selects i-th sample from batch
# Result → (num_steps, 2) → sequence of (x, y) coordinates
bev_to_lidar = np.array([[0, -(1/8.0), 32],[-(1/8.0), 0, 16],[0 , 0, 1]])
# Converts BEV 2D bounding boxes → 3D bounding boxes in LiDAR frame
def convert_to_3d_bboxes(boxes_2d):
    # Number of detected objects
    n_boxes = boxes_2d.shape[0]
    # Initializes output array: each box → (x, y, z, length, width, height, yaw)
    bbox_3d = np.zeros((n_boxes, 7))
    # xy position from bev pixels to metres
    # Converts (x, y) → homogeneous coordinates (x, y, 1) required for matrix multiplication with transformation matrix
    homogenous_coordinates = np.hstack([boxes_2d[:, :2], np.ones((n_boxes, 1))])
    # Applies transformation converts BEV pixel positionn -> LiDAR coordinates (meter)
    bbox_3d[:, :2] = (bev_to_lidar @ homogenous_coordinates.T).T[:, :2]
    # Converts height (h) from pixels → meters (scaled by 1/8)
    bbox_3d[:, 3] = boxes_2d[:, 3] / 8
    # Converts width (w) from pixels → meters
    bbox_3d[:, 4] = boxes_2d[:, 2] / 8
    # Converts yaw (orientation) negates angle to match LiDAR coorinates convention
    bbox_3d[:, 6] = -boxes_2d[:, 4]
    # Sets fixed height position (z-axis) assume objects are near ground level
    bbox_3d[:, 2] = -1.25
    # Sets fixed object height e.g: vehicle height = 2.5m
    bbox_3d[:, 5] = 2.5
    # Returns 3D bounding boxes (x, y, z, length, width, height, yaw)
    return bbox_3d

# Converts predicted BEV bounding boxes → 3D bounding boxes:
# - pred_bev_boxes → (N, 7) in BEV space (pixels)
# - convert_to_3d_bboxes → transforms into LiDAR coordinate system (meters)
pred_3d_boxes = convert_to_3d_bboxes(pred_bev_boxes)
# Prints all converted 3D bounding boxes:
# - Each row represents one detected object in 3D space
print(pred_3d_boxes)

# Imports utility function:
# - Converts 3D bounding boxes → 8 corner points representation
from utils import boxes_to_corners_3d
# Converts each 3D box into its 8 corner coordinates:
# - pred_3d_boxes → shape (N, 7)
#   (x, y, z, length, width, height, yaw)
# - Output → shape (N, 8, 3)
#   N → number of boxes
#   8 → number of corners per box
#   3 → (x, y, z) coordinates for each corner
pred_corners_3d = boxes_to_corners_3d(pred_3d_boxes)
# Prints shape of output: confirms each box is represented by 8 3D cornner points
print(pred_corners_3d.shape)


# Sets Plotly's default renderer to SVG (Scalable Vector Graphics)
pio.renderers.default = 'svg'

# Imports helper function:
# - Creates Plotly 3D visualization for LiDAR + bounding boxes
from utils import get_lidar3d_plots
# 3D bounding box corners used to draw predicted objects boxes in 3D
lidar_3d_plots = get_lidar3d_plots(lidar_pc,
                                   pc_kwargs=dict(colorscale='viridis', marker_size=0.9),
                                   pred_box_corners = pred_corners_3d,
                                   pred_box_colors = ['white'] * len(pred_corners_3d))
# Updates plot title to describe visualization
layout['title'] = 'Transfuser Predicted Bounding boxes'
# Creates Plotly figure: combines LiDAR point cloud + predicted 3D bounding boxes
fig = go.Figure(data=lidar_3d_plots, layout=layout)
# Displays 3D visualization:
# - Shows environment geometry (point cloud)
# - Overlays predicted object boxes
# - Evaluating detection performance
fig.show()

"""
8. 3D LiDAR point cloud visualization enhanced with predicted bounding boxes from the TransFuser model
Output Images
- Image presents a 3D LiDAR point cloud visualization enhanced with predicted bounding boxes from the TransFuser model, illustrating how the system detects objects in a spatial environment. The scene is displayed against a dark 
background, where the LiDAR data forms curved, concentric arc patterns that reflect the sensor’s scanning mechanism. The points are color-coded—ranging from green and yellow to purple—indicating variations in depth, intensity, or 
height. Near the center and extending outward, clusters of points represent physical structures or obstacles in the environment. Overlaid on this point cloud are several white 3D bounding boxes, which mark the locations and 
extents of detected objects, likely vehicles or obstacles relevant to driving. These boxes are aligned with the spatial geometry of the point cloud, showing both position and orientation in 3D space. The visualization perspective 
is slightly angled, giving a semi-top-down view that captures both horizontal layout and depth. The title “Transfuser Predicted Bounding boxes” indicates that these detections are generated by the model’s perception module. 
Overall, the image demonstrates how raw LiDAR data is transformed into meaningful object-level understanding, enabling the autonomous system to identify and localize obstacles for safe navigation.
"""

"""
5.2. BEV "HD MAP" Prediction
"""
# Displays predicted BEV (Bird’s Eye View) map from model output:
# - outputs['pred_bev'] → predicted BEV tensor (B, C, H, W)
plt.imshow(outputs['pred_bev'][0][1])

# Extracts predicted BEV map for i-th sample:
# - outputs['pred_bev'] → shape (B, C, H, W)
# - [i] → selects i-th sample
pred_bev = outputs['pred_bev'][i]
# Prints shape of BEV prediction
print(pred_bev.shape)
# Converts multi-channel prediction → single-channel class map:
# - Takes index of highest probability across channels
# - Output → (H, W) with class labels:
#     0 → unknown
#     1 → drivable area
#     2 → lane markings
pred_bev = np.argmax(pred_bev, axis=0)
# Displays BEV segmentation map:
# - 'terrain' colormap helps distinguish classes visually
plt.imshow(pred_bev, cmap = 'terrain');
# Removes axes for clean visualization
plt.axis('off');
# Adds title
plt.title('Predicted BEV image');
"""
9. Predicted Bird’s Eye View (BEV) Segmentation Map
Output Images
Image shows a predicted Bird’s Eye View (BEV) segmentation map generated by the model, representing the scene from a top-down perspective. The visualization uses distinct colors to encode different semantic classes, where the 
light yellow region dominates the center, indicating the drivable area or road surface. Running through this region are multiple thin, elongated white structures, which likely correspond to lane markings or road boundaries, 
arranged in parallel and extending vertically, suggesting a straight roadway with multiple lanes. On both the left and right sides, large dark blue regions indicate non-drivable areas, such as sidewalks, buildings, or off-road 
zones. The segmentation appears clean and structured, with sharp transitions between classes, reflecting the model’s ability to distinguish between navigable space and surrounding environment. The vertical orientation of features 
aligns with the forward direction of the vehicle, and the consistent spacing of lane-like structures suggests organized road geometry. Overall, this BEV image captures the model’s understanding of the road layout, highlighting 
drivable space and lane structure
"""

""""
 5.3. Depth Prediction
"""
# Extracts predicted depth map for i-th sample:
# Result → depth image representing distance from camera
pred_depth_image = outputs['pred_depth'][i]
# Prints shape of depth map
print(f"Predicted Depth image shape = {pred_depth_image.shape}")
# Displays depth image highlights depth variations clearly
plt.imshow(pred_depth_image, cmap = 'inferno');
# Removes axis ticks for cleaner visualization
plt.axis('off');
# Adds title
plt.title('Predicted Depth image');

"""
5.4. Semantic Segmentation
"""
# Displays predicted semantic segmentation map
plt.imshow(outputs['pred_semantic'][i][2])
"""
10. Predicted Semantic Segmentation Map Driving Scene
Output Images
Image represents a predicted depth map of a driving scene, where pixel intensities encode the estimated distance of objects from the camera. The visualization uses a smooth color gradient ranging from darker blue/green tones to 
brighter yellow regions, indicating varying depth values across the scene. The lower portion of the image contains brighter yellow-green areas, which likely correspond to closer objects or road surfaces, while the upper regions 
appear darker and more muted, indicating areas that are farther away, such as distant structures or the horizon. Subtle horizontal and diagonal patterns can be observed, reflecting the geometry of the road and surrounding 
environment, with faint lines suggesting lane structure or surface variations. The depth transitions are continuous and smooth, showing that the model produces a dense depth estimation rather than discrete layers, capturing 
gradual changes in distance. The elongated horizontal format of the image aligns with a wide camera field of view, typical in autonomous driving scenarios. Overall, this depth map provides a spatial understanding of the scene, 
enabling the system to perceive relative distances and 3D structure, which are critical for obstacle avoidance, path planning, and safe navigation.
"""
# Extracts predicted semantic segmentation
pred_semantic_image = outputs['pred_semantic'][i]
# Converts multi-channel prediction → single-channel class map
pred_semantic_image = np.argmax(pred_semantic_image, axis = 0)
# Prints shape of predicted semantic map
print(f"Predicted Semantic image shape = {pred_depth_image.shape}")
# Displays semantic segmentation
plt.imshow(pred_semantic_image, cmap = 'jet');
# Removes axis ticks for cleaner visualization
plt.axis('off');
# Adds title
plt.title('Predicted Semantic image');
"""
11. Predicted Bird’s Eye View (BEV) Segmentation map of a Road Scene
Output Images
Image shows a predicted semantic segmentation map of a driving scene, where each pixel is assigned to a specific class, providing a structured understanding of the environment. The scene is dominated by a dark blue region in the 
upper portion, which likely represents the sky or background, clearly separated from the ground-level elements. In the lower half, a large light blue region forms the central roadway, indicating the drivable area extending forward 
in perspective. Flanking this road, different colored regions appear: on one side, a reddish-brown area likely represents non-drivable surfaces such as sidewalks, terrain, or roadside structures, while on the other side, darker 
blue patches may correspond to objects like vehicles, buildings, or shadows. Thin yellow or lighter lines running along the road suggest lane markings or boundaries, highlighting the model’s ability to capture fine structural 
details. The segmentation boundaries are relatively smooth and consistent, showing that the model can distinguish between different semantic classes such as road, surroundings, and obstacles. Overall, the image reflects the 
model’s capability to transform raw visual input into a meaningful, class-wise representation of the scene
"""

# Imports constants:
# - VEHICLE_TO_LIDAR_FWD → forward offset between vehicle and LiDAR sensor
# - LIDAR_HEIGHT → height of LiDAR sensor above ground
from utils import VEHICLE_TO_LIDAR_FWD, LIDAR_HEIGHT
# Generates left and right lane boundary points from centerline waypoints
# Input:
# - waypoints → (N, 2) → (x, y) coordinates in vehicle frame
# Output:
# - lane_points → (2N, 3) → 3D points (x, y, z) in LiDAR frame
def generate_lane_points(waypoints, lane_width = 1.0):
    # Number of waypoints
    n_points = waypoints.shape[0]
    # Initializes array for left + right lane points:
    # - first N → left lane
    # - next N → right lane
    lane_points = np.zeros((n_points * 2 , 3))
    # Converts x-coordinate from vehicle frame → LiDAR frame adds forward offset
    lane_points[:n_points, 0] = waypoints[:,0] + VEHICLE_TO_LIDAR_FWD
    lane_points[n_points:, 0] = waypoints[:,0] + VEHICLE_TO_LIDAR_FWD
    # Left lane boundary → shift left by half lane width
    lane_points[:n_points,1] = waypoints[:,1] - (lane_width * 0.5)
    # Right lane boundary → shift right by half lane width
    lane_points[n_points:,1] = waypoints[:,1] + (lane_width * 0.5)
    # Places all points on ground plane relative to LiDAR
    lane_points[:,2] = -LIDAR_HEIGHT
    # Returns 3D lane boundary points
    return lane_points

# Converts a BEV bounding box into its 4 rotated corner points
def get_rotated_bbox(bbox):
    # Unpacks bounding box parameters:
    # - x, y → center position in BEV
    # - w, h → width and height
    # - yaw → orientation angle (rotation)
    x, y, w, h, yaw, _, _  =  bbox
    # Defines 4 corner points relative to center:
    # - Uses (h, w) combinations to form rectangle
    # - Third column (1) → for homogeneous coordinates
    bbox = np.array([[h,   w, 1],
                     [h,  -w, 1],
                     [-h, -w, 1],
                     [-h,  w, 1],
                ])
    #  Scales width and height by 0.5: corrects dataset-specific scaling issue
    bbox[:, :2] /= 2
    # Swaps x and y axes: adjusts coordinate convention (BEV vs LiDAR)
    bbox[:, :2] = bbox[:, [1, 0]]
    # Computes cosine and sine of rotation angle
    c, s = np.cos(yaw), np.sin(yaw)
    # Builds 2D transformation matrix rotation (yaw), Translation(x, y), Homogeneous coodinates for matrix multiplication
    r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
    # applies rotation + translation to all cornners
    bbox = r1_to_world @ bbox.T
    #  Transposes back to shape (4, 3)
    bbox = bbox.T
    # Clips coordinates within BEV image bounds ensures cornners stay inside valid range
    bbox = np.clip(bbox, 0, 256)
    # Returns rotated bounding box corner represents polygon of detected object in BEV
    return bbox

# Creates a Plotly 2D scatter plot object
def get_scatter_plot(x,y, mode='lines', marker_size=2, color=None, **kwargs):
    return go.Scatter(x=x, y=y, mode=mode, hoverinfo='skip',showlegend=False,
                        marker = dict(size=marker_size, color=color), **kwargs)
#  Plots 2D bounding box using its corner points
def plot_box_corners2d(box2d, color,**kwargs):
    return [
        # edge 1: corner 0 → corner 1
        get_scatter_plot([box2d[0,0], box2d[1,0]],[box2d[0,1], box2d[1,1]], color=color, **kwargs),
        # edge 2: corner 1 → corner 2
        get_scatter_plot([box2d[1,0], box2d[2,0]], [box2d[1,1], box2d[2,1]], color=color, **kwargs),
        # edge 3: corner 2 → corner 3
        get_scatter_plot([box2d[2,0], box2d[3,0]], [box2d[2,1], box2d[3,1]], color=color, **kwargs),
        # edge 4:  corner 3 → corner 0 (closing the box)
        get_scatter_plot([box2d[3,0], box2d[0,0]], [box2d[3,1], box2d[0,1]], color=color, **kwargs),
    ]


# Defines camera view settings for 3D Plotly visualization
PCD_CAM_VIEW = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=-0.9, y=0, z=0.2)
    )

# Imports helper function to create 2D image plots (RGB, BEV, etc.)
from utils import get_image2d_plots
# Imports Plotly utility to create multi-panel (grid) figures
from plotly.subplots import make_subplots
#  Defines visualization class for autonomous driving outputs
class Visualizer:
    def __init__(self, model_name, fig_width=1000, fig_height=800,
                 pred_box_color='white', waypoints_color = 'red',
                 bbox_2d_color = 'cyan', scene=PCD_SCENE, cam_view=PCD_CAM_VIEW):
        # Stores model name for display
        self.model_name = model_name
        # Create a figure:
        # - Top row → 3D LiDAR visualization
        # - Bottom rows → 2D visualizations (RGB, BEV, etc.)
        self.fig = make_subplots(rows=3, cols=2,
                                 specs=[[{"type": "scatter3d", "colspan": 2}, None],
                                        [{}, {"rowspan": 2}],
                                        [{}, None]],
                                row_heights=[0.6, 0.2, 0.2], horizontal_spacing=0.0, vertical_spacing = 0.0)
        # Applies predefined scene + camera settings for 3D plot
        self.fig.update_layout(template="plotly_dark", scene=scene, scene_camera = cam_view,
                height = fig_height, width = fig_width, autosize=False,
                title=f"END TO END AUTONOMOUS DRIVING {self.model_name}", title_x=0.5, title_y=0.95,
                margin=dict(r=0, b=0, l=0, t=0))
        # Hide axes for 2D subplots for clean visualization
        for row in range(2,4):
            for col in range(1,4):
                self.fig.update_xaxes(showticklabels=False, visible=False, row=row, col=col)
                self.fig.update_yaxes(showticklabels=False, visible=False, row=row, col=col)
        # Enables exporting figure as PNG using Kaleido engine
        self.fig.to_image(format="png", engine="kaleido")
        # Color for predicted 3D bounding boxes
        self.pred_color = pred_box_color
        # Color for trajectory/waypoints
        self.waypoints_color = waypoints_color
        # Color for 2D bounding boxes (on RGB/BEV)
        self.box2d_color = bbox_2d_color
    # Clears all existing traces (plots) from the figure
    def clear_figure_data(self):
        # updating visualization for a new frame
        self.fig.data = []
    # Returns list of colors for bounding boxes:
    # - One color per box (same color repeated)
    # - If no boxes → returns None
    def get_bbox_colors(self, bbox_corners):
        return [self.pred_color] * bbox_corners.shape[0] if bbox_corners is not None else None
    #  Creates 3D mesh plot of predicted trajectory waypoints
    def plot_waypoints(self, waypoints):
        return go.Mesh3d(x=waypoints[:,0], y=waypoints[:,1], z=waypoints[:,2],
                         opacity=0.4, color=self.waypoints_color,
                         hoverinfo='skip',showlegend=False)
    # Adds LiDAR point cloud + bounding boxes + waypoints to figure
    def add_lidar_plots(self, points, waypoints, pred_corners=None):
        lidar_3d_plots = get_lidar3d_plots(points, pc_kwargs=dict(colorscale='viridis', marker_size=0.9),
                                   pred_box_corners = pred_corners,
                                   pred_box_colors = self.get_bbox_colors(pred_corners))
        # Adds waypoint trajectory mesh to LiDAR plots
        lidar_3d_plots.append(self.plot_waypoints(waypoints))
        # Adds all traces to first subplot (top row, 3D view)
        for trace in lidar_3d_plots:
            self.fig.add_trace(trace, row=1, col=1)
    # Adds 2D visualizations (RGB, depth, BEV + boxes) to figure
    def add_image_plots(self, rgb_image, depth_image, lidar_data, pred_corners_2d):
        # Adds RGB camera image to subplot (row 2, col 1)
        self.fig.add_trace(get_image2d_plots(rgb_image), row=2, col=1)
        # Converts single-channel depth → 3-channel image (H, W, 3)
        depth_image = np.tile(depth_image[:, :, None], (1,1,3))
        #  Applies color map to depth: converts depth values to colored heatmap & enhances visualization (near/far distinaction)
        depth_image = cv2.applyColorMap(np.uint8(255 * depth_image), cv2.COLORMAP_JET)
        # Adds colored depth image to subplot (row 3, col 1)
        self.fig.add_trace(get_image2d_plots(depth_image), row=3, col=1)
        # Adds LiDAR BEV image to subplot (row 2, col 2)
        self.fig.add_trace(get_image2d_plots(lidar_data), row=2, col=2)
        # Creates list of colors for each 2D bounding box
        box_colors = [self.box2d_color] * len(pred_corners_2d)
        # Iterates over each detected object
        for i, obj_i in enumerate(pred_corners_2d):
            # Converts box corners → 4 line segments (edges)
            obj_plots = plot_box_corners2d(obj_i, color = box_colors[i])
            # Draws each edge of bounding box on BEV image
            for plot in obj_plots:
                self.fig.add_trace(plot, row=2, col=2)
    # Main function to visualize full prediction pipeline
    def visualize_predictions(self, points, waypoints, pred_corners_3d,
                              rgb_image, depth_image, lidar_data, pred_corners_2d):
        # Removes old plots from figure
        self.clear_figure_data()
        # Adds 3D LiDAR visualization point clound, predicted 3D bounding boxes, predicted trajecoty(waypoints)
        self.add_lidar_plots(points=points, waypoints=waypoints, pred_corners=pred_corners_3d)
        # Adds 2D visualizations RGB image, Depth map, BEV LiDAR + 2D Bounding boxes
        self.add_image_plots(rgb_image, depth_image, lidar_data, pred_corners_2d)
    # Display the complete visualization dashboard
    def show_figure(self):
        self.fig.show()
    # Saves current figure as PNG file
    def save_to_png(self, output_path):
        self.fig.write_image(output_path)

"""
Image & Video Output
"""
# Commented out IPython magic to ensure Python compatibility.
# Sets matplotlib usefuls for saving plots/ images without display them
# %matplotlib agg
# Creates DataLoader for demo dataset
dataloader_demo = DataLoader(demo_set, shuffle=False, batch_size=2, num_workers=4)
# Initializes visualization dashboard
visualizer = Visualizer(model_name='TRANSFUSER')
#  Utility to convert 3D boxes → corner points
from utils import boxes_to_corners_3d
# Counter for saving frames
frameIdx = 0
# Loop through dataset with progress bar
for data in tqdm(dataloader_demo):
    # load data to device, according to type
    for k in ['rgb', 'depth', 'lidar', 'label', 'ego_waypoint',
              'target_point', 'target_point_image', 'speed']:
        data[k] = data[k].to(device, torch.float32)
    # Moves all inputs to GPU/CPU
    for k in ['semantic', 'bev']:
        data[k] = data[k].to(device, torch.long)
    # get model predictions (waypoints, detections, depth, BEV, etc.)
    _, outputs = model(data)
    # iterate through each sample in batch
    bs = data['rgb'].shape[0]
    for i in range(bs):
        #  ================= INPUT DATA =================
        # Extract RGB image (H, W, C)
        rgb_image = data['rgb'][i].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        # Ground-truth waypoints
        tgt_waypoints = data['ego_waypoint'][i].detach().cpu().numpy()
        # Extract valid LiDAR point cloud
        lidar_pc = data['raw_lidar'][i].detach().cpu().numpy()
        num_points = data['num_raw_lidar_points'].detach().cpu().numpy()[i]
        lidar_pc = lidar_pc[:num_points, :]
        # Convert BEV LiDAR to image format
        lidar_data = data['lidar'][i].detach().cpu().numpy().transpose(1,2,0)
        lidar_data = (lidar_data * 255).astype(np.uint8)
        #  ================= MODEL PREDICTIONS =================
        # Predicted trajectory
        pred_waypoints = outputs['pred_wp'][i]
        # Convert from LiDAR → vehicle frame (flip Y)
        pred_waypoints[:, 1] *= -1
        # Generate left/right lane boundaries from trajectory
        pred_lanepoints = generate_lane_points(pred_waypoints)
        # ================= AUXILIARY TASKS =================
        # Predicted BEV bounding boxes
        pred_boxes = outputs['detections'][i]
        # Convert to 3D boxes (LiDAR frame)
        pred_3d_boxes = convert_to_3d_bboxes(pred_boxes)
        # Convert to 8-corner representation for visualization
        pred_corners_3d = boxes_to_corners_3d(pred_3d_boxes)
        # Convert boxes → rotated 2D corners for BEV image overlay
        pred_corners_2d = [get_rotated_bbox(bbox)[:, :2] for bbox in pred_boxes]
        ## Predicted depth map scaled to 0–255
        pred_depth = (outputs['pred_depth'][i] * 255).astype(np.uint8)
        pred_bev = outputs['pred_bev'][i].argmax(axis=0).astype(np.uint8)
        # ================= VISUALIZATION =================
        # Combines all modalities: 3D LiDAR + Boxes + trajectory, RGB image, Depth map, BEV LiDAR + 2D boxes
        visualizer.visualize_predictions(lidar_pc, pred_lanepoints, pred_corners_3d,
                                         rgb_image, pred_depth, lidar_data,
                                         pred_corners_2d = pred_corners_2d)
        # Saves visualization as image file
        visualizer.save_to_png(f"Frame{frameIdx}.png")
        # Increment frame index
        frameIdx +=1

# Imports natsorted:
# - Sorts filenames in natural order (Frame1, Frame2, ..., Frame10)
from natsort import natsorted
# Converts sequence of images → video file
def convert_images_to_video(images_dir, output_video_path, fps : int = 8):
    # Lists all PNG images in directory and sorts them naturally
    input_images = [x for x in natsorted(os.listdir(images_dir)) if x.endswith('png')]
    # Converts filenames → full file paths
    input_images = [os.path.join(images_dir, x) for x in input_images]
    # Proceed only if images exist
    if(len(input_images) > 0):
        # Reads first image to get frame dimensions
        sample_image = cv2.imread(input_images[0])
        height, width, _ = sample_image.shape
        # handles for input output videos
        output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
        # Progress bar for tracking video creation
        num_frames = int(len(input_images))
        pbar = tqdm(total = num_frames, position=0, leave=True)
        # Iterates over all image frames with progress bar
        for i in tqdm(range(num_frames), position=0, leave=True):
            # Reads each image frame
            frame = cv2.imread(input_images[i])
            # Writes frame into video
            output_handle.write(frame)
            # Updates progress bar
            pbar.update(1)
        # Finalizes and saves video file
        output_handle.release()
    # If no images found, do nothing
    else:
        pass

# Sets frames per second for output video:
# - Lower FPS → slower playback
# - Higher FPS → smoother/faster video
FPS = 6
# Converts saved images into video
convert_images_to_video('./', f'scenario1_route13_{model.pred_len}pts_{FPS}fps.mp4', fps=FPS)

"""
12. Video Multi-Model Visualization End-To-End Autonomous Driving System
Video Output Understadning
Video represents a complete end-to-end autonomous driving system where perception, sensor fusion, object detection, and trajectory planning are integrated into a unified pipeline. The visualization captures multiple 
synchronized modalities per frame, providing a comprehensive view of how the system interprets its surroundings and generates driving decisions. It is not just a visualization—it is a full system-level demonstration of an 
AI-driven autonomous agent, combining deep learning, sensor fusion, and control.
1. Overall Structure of the Video
Each frame in the video is composed of multiple sub-visualizations, arranged in a structured layout:
(a) Top Section (Primary View)    
    - 3D LiDAR point cloud
    - Predicted 3D bounding boxes
    - Predicted trajectory / waypoints
(b) Bottom Section (Auxiallary View)    
    - RGB camera image
    - Depth prediction
    - BEV (Bird’s Eye View) LiDAR representation
    - 2D bounding boxes projection
Layout is extremely important because it reflects how modern autonomous systems operate internally:
    - No Single Sensor Is Enough
    - Multi-Model Fusion Is Required
    - Decisions are made using combined spatial + semantic understanding
2. LiDAR Point Cloud (3D Perception Layer)
Most dominant and information-rich component of the video is the LiDAR point cloud visualization.
(a) What See
    -  Sparse colored points forming arcs
    -  Circular / semi-circular scan patterns
    -  Dense region near ego vehicle
    - Sparse Distant Regions
  (b) What It Means
    - LiDAR works by emitting laser pulses and measuring return times. This produces:3D spatial coordinates (x, y, z), often intensity values
    - Obsevations
        1. Arc Patterns : represents rotationla scanning
        2. Density Variation : High density near sensor, low density far away
        3. Cluster : represents physical objects are vehicles, Bulidings, Obstacles
Importance for LiDAR Provides accurate 3D geometry, robust performance in low light, reliable distance meaurement
3. Predicted 3D Bounding Boxes (Detection Module)
(a) What See
      - Rectangular boxes aligned with objects
      - Positioned in 3D space
      - Rotated based on object orientation
(b) What It Means
      - Each bounding box encodes:
          (x, y, z, length, width, height, yaw)
      - Meaning
          - (x, y, z) → object center
          - length, width, height -> object size
          - yaw -> orientation (speed, brake)
4. Predicted Waypoints (Planning Module)
(a) What See
      - A sequence of points extending forward
      - Smooth curve or line
      - Positioned relative to ego vehicle
(b) What It Means
      - model predicts: future positions of the ego vehicle
      - Interpretation trajectory reflects road structure, obstacles, driving goal
(c) Obsevation
      - Smoothness : No sudden jumps, indicates stable planning
      - Forward Progression : Increassing X Values
      - Lateral Adjustment : Avoid Obstacles, Fllow Lane
5. BEV (Bird’s Eye View) Representation
(a) What See
    - Drivable area (light color)
    - Lane markings (thin lines)
    - Non-drivable areas (dark regions)
(b) Why BEV is important: Easier for planning
(c) BEV output shows: road layout, lane stucture, navigable space
6. Depth Map (3D Understanding from RGB)
(a) What See
    - Heatmap-like image
    - Bright = close
    - Dark = far  
(b) What It Means
      - Dense depth estimation: distance per pixel
(c) Depth map obsevation are understand 3D from 2D images, complement LiDAR, improve fusion
7. Semantic Segmentation (Scene Understanding)
(a) What See
      - Colored regions
      - Colored regions
      - Typical classes are include road, lane, vehicles, background
(b) What It Means
        -Pixel-level understanding
(c) Segmentation helps Identify drivable area, Detect boundaries, Understand scene layout
8. Sensor Fusion
    (a) Why fusion matters are rich semantics on RGB, and accurate geomety on LiDAR
"""

