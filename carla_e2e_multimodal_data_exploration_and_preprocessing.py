# Imports kagglehub library used to download datasets directly from Kaggle inside notebooks
import kagglehub
# Downloads the "transfuser-e2e-scripts"
Helper_Scripts = kagglehub.dataset_download("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Helper Scripts")
#  Downloads the CARLA driving dataset contrians training and validation data (RGB, LiDAR, BEV, labels, etc.)
DataSet = kagglehub.dataset_download("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/DataSet")
# Prints confirmation message indicating datasets have been successfully downloaded
print('Data source import complete.')

# Imports built-in 'os' module, which provides functions to interact
import os
#  Lists all files and folders inside the directory
os.listdir("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning")
# Imports OpenCV library for image processing tasks (reading images, transformations, visualization, etc.)
import cv2
# Imports system-specific parameters and functions
import sys
# Adds a custom directory to Python’s search path so that modules/scripts
sys.path.append("F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/Helper Scripts")

#  Imports UltraJSON, a faster alternative to Python’s built-in json library for reading and writing JSON data efficiently
import ujson
#  Imports NumPy for numerical operations, arrays, and matrix computations
import numpy as np
# Imports tqdm for creating progress bars in loops useful for tracking long-running tasks
from tqdm import tqdm
# Imports Matplotlib’s pyplot module for plotting graphs and visualizing images
import matplotlib.pyplot as plt
#  Configures NumPy array printing:
# - suppress=True → prevents scientific notation (e.g., prints 0.0001 instead of 1e-4)
# - precision=4 → limits decimal values to 4 digits for cleaner output
np.set_printoptions(suppress=True, precision=4)

# Imports the GlobalConfig class, which contains configuration settings
# (e.g., image size, sequence length, sensor settings, etc.) for the model and dataset
from config import GlobalConfig
# Imports the CARLA_Data class, which is a custom dataset loader used to laod and preprocess CARLA autonomous directory in Kaggle
from data import CARLA_Data
# Defines the path to the training dataset directory
root_dir = "F:/Complete Project Stores/Thesis Work/3. Towards Interpretables End-To-End Autonomous Driving Via Multimodal Transformer Fusion And Neural Motion Planning/DataSet/train"
# Creates an instance of the configuration class with default parameters
config = GlobalConfig()
# Initializes the dataset:
# - root → path where the dataset is stored
# - config → configuration settings used for preprocessing and loading data
train_set = CARLA_Data(root=root_dir, config=config)
# Prints the total number of samples (data points) available in the training dataset
# len(train_set) internally calls the dataset's __len__() method
print(f"There are {len(train_set)} samples in training set")

"""
The dataset we collected contains two folders: Train and Val, and each element is structured as follows:
- Train
    - Scenario_Name_1/
        - depth/
            - 0000.png
        - label_raw/
            - 0000.json
        - lidar/
            - 0000.npy
        - measurements/
            - 0000.json
        - rgb/
            - 0000.png
        - semantics/
            - 0000.png
        - topdown/
            - 0000.png
    - Scenario_Name_2/
Some elements are straightforward, in particular the images (RGB, segmentation, depth, ...) which need very little scripts to load. The other part are the JSON Files 
(containing the driving information we need (steering, accel, waypoints, ...) and the NPY files, used to compress large point clouds.
Some extra elements:
* Each route contains sensor data, where the ego vehicle goes from a start location to end location, while interacting with other agents in the environment.
* There are different Towns, each with its own layout in CARLA environment
* The routes are also categorised by Scenarios, where a specific set of events occur. Example: Ego vehicle trying to make Right turn, at an intersection with crossing traffic
In the dataset file, as well as in the config file, there are functions defined for us.
* For example, in the config file, can see:
* pred_len = 4: future waypoints predicted
* img_resolution = (160, 704): image pre-processing in H, W
*  pixels_per_meter = 8.0: How many pixels make up 1 meter. 1 / pixels_per_meter = size of pixel in meters
And this matters when we process our dataset. Similarly, if you start digging into the dataset file, you'll see some functions, such as:
* get_depth(data)
* get_waypoints(labels, len_labels)
* transform_waypoints(waypoints)
* crop_image_cv2(image, crop=(128, 640), channelFirst=False)
Let's load a sample from our "train_set", just to try and see what we have to understand during this workshop.
"""

# Sets the index of the dataset sample want to inspect
index = 0
# Converts the returned dictionary into a list of its keys and prints them this help see what types of data are avalible in one sample (e.g: rgb, lidar, segmentation, depth, ego_waypoint, etc)
print(list(train_set[index]))
print()
# Accesses a specific field ('ego_waypoint') from the selected sample 'ego_waypoint' typically contain future trajectory point (x, y coordinates) that the vehicle should follow used for planning
print("Data: ", train_set[index]['ego_waypoint'])

"""
1. Images
RGB image
The easiest thing can do is visualize the image from your dataset. could simply load the image file manually
From the paper on TransFuser: 
Link : https://arxiv.org/pdf/2205.15997
For the RGB input, use three cameras (facing forward, 60◦ left and 60◦ right). Each camera has a horizontal FOV of 120◦.
Extract the images at a resolution of 960× 480 pixels**, which crop to 320 × 160 to remove radial distortion at the edges.
These three undistorted images are composed into a single image input to the encoder, which has a resolution of 704 × 160 pixels and 132◦ FOV.
"""

# Prints the file path of the RGB image corresponding to the selected dataset index
print(train_set.images[index])
# Load RGB image, scale to resolution, change to (C, H, W) format
# Reads the image from disk using OpenCV:
# - Converts the file path to string (handles byte encoding if needed)
# - cv2.IMREAD_COLOR loads the image in BGR format (default in OpenCV)
rgb_image = cv2.imread(str(train_set.images[index], encoding='utf-8'), cv2.IMREAD_COLOR)
# Converts image from BGR (OpenCV default) to RGB format
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
# Prints the shape of image array -> (height, width, Channels)
print(rgb_image.shape)
# Display the image Using matplotlib
plt.imshow(rgb_image)
# Renders the plotted image on the screen
plt.show()
"""
1. RGB Image
Crop the extremities of the image to match the TransFuser paper input.
"""
# Defines a function to crop an image from the center
# - Image -> input image
# - crop -> desired crop size (height, width)
# - channelFirst → whether to convert output format to (C, H, W)
def crop_image_cv2(image, crop=(128, 640), channelFirst=False):
    # Extracts the width of the image
    width = image.shape[1]
    # Extracts the height of the image
    height = image.shape[0]
    # Unpacks the desired crop height and width
    crop_h, crop_w = crop
    # Calculates the starting Y coordinate (top) for center cropping
    start_y = height // 2 - crop_h // 2
    # Calculates the starting X coordinate (left) for center cropping
    start_x = width // 2 - crop_w // 2
    # Crops the image using NumPy slicing: [rows (y-axis), columns (x-axis)]
    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    # If required, converts image format from (H, W, C) → (C, H, W)
    if channelFirst:
        cropped_image = np.transpose(cropped_image, (2, 0, 1))
    # Returns the cropped (and possibly transposed) image
    return cropped_image

# Crops the RGB image using the previously defined function:
# - config.img_resolution specifies the target (height, width)
# - crop is taken from the center of the image
rgb_image = crop_image_cv2(rgb_image, crop=config.img_resolution)
# Prints the shape of the cropped image → (Height, Width, Channels)
print(rgb_image.shape)
# Displays the cropped RGB image using matplotlib
plt.imshow(rgb_image)
# Renders the image output on the screen
plt.show()
"""
2. Cropped RGB Image
Image Given Output:
  1. Narrower Field of View (cropped) -> Focuses more on the center of the scene (vehicles ahead).
  2. Higher relative vertical detail -> Objects (bike, car) appear larger and clearer.
  3. Less background noise -> Removes unnecessary side regions → better for model input.
Likely after applying your crop_image_cv2()
"""

""""
BEV image : load a BEV image**, which is stored as an encoded image. file is encoded to contain different classes. crop the BEV image to 32m x 32m grid.
"""
# Prints the file path (or reference) of the BEV (Bird’s Eye View) data  this typically points to a stored BEV image or array representing the top-down view of the environment from LiDAR or sensor fusion.
print(train_set.bevs[index])

# Reads the BEV (Bird’s Eye View) image from disk:
# - Converts path to string (handles byte encoding)
# - IMREAD_UNCHANGED ensures original format is preserved (no compression or channel change)
bev_array = cv2.imread(str(train_set.bevs[index], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
# Converts BEV image from BGR (OpenCV default) to RGB format
bev_array = cv2.cvtColor(bev_array, cv2.COLOR_BGR2RGB)
# Prints:
# - Shape of BEV array → (Height, Width, Channels)
# - Unique pixel values → useful to understand encoding (e.g., segmentation classes, intensity values)
print(f"bev_array shape = {bev_array.shape}, Unique values are {np.unique(bev_array)}")
#  Changes array format from (H, W, C) → (C, H, W)
bev_array = np.moveaxis(bev_array, -1, 0)
# Displays one channel of the BEV image:
# - bev_array[0], bev_array[1], bev_array[2] correspond to different channels
# - Each channel may represent different semantic or spatial information
plt.imshow(bev_array[2])
# Renders the plotted BEV channel image
plt.show()
"""
3. Bird Eye View
Output Image:
-> Image represents a **Bird’s Eye View (BEV) semantic map** of a road intersection, commonly used in autonomous driving systems for spatial understanding and planning. scene is visualized from a top-down perspective, 
where different colors encode different semantic classes of the environment.
-> The dark purple regions indicate non-drivable or background areas (such as sidewalks, buildings, or empty space), while the **greenish/teal region** forms a clear T-shaped intersection, representing the drivable road surface 
this structure shows a vertical road intersecting with a horizontal road from the left, suggesting a junction where the ego vehicle may need to make navigation decisions (e.g., go straight or turn).
-> Within the drivable area, the bright yellow parallel lines correspond to lane markings, indicating lane boundaries and road structure. On the vertical road, multiple lane lines suggest a multi-lane segment, while the horizontal 
segment also contains lane markings, reinforcing the structured layout of the road network.
-> The geometry is clean and grid-like, indicating that this is not a raw sensor image but a processed semantic BEV representation, likely derived from LiDAR. such maps are crucial for downstream tasks like path planning, trajectory 
prediction, and control, as they provide a simplified yet information-rich representation of the environment, focusing only on relevant driving features like drivable space and lane topology.
-> Overall, the image captures a structured intersection with clearly defined drivable regions and lane boundaries, serving as an ideal input for autonomous driving models that rely on BEV representations for decision-making.
"""

"""
Visuallize each element of the encoded Bird Eye View
"""

# Crop portion of BEV image, visible to ego vehicle
# Imports helper functions:
# - decode_pil_to_npy → decodes BEV image into multi-channel numpy format
# - load_crop_bev_npy → crops BEV to region relevant for ego vehicle (top-down ROI)
from data import decode_pil_to_npy, load_crop_bev_npy
# # Decodes the BEV image into structured channels (e.g., lanes, drivable area, objects)
# Converts data type to uint8 (0–255 range), commonly used for image-like data
loaded_bevs = decode_pil_to_npy(bev_array).astype(np.uint8)
# # Crops the BEV around the ego vehicle:
# - Keeps only the region visible/relevant for driving
# - degree=0 means no rotation (aligned with default orientation)
bev = load_crop_bev_npy(loaded_bevs, degree=0)
#  Prints shape of decoded BEV (typically multi-channel: C, H, W)
print(f"Decoded bev shape = {loaded_bevs.shape}")
# Prints:
# - Shape of cropped BEV
# - Unique values → helps verify semantic classes or occupancy encoding
# - Data type → ensures correct format (uint8)
print(f"Cropped bev shape = {bev.shape}, unique values = {np.unique(bev)}, dtype = {bev.dtype}")

# Creates a figure with 1 row and 4 columns of subplots
# figsize controls the size of the entire figure
fig, ax  = plt.subplots(1,4, figsize=(18,6))
# Displays the original BEV image:
# - Converts from (C, H, W) → (H, W, C) for visualization
# - moveaxis shifts channel dimension to last position
ax[0].imshow(np.moveaxis(bev_array, 0, 2))
# Sets title for first subplot
ax[0].set_title("Original BEV map")
# Displays channel 0 of decoded BEV: typically represents drivable area
ax[1].imshow(loaded_bevs[0])
# Title for second subplot
ax[1].set_title("Drivable area")
# Displays channel 1 of decoded BEV:
# - Typically represents lane markings
ax[2].imshow(loaded_bevs[1])
# Title for third subplot
ax[2].set_title("Lane markings")
# Displays the cropped BEV: region focused around ego vehicle for planning
ax[3].imshow(bev)
# Title for fourth subplot
ax[3].set_title("Cropped BEV")
"""
4. Original & Drivable & Lane Marking & Cropped Bird Eye View 
Output Image:
-> Figure illustrates a complete BEV (Bird’s Eye View) processing pipeline used in autonomous driving, showing how raw sensor-derived data is progressively transformed into structured representations for perception and planning.
-> First panel, Original BEV map displays the raw top-down representation of the environment, likely generated from LiDAR or fused sensor inputs. map uses intensity values to represent occupancy or reflectance, with visible 
structures forming a T-shaped intersection. Bright spots indicate detected objects or higher-intensity returns, while darker regions represent empty space or unknown areas. raw format contains rich but unstructured information.
-> Second panel, Drivable area simplifies the scene into a semantic map where the road surface is clearly highlighted (in yellow) and everything else is suppressed into a dark background. this step extracts only the navigable 
regions, forming a clean intersection topology, which is crucial for decision-making and safe path planning.
-> Third panel, Lane markings further refines the representation by isolating lane boundaries. Thin, bright lines indicate lane dividers on both the vertical and horizontal road segments. this representation encodes the road geometry 
and lane structure** which is essential for maintaining lane discipline and understanding how vehicles should move within the road.
-> Finally, the fourth panel, Cropped BEV focuses on a region of interest centered around the ego vehicle’s forward direction It removes irrelevant areas and retains only the most important portion of the scene—primarily the lane 
ahead. this cropped map is more compact and efficient, making it ideal as input for downstream modules such as trajectory prediction, control, or planning networks.
-> Overall, the image demonstrates a step-by-step transformation pipeline: from raw BEV data → semantic drivable space → lane structure extraction → focused region cropping. this progression highlights how autonomous systems convert 
complex sensor data into clean, structured, and task-specific representations that enable accurate perception and efficient decision-making.
"""

"""
Depth image
-> image codifies depth value per pixel using 3 channels of the RGB color space, from less to more significant bytes: R -> G -> B. actual distance in meters can be decoded with:
-> normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
-> in_meters = 1000 * normalized
"""
# prints the file path to the depth map corresponding to the selected dataset index
print(print(train_set.depths[index]))

"""
Normalize the depth image to get a correct ouput**
"""
# Reads the depth map image from disk using OpenCV convert file path to string
depth_image = cv2.imread(str(train_set.depths[index], encoding='utf-8'))
# Converts depth image from BGR → RGB format for correct visualization
depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
# Crops the depth image to the desired resolution:
# - Uses center cropping
# - channelFirst=True converts format from (H, W, C) → (C, H, W)
depth_image = crop_image_cv2(depth_image, crop=config.img_resolution, channelFirst=True)
# Prints shape of processed depth image → (Channels, Height, Width)
print(depth_image.shape)
# Displays one channel of the depth image (index 2) depth maps are often encoded across channels
plt.imshow(depth_image[2])
# renders the plotted depth image
plt.show()
"""
5. Renders Plotted Depth Images
Output Image:
-> In the image, the foreground (road and nearby objects) appears in darker tones (purple/blue), indicating closer distances, while the background (sky and far-away regions) is shown in bright yellow, representing greater depth 
(farther away from the sensor). The scene itself depicts a road with vehicles ahead, roadside structures like fences, buildings, and trees, all captured in a compressed horizontal view similar to a camera frame used in autonomous
 driving.
-> Horizontal banding or streaking artifacts across the image, which are typical when depth data is either quantized, encoded in RGB channels, or improperly decoded before visualization. Despite these artifacts, the depth gradient 
is still clear: the road surface gradually transitions from near to far, and vertical structures like poles and trees show distinct depth discontinuities.
-> Type of depth map is crucial for autonomous systems because it provides 3D spatial understanding—allowing the vehicle to estimate how far objects are, detect obstacles, and plan safe trajectories. Unlike RGB images, which capture 
appearance, this representation focuses purely on geometry and distance
"""

# depth_image is in (C, H, W) format, it will display values for all channels show pixel intensity values for each channel (depth information encoded as RGB or multi-channel)
print(depth_image)

# Depth sensor data logic from CARLA
def get_depth(data):
    # Converts image format from (C, H, W) → (H, W, C) required because CARLA encodes depth in RGB channels
    data = np.transpose(data, (1,2,0))
    # Converts data type to float32 for precise numerical computation
    data = data.astype(np.float32)
    # Decodes depth from RGB channels:
    # - CARLA encodes depth using 24-bit format (R, G, B)
    # - Formula reconstructs depth value from 3 channels
    normalized = np.dot(data, [65536.0, 256.0, 1.0])
    # Normalizes depth values to range [0, 1]
    normalized /=  (256 * 256 * 256 - 1)
    # CARLA gets ground truth depth map for 1km clip to 50 meters to keep it realistic
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    #  Clips depth values: 0.05 corresponds to ~50 meters (since 1.0 = 1000m in CARLA)
    normalized = normalized * 20.0
    # Returns processed depth map (single-channel, normalized)
    return normalized
# Applies depth decoding and normalization to the input depth image
depth_image = get_depth(depth_image)
# Prints:
# - shape of depth map → (H, W)
# - data type → float32
print(f"depth_image shape = {depth_image.shape}, dtype = {depth_image.dtype}")
# Displays depth map using "inferno" colormap:
# - dark colors → near objects
# - bright colors → far objects
plt.imshow(depth_image, cmap="inferno");
# Renders the depth visualization
plt.show()
"""
6. Renders Plotted Depth Images
Output Image:
-> In the image, closer objects—such as the road surface, nearby vehicles, and pedestrians—are depicted in dark purple and blue tones, while farther regions, including the sky and distant background, transition into bright yellow 
and orange hues. this gradient clearly encodes depth: darker colors represent proximity, and brighter colors indicate increasing distance from the camera.
-> Scene itself captures a typical urban road environment. observe vehicles ahead, a motorcycle in the center, buildings on the left, and a fence with vertical poles on the right. depth map highlights the geometric structure of 
these elements: for example, the fence shows a repeating pattern of depth discontinuities, while the road gradually changes color as it extends into the distance, reflecting increasing depth.
-> Smooth transitions between colors indicate that the depth values are continuous and well-calibrated unlike earlier noisy representations. this makes the image highly useful for downstream tasks such as obstacle detection, 
distance estimation, and trajectory planning. Overall, this visualization demonstrates how depth sensing transforms a regular scene into a geometry-focused representation, enabling autonomous systems to understand the 3D structure 
of their surroundings.
"""

print(depth_image)

"""
Segmentation image
"""
# Reads the semantic segmentation image from disk:
# - uses IMREAD_UNCHANGED to preserve original class label values
# - each pixel represents a class ID (e.g., road, car, pedestrian, etc.)
semantic_image = cv2.imread(str(train_set.semantics[index], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
# Processes the semantic image:
# 1. crops the image to desired resolution using center cropping
# 2. applies 'converter' mapping:
#    - converts raw CARLA class IDs into a reduced/standard set of class labels
#    - often used to remap classes into training-friendly categories
semantic_image = train_set.converter[crop_image_cv2(semantic_image, crop=train_set.img_resolution)]
# Prints:
# - Shape of the semantic image → (H, W)
# - Unique class labels present in the image helps verify segmentation classes
print(f"semantic_image shape = {semantic_image.shape}, unique_values = {np.unique(semantic_image.flatten())}")
# Displays the semantic segmentation map "Paired" colormap assign different colors to diffrent class labels
plt.imshow(semantic_image, cmap="Paired")
# Renders the segmentation visualization
plt.show()
"""
7. Renders Depth Visualization 
Output Image:
-> In the image, the upper region (light blue) corresponds to the sky,indicating areas with no drivable relevance. The central region (pink) represents the road surface, clearly marking the drivable path ahead. On the sides, 
brown/orange regions likely correspond to sidewalks, terrain, or non-drivable areas separating the road from the environment. Small patches of other colors (such as green or darker tones)
-> Scene layout reflects a forward-facing driving view, where the road narrows into the distance, and surrounding structures like buildings, fences, and vegetation are abstracted into semantic categories. The segmentation simplifies 
the complex visual world into a structured map of classes making it easier for an autonomous system to reason about the environment.
-> Overall, this type of image is crucial for perception systems because it provides high-level scene understanding—identifying where the vehicle can drive, where obstacles exist, and how the environment is organized. It is commonly 
used in tasks such as lane detection, drivable area estimation, and decision-making in autonomous navigation.
"""

"""
2. Measurements, Point Clouds, and Bounding Boxes
* `measurements` — the ego vehicle data
* `label_raw` — the external labels
* `lidar` — the point cloud
"""

# Ego vehicle measurements
# Opens the measurements file (JSON format) corresponding to the selected index
with open(str(train_set.measurements[index], encoding='utf-8'), 'r') as f1:
    # Loads and parses the JSON file using ujson (fast JSON parser)
    # Stores the data as a Python dictionary this typically contains ego vehicle data such as:
    # position (x, y, z), speed, acceleration, steering, throttle, brake, GPS, IMU, etc.
    measurements = ujson.load(f1)
# Imports pprint (pretty print) for better formatted display of complex data structures
from pprint import pprint
# Prints the measurements dictionary
pprint(measurements)

""" 
Goal location
"""
# Extracts the orientation (heading angle in radians) of the ego vehicle
ego_theta = measurements['theta']
# Extracts the current global position (x, y) of the ego vehicle
ego_x = measurements['x']
ego_y = measurements['y']
# Extracts the target/goal position (command point) in global coordinates
x_command = measurements['x_command']
y_command = measurements['y_command']
# Constructs a 2D rotation matrix:
# - Rotates coordinates from global frame to ego vehicle frame
# - Adds π/2 to align CARLA coordinate system with vehicle heading
R = np.array([
    [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
    [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
    ])
# Translates the global command point into ego-centered coordinates
local_command_point = np.array([x_command-ego_x, y_command-ego_y])
# Applies inverse rotation transpose of R to convert from global → local frame
local_command_point = R.T.dot(local_command_point)
# Prints ego vehicle heading rounded to 3 decimal places
print(f"ego_theta = {ego_theta : .3f}")
# Prints global position of ego vehicle and command point
print(f"ego_x = {ego_x : .3f}, ego_y = {ego_y: .3f}, x_command = {x_command: .3f}, y_command = {y_command: .3f}")
# Prints the final command point in ego (local) coordinate frame this model & planner typically uses for navigation
print(f"local_command_point = {local_command_point}")

"""
Instead of directly feeding in the target point, plot the destination coordintes in BEV frame and provide it as input
"""
# Imports a helper function that converts a target point into a BEV/LiDAR map representation
from data import draw_target_point
# Draws the local command point (ego frame) onto a BEV/LiDAR grid:
# - Converts (x, y) coordinate into a spatial map
# - Typically creates a 2D grid with a marker (e.g., a point or small blob)
# - Used as input for planning networks (helps model know target direction)
tgt_pt_in_lidar = draw_target_point(local_command_point)
# Prints the shape of the generated target map
# Usually (C, H, W), where:
# - C = number of channels (often 1)
# - H, W = spatial resolution of BEV grid
print(f"Target point in lidar shape = {tgt_pt_in_lidar.shape}")
# Visualizes the first channel of the target map show where the target point lies relative to the ego vehicle is BEV space
plt.imshow(tgt_pt_in_lidar[0]);
"""
8. Target Points Lies Ego Vehicle Bird Eye View Space
Image Output
-> Image represents a target point map in Bird’s Eye View (BEV) space, commonly used in autonomous driving systems for goal-conditioned planning. The visualization is mostly a dark purple background, indicating empty or non-activated 
regions of the spatial grid, with a small bright yellow/green circular blob near the top center. this highlighted region marks the target waypoint (goal location) that the ego vehicle is expected to move toward.
-> Map is structured as a 2D grid representation (top-down view), where each pixel corresponds to a spatial location around the vehicle. The absence of other features suggests that this is not a full environment map (like LiDAR or 
semantic BEV), but rather a sparse supervisory signal used by the model to indicate direction or destination.
-> Position of the bright spot near the top edge implies that the goal lies ahead of the vehicle, consistent with forward navigation. Its centered horizontal placement suggests the target is roughly aligned with the vehicle’s 
current heading, meaning the vehicle likely needs to continue straight
-> Used as an additional input to neural networks (e.g., GRU-based planners) to guide trajectory prediction. By encoding the goal as a spatial heatmap, the model can easily integrate it with other BEV features such as LiDAR or 
semantic maps.
"""

"""
Lidar Point Cloud
"""
# Prints the file path of the LiDAR data corresponding to the selected dataset index.
# This usually points to a file containing LiDAR information such as:
# - Point cloud data (x, y, z, intensity)
# - Or a preprocessed BEV / voxelized representation
print(train_set.lidars[index])

# Loads the LiDAR data file (usually a .npy file):
# - converts path to string (handles encoding)
# - allow_pickle=True allows loading complex stored objects
# - [1] extracts the actual point cloud data (often stored as second element)
#   format is typically (N, 4) → [x, y, z, intensity]
lidars_pc = np.load(str(train_set.lidars[index], encoding='utf-8'), allow_pickle=True)[1]
# Flips the Y-axis of the point cloud:
# - converts coordinate system (CARLA ↔ model convention)
# - ensures consistency with BEV/map representation
lidars_pc[:, 1] *= -1
# Prints shape of the point cloud:
# - N = number of points
# - 4 = features per point (x, y, z, intensity)
print(f"Lidar point cloud shape = {lidars_pc.shape}")

# Selects LiDAR points with z ≤ -2.3:
# - lidars_pc[..., 2] refers to the z-coordinate (height)
# - These points are considered "below ground" (or very low points)
below = lidars_pc[lidars_pc[...,2]<=-2.3]
# Selects LiDAR points with z > -2.3:
# - These represent points above the threshold (road surface, vehicles, objects)
above = lidars_pc[lidars_pc[...,2]>-2.3]
# Prints number of points in each category
print(f"below shape = {below.shape}, above shape = {above.shape}")
# Plots below-ground points:
# - Uses x and y coordinates
# - Negates values to match visualization coordinate convention
# - s=0.2 sets very small point size since LiDAR has many points
plt.scatter(-below[:,0], -below[:,1], s = 0.2)
# Plots above-ground points on same graph helps visually distinguish spatial distribution of points
plt.scatter(-above[:,0], -above[:,1], s = 0.2)
#  Displays the scatter plot of LiDAR points top-down view
plt.show()
"""
9. LiDAR Points Top-Down View
Image Output:
-> Image shows a top-down visualization of a LiDAR point cloud, where spatial information about the surrounding environment is represented as a collection of scattered points. Each point corresponds to a LiDAR return, capturing the 
3D structure of objects around the ego vehicle. The plot is displayed in a 2D plane (x–y coordinates) effectively forming a Bird’s Eye View (BEV) of the scene.
-> At the center of the plot, there is a dense cluster of points representing the immediate surroundings of the ego vehicle—likely including nearby road surfaces and possibly parts of the vehicle itself. The points are color-coded 
(blue and orange), which typically indicates different categories or filtering conditions** such as separating points above and below a certain height threshold (e.g., ground vs. obstacles)
-> Blue points appear more structured and concentrated, forming curved and linear patterns that resemble nearby objects, road edges, or sensor reflections. On the right side, the blue points form arc-like patterns which are characteristic 
of LiDAR scanning geometry—these arcs often represent reflections from objects at increasing distances. The orange points are more scattered and dispersed, likely representing either ground points, noise, or points filtered based 
on height (e.g., below-ground or low-ground regions).
-> Spatial distribution extends roughly symmetrically along both axes, with values ranging from about -80 to +80 meters, indicating a fairly large sensing range. The forward direction (positive y-axis or upward in the image) contains 
more structured and dense information, consistent with the vehicle’s direction of motion.
-> LiDAR sensors perceive the world as discrete spatial points. Such representations are fundamental for tasks like obstacle detection,mapping,localization, and BEV feature generation, as they provide precise distance and structural 
information independent of lighting or texture.
-> LiDAR point cloud from simulator covers 360 degree for larger range (distance). crop the points within BEV range (32m x32m)
"""

# Generate BEV histogram features, separately for above ground and below ground points
# Imports a function that converts raw LiDAR point cloud into BEV histogram features
from data import lidar_to_histogram_features
# Converts LiDAR point cloud (XYZI) into BEV feature maps:
# - discretizes space into a grid (top-down view)
# - counts points falling into each grid cell (histogram)
# - typically separates:
#     channel 0 → above-ground points
#     channel 1 → below-ground points
lidars_pc = lidar_to_histogram_features(lidars_pc)
# Prints shape of generated BEV features → (C, H, W)
#  C = number of channels (e.g., 2: above & below ground)
print(f"Lidar histogram feature shape = {lidars_pc.shape}")
# Creates a figure with 2 subplots side-by-side
fig, ax  = plt.subplots(1,2, figsize=(16,6))
# Displays BEV histogram for above-ground points show density of objects like vehicles. pedestrians, obstacles
ax[0].imshow(lidars_pc[0]);
# Title for first subplot
ax[0].set_title("BEV Histogram feature for Above ground points");
# Displays BEV histogram for below-ground points captures road surface, ground structures
ax[1].imshow(lidars_pc[1]);
# Title for second subplot
ax[1].set_title("BEV Histogram feature for Below ground points");
"""
10. Bird Eye View Histogram Above & Below Ground Points 
Output Image
-> Figure presents a two-channel Bird’s Eye View (BEV) histogram representation of LiDAR data, where the point cloud has been discretized into grid-based features and separated based on height relative to the ground. Each panel 
visualizes how LiDAR points are distributed spatially after being converted into a density histogram over a 2D grid.
-> Left panel, titled BEV Histogram feature for Above ground points shows the spatial distribution of points that lie above a certain height threshold (e.g., vehicles, pedestrians, poles, and other obstacles). The bright spots and 
sparse clusters correspond to physical objects in the environment such as vertical structures or moving entities. These points are relatively scattered because objects occupy specific regions in space rather than forming continuous 
surfaces.
-> Right panel, titled BEV Histogram feature for Below ground points represents points at or below the ground level typically corresponding to the road surface and terrain. This panel exhibits a much denser and more structured 
pattern, including concentric arc-like formations that reflect the LiDAR sensor’s scanning pattern. The semicircular dense region near the bottom indicates the ego vehicle’s immediate surroundings where the sensor captures many 
returns from the ground.
"""

"""
BEV Bounding Boxes
Bounding boxes are represented by 2D rectangles, with [center x, center y, width, height, yaw] representation. Additionally, add the brake and speed information for each agent
"""
# Initializes an empty list to store labels for all timesteps (past + future)
labels = []
# Loops over total sequence length:
# - seq_len → past frames
# - pred_len → future frames (for prediction tasks)
for i in range(train_set.seq_len + train_set.pred_len):
    # Opens label file for timestep i
    with open(str(train_set.labels[index][i], encoding='utf-8'), 'r') as f2:
        # Loads JSON label file into Python dictionary each file typically contains annotations like : bounding boxes, object classes, position, velocities
        labels_i = ujson.load(f2)
    #  Appends loaded labels for this timestep to the list
    labels.append(labels_i)
# Imports pretty print for better visualization of nested data
from pprint import pprint
# Prints the first object annotation from the first timestep:
# - labels[0] → first timestep
# - labels[0][0] → first object in that frame
# Displays structured info (e.g., class, bbox, location, etc.)
pprint(labels[0][0])

# Prints the entire 'labels' list
# What this contains:
# - A list of timesteps (length = seq_len + pred_len)
# - Each element corresponds to one frame's annotations
# - Each frame typically contains a list of objects (cars, pedestrians, etc.)
# - Each object is represented as a dictionary (e.g., bbox, class, position)
print(labels)

# Imports a function that parses raw label dictionaries into structured bounding boxes
from data import parse_labels
# Parses labels of the current frame (timestep 0):
# - Filters objects within BEV field of view (32m x 32m)
# - Extracts bounding box information (e.g., x, y, w, h, orientation, class)
# - Returns a dictionary where each key corresponds to an object
bboxes = parse_labels(labels[0])
# Converts dictionary values (bounding boxes) into a NumPy array each row represents one objects bounding box featues
label = np.array(list(bboxes.values()))
# Prints shape of label array → (N, D)
# N = number of detected objects
# D = number of attributes per object (e.g., position, size, angle, class)
print(f"Label shape = {label.shape}")
#  Prints the full bounding box array each row corresponds to one object in the current frame
print(label)

# Displays the first channel of the LiDAR BEV histogram feature:
# - lidars_pc[0] typically represents "above-ground" point density
# - Each pixel shows how many LiDAR points fall into that grid cell
# - Brighter regions → higher point density (objects like cars, buildings)
plt.imshow(lidars_pc[0])
"""
11. LiDAR Bird Eye View Histogram
Output Image
-> Image represents a Bird’s Eye View (BEV) histogram feature map for LiDAR data specifically highlighting the above-ground points in the environment. It is a grid-based top-down representation where each pixel encodes the density 
of LiDAR returns falling into that spatial cell.
-> Background is predominantly dark purple indicating regions with little to no LiDAR returns, while the brighter cyan/green spots correspond to areas with higher point density. These brighter regions form **sparse, irregular clusters 
which typically represent objects above the ground such as vehicles, poles, pedestrians, or other vertical structures in the scene.
-> Observe thin vertical and scattered line-like patterns** which likely correspond to elongated objects like fences, poles, or edges of structures. Near the lower-left region, there are curved arc-like patterns reflecting the LiDAR 
sensor’s scanning geometry and how it captures nearby surfaces at different angles. A prominent vertical line toward the right side suggests a continuous structure (possibly a wall, fence, or roadside boundary).
-> Unlike ground-based features, which tend to form dense and smooth distributions, this map is sparse and object-centric, emphasizing only elevated elements in the environment. This makes it particularly useful for object detection 
and obstacle awareness, as it isolates relevant entities from the background.
"""

# Takes the first BEV channel (usually above-ground points) and makes a copy this will be used for visuallization so original data
lidars_pc_viz = lidars_pc[0].copy()
# Normalizes the BEV feature values to range [0, 255]:
# - divides by max value to scale to [0,1]
# - multiplies by 255 for image intensity range
# - converts to uint8 for visualization (image format)
lidars_pc_viz = (lidars_pc_viz / lidars_pc_viz.max() * 255).astype(np.uint8)
# Iterates over each bounding box in the label array
for lb in label:
    # Computes left (x1) and right (x2) coordinates of bounding box lb[0] = center x, lb[2] = width
    x1 = int(lb[0] - lb[2]/2)
    x2 = int(lb[0] + lb[2]/2)
    # Computes top (y1) and bottom (y2) coordinates of bounding box lb[1] = center y, lb[3] = height
    y1 = int(lb[1] - lb[3]/2)
    y2 = int(lb[1] + lb[3]/2)
    #  Draws rectangle (bounding box) on BEV image:
    # - (x1, y1) → top-left corner
    # - (x2, y2) → bottom-right corner
    # - color=(255,0,0) → blue in OpenCV (BGR format)
    cv2.rectangle(lidars_pc_viz, (x1,y1), (x2,y2),color=(255,0,0))
# Displays the BEV image with bounding boxes overlaid
plt.imshow(lidars_pc_viz)
# Renders the final visualization
plt.show()
"""
12. Bird View Image With Bounding Boxes Overlaid
Output Image
-> Image shows a Bird’s Eye View (BEV) LiDAR histogram feature map with overlaid bounding boxes representing detected objects in the environment. The base layer is a top-down grid visualization of LiDAR data where the dark purple 
background indicates empty regions, and the brighter cyan/green pixels represent areas with higher point density corresponding to physical structures or objects above the ground.
-> Superimposed on this map are several rectangular bounding boxes (outlined in light green) which highlight the locations of detected objects. These boxes are centered around clusters of LiDAR points, indicating that the system 
has identified these regions as potential obstacles or entities such as vehicles, pedestrians, or roadside objects. The bounding boxes vary in size, reflecting differences in object dimensions or distance from the sensor.
-> Spatial layout suggests that the ego vehicle is positioned near the bottom center with the forward direction extending upward. The detected objects appear both in front and slightly to the sides, indicating a multi-object 
scenario. Some boxes are larger and positioned farther away (top region), while smaller boxes closer to the center likely represent nearer or smaller objects.
-> Additionally, faint vertical and curved patterns in the LiDAR data—such as the line on the right and arc-like structures on the left—indicate environmental features like fences or sensor scanning artifacts** but these are not 
enclosed by bounding boxes, suggesting they are not classified as primary objects of interest.
"""

"""
Future Ego waypoints
"""
# Prints the first object annotation from the first timestep
# Dictionary containing details of that object, such as: class/ types (e.g: car, pedestrian), bouding box coordiates, position(x,y,z), rotation, orientation, velociy, brakes
print((labels[0][0]))

# train_set.pred_len → number of future timesteps used for prediction
print(train_set.pred_len+1)

# Ego car is always the first one in label file
# Accesses the ID of the ego vehicle:
# - labels[0] → current frame (timestep 0)
# - labels[0][0] → first object in that frame (assumed to be ego vehicle)
# - ['id'] → unique identifier of that object
ego_id = labels[0][0]['id']
# Prints the ego vehicle's ID
print(ego_id)

# Imports helper functions:
# - get_waypoints → extracts future ego positions from label data
# - transform_waypoints → converts waypoints into a desired coordinate frame (e.g., ego frame)
from data import get_waypoints, transform_waypoints
# Extracts future ego vehicle waypoints:
# - labels[train_set.seq_len - 1:] → starts from the last observed frame (current timestep) and includes future frames
# - train_set.pred_len + 1 → number of waypoints: includes current position + future positions
waypoints = get_waypoints(labels[train_set.seq_len-1:], train_set.pred_len+1)
# Prints the extracted waypoints (trajectory points) each entry typically represents ego vehicle position at a timestep
print(waypoints)

# Transforms the waypoint coordinates into a different reference frame:
# - Typically converts from global/world coordinates → ego vehicle coordinates
# - Aligns trajectory relative to vehicle’s current position and heading
# - Makes waypoints suitable as model training targets
waypoints = transform_waypoints(waypoints)
# Prints the transformed waypoints output usually contains sequence of (x, y) position in ego frame representing the future trajectory the vehicle should follow
print(waypoints)

# Extracts ego vehicle future waypoints from transformed waypoint data:
# - waypoints[ego_id] → gets all waypoint data corresponding to the ego vehicle
# - [1:] → skips the current timestep and keeps only future timesteps
# - x[0] → accesses the transformation matrix (usually 4x4 pose matrix)
# - [:2, 3] → extracts the translation components (x, y) from the matrix
# - [:2] selects first two rows (x, y)
# - column 3 contains position (translation vector)
# - Collects all (x, y) positions into a NumPy array
ego_waypoints = np.array([x[0][:2,3] for x in waypoints[ego_id][1:]])
# Prints shape of ego waypoints → (T, 2)
# T = number of future timesteps, 2 = (x, y)
print(f"ego_waypoints shape = {ego_waypoints.shape}")
# Prints the actual future trajectory points of the ego vehicle these are used as ground truth for trajectory prediction models
print(ego_waypoints)

# Plots the trajectory line of ego vehicle future waypoints:
# - ego_waypoints[:, 0] → x-coordinates
# - ego_waypoints[:, 1] → y-coordinates
# - Connects points to show path/trajectory
plt.plot(ego_waypoints[:,0], ego_waypoints[:,1]);
# Plots individual waypoint points on top of the line:
# - s=10 → small marker size
# - color='g' → green points
plt.scatter(ego_waypoints[:,0], ego_waypoints[:,1], s = 10, color='g');
# Ensures equal scaling on both axes (X and Y)
plt.axis('equal')
# Labels X-axis as position in meters
plt.xlabel('X (m)')
# Labels Y-axis as position in meters
plt.ylabel('Y (m)')
# Sets title of the plot describing what is being visualized
plt.title('Ego vehicle future waypoints')
"""
13. Ego Vehicle Future Waypoints
Ouput Image
-> Image presents a 2D trajectory plot of the ego vehicle’s future waypoints visualized in the vehicle’s local coordinate frame. The horizontal axis represents the X position (in meters) and the vertical axis represents the Y position 
(in meters) relative to the ego vehicle. The plotted trajectory consists of a straight horizontal line along the X-axis, with several discrete points marked in green, indicating the predicted or ground-truth future positions of the 
vehicle over time.
-> All the waypoints lie very close to Y = 0, which indicates that the vehicle is expected to move straight forward without any lateral deviation—there is no turning or lane change The progression of points along the X-axis, from 
negative to positive values, reflects the vehicle’s motion over time in the forward direction (depending on the coordinate convention, this could represent forward movement relative to the ego frame). The evenly spaced points suggest 
a smooth and consistent trajectory, likely corresponding to steady without abrupt acceleration or steering changes.
-> Overall, this plot depicts a simple, straight-line future trajectory, which is typical in scenarios where the vehicle continues along its current lane without obstacles or navigation changes. It serves as a ground-truth reference 
or prediction target for trajectory planning models in autonomous driving systems.
"""

