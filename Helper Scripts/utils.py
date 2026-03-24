# Generated from: utils.ipynb
# Converted at: 2026-03-19T13:14:03.430Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# used to encode and decode data in Base64 format (commonly used for images in web APIs)
import base64  

# numPy is used for numerical operations, especially arrays and matrices 
import numpy as np  

# PIL (Python Imaging Library) is used for opening, manipulating, and saving images
from PIL import Image  

# BytesIO allows to handle binary data (like images) in memory as a file-like object
from io import BytesIO  

# pyTorch library used for deep learning, tensor operations, and building neural networks
import torch

# plotly's graph_objects module is used to create interactive and customizable  visualizations (charts, graphs, plots)
import plotly.graph_objects as go

# Height of the camera sensor from the ground (in meters)
CAMERA_HEIGHT = 2.3  

# Height of the LiDAR sensor from the ground (in meters)
LIDAR_HEIGHT = 2.5  

# Defines the coordinate system axes (x, y, z) used for spatial representation
COORDINATE_AXIS = ['x', 'y', 'z']  

# Defines the vehicle’s coordinate axes (typically same as global/local frame)
VEHICLE_AXIS = ['x', 'y', 'z']  

# Forward distance (in meters) from the vehicle's origin to the LiDAR sensor
VEHICLE_TO_LIDAR_FWD = 1.3  

# Indices representing connections (likely edges or faces) between points, commonly used in 3D bounding boxes or geometric structures
# First set of indices connecting points 5 → 4 → 0 → 1 (forms one face/edge loop)
INDICES_1 = [5, 4, 0, 1]

# Second set connecting points 6 → 5 → 1 → 2
INDICES_2 = [6, 5, 1, 2]

# Third set connecting points 7 → 6 → 2 → 3
INDICES_3 = [7, 6, 2, 3]

# Fourth set connecting points 4 → 7 → 3 → 0
INDICES_4 = [4, 7, 3, 0]

# creates a transformation matrix from LiDAR frame to vechicle frame
def get_virtual_lidar_to_vehicle_transform():
    # initialize a 4x4 identity matrix used in homogeneous transformations
    T = np.eye(4)
    # set translation along x-axis forward direction of vehicle
    T[0, 3] = 1.3
    # set translation along y-axis sideways direction, here no offset
    T[1, 3] = 0.0
    # set translation along z-axis height from ground
    T[2, 3] = 2.5
    # return the transformation matrix no rotation, only translation
    return T

# computes the transformation matrix from vehicle frame to LiDAR frame
def get_vehicle_to_virtual_lidar_transform():
    # take the inverse of the LiDAR → vehicle transformation matrix this reverses 
    # the transformation direction: (vehicle → LiDAR) = inverse of (LiDAR → vehicle)
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

# defines the rotation matrix to convert coordiantes from LiDAR frame to vehicle 
def get_lidar_to_vehicle_transform():
    # rotation changes axis alignment (e.g., LiDAR x → vehicle y, etc.)
    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]], dtype=np.float32)
    # initialize a 4x4 identity matrix for homogeneous transformation
    T = np.eye(4)
    # insert the rotation matrix into the top-left 3x3 block
    T[:3, :3] = rot
    # set translation along x-axis forward offset of LiDAR from vehicle origin
    T[0, 3] = 1.3
    # set translation along y-axis no lateral offset
    T[1, 3] = 0.0    
    # set translation along z-axis height of LiDAR from ground
    T[2, 3] = 2.5
    # return full transformation matrix rotation + translation
    return T

# computes the transformation matrix from vehicle frame ato LiDAR frame
def get_vehicle_to_lidar_transform():    
    # take the inverse of the LiDAR → vehicle transformation matrix
    # this reverses both rotation and translation: (vehicle → LiDAR) = inverse of (LiDAR → vehicle)
    return np.linalg.inv(get_lidar_to_vehicle_transform())

# creates a transformation matrix to map LiDAR cooridnates -> BEV (Bird's Eye View)
def get_lidar_to_bevimage_transform():    
    # initial 2D affine transform (3x3):
    # - rotation: swaps and flips axes to align LiDAR frame with image frame
    # - translation: shifts origin so coordinates fall inside the image canvas
    T = np.array([[0, -1, 16],
                  [-1, 0, 32],
                  [0, 0, 1]], dtype=np.float32)
    # scale transformation:
    # - multiplies x and y coordinates by 8 (meters → pixels conversion) also scales translation accordingly
    T[:2, :] *= 8
    # return final LiDAR → BEV image transformation matrix
    return T

# normalizes an angles (in radians) to the range [--π, π]
def normalize_angle(x):
    # first, wrap the angle into [0, 2π)
    x = x % (2 * np.pi)
    # if angle is greater than π, shift it into [-π, π) this ensures symmetry around zero useful for rotations and heading angles
    if x > np.pi:
        x -= 2 * np.pi
    # return the normalized angle
    return x

# normalies an angle (in degrees) to the range [-180°, 180°]
def normalize_angle_degree(x):
    # first, wrap the angle into [0°, 360°]
    x = x % 360.0
    # if angle is greater than 180°, shift it into [-180°, 180°] this keeps the 
    # angle centered around zero for easier interpretation (e.g., yaw/heading)
    if (x > 180.0):
        x -= 360.0
    # return the normalized angle
    return x

# check whether the input 'x' is numpy array
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        # if it is a NumPy array:
        # - convert it to a PyTorch tensor
        # - cast it to float type (commonly used in deep learning models)
        # - return the tensor along with a flag indicating conversion was done (True)
        return torch.from_numpy(x).float(), True
    # if 'x' is already a PyTorch tensor (or any other type):
    # - return it as-is
    # - return False indicating no conversion was needed
    return x, False

def rotate_points_along_z(points, angle):
    """
    rotates 3D points around the Z-axis (yaw rotation)
    Args:
        points: (B, N, 3 + C)
            B = batch size
            N = number of points
            3 = (x, y, z) coordinates
            C = additional features (e.g., intensity, color, etc.)
        angle: (B)
            rotation angle (in radians) for each batch
            positive angle rotates points from x → y direction
    returns:
        rotated points with same shape as input
    """
    # ensure inputs are PyTorch tensors
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    # compute cosine and sine of rotation angles
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    # create helper tensors (same device & dtype as angle)
    zeros = angle.new_zeros(points.shape[0])  
    ones = angle.new_ones(points.shape[0])    
    # Construct batch-wise rotation matrices (B, 3, 3)
    # each matrix corresponds to rotation around Z-axis:
    # [ cosθ   sinθ   0 ]
    # [-sinθ   cosθ   0 ]
    # [  0      0     1 ]
    rot_matrix = torch.stack((cosa,  sina, zeros,
                              -sina, cosa, zeros,
                              zeros, zeros, ones
                              ),
                             dim=1).view(-1, 3, 3).float()

    # apply rotation to XYZ coordinates only (B, N, 3) × (B, 3, 3) → (B, N, 3)
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    # concatenate back any extra features (C)
    # final shape: (B, N, 3 + C)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    # if original input was NumPy, convert back to NumPy
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
    converts 3D bounding boxes into their 8 corner coordinates

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d: (N, 7)
            each box is represented as:
            [x, y, z, dx, dy, dz, heading]
            (x, y, z) → center of box
            (dx, dy, dz) → dimensions (length, width, height)
            heading → rotation around Z-axis (yaw)
    Returns:
        corners3d: (N, 8, 3)
            8 corner points for each box
    """
    # convert input to PyTorch tensor if it's NumPy
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    # Define a normalized cube template centered at origin values are in range 
    # [-0.5, 0.5] representing relative positions this defines the shape of a unit cube
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],  # bottom face
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],      # top face
    )) / 2
    # scale the template using box dimensions (dx, dy, dz)
    # boxes3d[:, None, 3:6] → (N, 1, 3)
    # result → (N, 8, 3)
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    # rotate each box’s corners around Z-axis using heading (yaw)
    # reshape to (N, 8, 3) for batch rotation
    corners3d = rotate_points_along_z(
        corners3d.view(-1, 8, 3),  # points
        boxes3d[:, 6]              # angles
    ).view(-1, 8, 3)
    # translate corners to the actual box center (x, y, z)
    corners3d += boxes3d[:, None, 0:3]
    # return in original format (NumPy or Torch)
    return corners3d.numpy() if is_numpy else corners3d

# prints the minimum and maximum values along each coordinates axis (x, y, z)
def print_data_range(data):
    # loop through each axis name along with its index
    for i, ax in enumerate(COORDINATE_AXIS): 
        # for each axis:
        # - data[:, i] selects all values along that axis
        # - .min() gives minimum value
        # - .max() gives maximum value
        # print formatted output showing range for that axis
        print(f"{ax} axis | min = {data[:, i].min()} | max = {data[:, i].max()}")

# creates a 3D scatter plot object using plotly 
def get_scatter3d_plot(x, y, z, mode='lines', marker_size=1, color=None, opacity=1, colorscale=None, **kwargs):
    # cooridnates for 3D points 
    return go.Scatter3d(
        x=x, y=y, z=z,                  
        mode=mode,                     
        # disable hover text (useful for cleaner visualization in dense plots)
        hoverinfo='skip',
        # do not show legend entry for this plot
        showlegend=False, 
        # Marker styling used when mode includes 'markers'
        marker=dict(
            size=marker_size,          # size of each point
            color=color,               # color of points can be scalar or array
            opacity=opacity,           # transparency level
            colorscale=colorscale      # optional color mapping 
        ),
        # additional Plotly parameters (flexibility for customization)
        **kwargs
    )

# creates 3D point clound visualization using plotly
def plot_pc_data3d(x, y, z, apply_color_gradient=True, color=None, marker_size=1, colorscale=None, **kwargs):
    # If enabled, compute a color gradient based on distance from origin
    # sqrt(x^2 + y^2 + z^2) gives radial distance → useful for depth visualization
    if apply_color_gradient:
        color = np.sqrt(x**2 + y**2 + z**2)
    # call helper function to generate a Scatter3D plot in 'markers' mode each 
    # point is rendered as a marker in 3D space
    return get_scatter3d_plot(
        x, y, z,
        mode='markers',            # plot points instead of lines
        color=color,               # Color per point 
        colorscale=colorscale,     # optional color mapping 
        marker_size=marker_size,   # size of each point
        **kwargs                   # additional Plotly parameters
    )

def plot_box_corners3d(box3d, color, **kwargs):
    """
    # visualizes a 3D bounding box by plotting its edges/faces using Plotly
    # box3d: (8, 3)
    # contains 8 corner points of the 3D box
    # each row is a point → [x, y, z]
    # color: color of the box edges (e.g., 'red', 'blue', or RGB values)
    # INDICES_1 to INDICES_4 define groups of corner indices each group represents 
    # one face (or loop of edges) of the 3D box
    """
    return [
        # plot first face/edge loop using points indexed by INDICES_1
        get_scatter3d_plot(
            box3d[INDICES_1, 0],  # x-coordinates
            box3d[INDICES_1, 1],  # y-coordinates
            box3d[INDICES_1, 2],  # z-coordinates
            color=color,
            **kwargs
        ),
        # plot second face
        get_scatter3d_plot(
            box3d[INDICES_2, 0],
            box3d[INDICES_2, 1],
            box3d[INDICES_2, 2],
            color=color,
            **kwargs
        ),
        # plot third face
        get_scatter3d_plot(
            box3d[INDICES_3, 0],
            box3d[INDICES_3, 1],
            box3d[INDICES_3, 2],
            color=color,
            **kwargs
        ),
        # plot fourth face
        get_scatter3d_plot(
            box3d[INDICES_4, 0],
            box3d[INDICES_4, 1],
            box3d[INDICES_4, 2],
            color=color,
            **kwargs
        ),
    ]

# visuallizes multiple 3D bouning boxes using ploty
# - boxes3d: (N, 8, 3)
# - N = number of boxes
# - each box has 8 corner points with (x, y, z) coordinates
# - box_colors = N
# - color assigned to each box one color per bounding box
def plot_bboxes_3d(boxes3d, box_colors, **kwargs):
    # initialize list to store all Plotly objects lines for each box
    boxes3d_objs = []
    # loop through each bounding box
    for obj_i in range(boxes3d.shape[0]): 
        # for each box:
        # - call plot_box_corners3d to generate its edges/faces
        # - this returns a list of Scatter3D objects
        # - extend the main list with those objects
        boxes3d_objs.extend(
            plot_box_corners3d(
                boxes3d[obj_i],          # (8, 3) corners of current box
                color=box_colors[obj_i], # color for this box
                **kwargs                 # additional Plotly settings
            )
        )
    # return list of all plotly objects for rendering
    return boxes3d_objs

# creates a complete set of 3D ploty objects for LiDAR visuallization
# - point cloud
# - ground truth bounding boxes
# - predicted bounding boxes
def get_lidar3d_plots(points, pc_kwargs={}, gt_box_corners=None, gt_box_colors=None, 
                      pred_box_corners=None, pred_box_colors=None, **kwargs):
    # initialize list to store all plot elements
    lidar3d_plots = []
    # -----------------------------
    # 1. point cloud visualization
    # -----------------------------
    # extract x, y, z from points and plot them as 3D markers pc_kwargs allows 
    # customization (color, size, etc.)
    lidar3d_plots.append(
        plot_pc_data3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            **pc_kwargs
        )
    )
    # -----------------------------
    # 2. ground truth boxes (GT)
    # -----------------------------
    # if GT boxes and colors are provided:
    # - convert each box into line plots
    # - add them to visualization
    if (gt_box_corners is not None) and (gt_box_colors is not None):
        lidar3d_plots.extend(
            plot_bboxes_3d(
                gt_box_corners,     # (N, 8, 3)
                gt_box_colors,      # (N)
                **kwargs
            )
        )
    # -----------------------------
    # 3. predicted boxes
    # -----------------------------
    # same as GT, but for model predictions
    if (pred_box_corners is not None) and (pred_box_colors is not None):
        lidar3d_plots.extend(
            plot_bboxes_3d(
                pred_box_corners,   # (M, 8, 3)
                pred_box_colors,    # (M)
                **kwargs
            )
        )
    # return all Plotly objects (to be passed into go.Figure)
    return lidar3d_plots

# comvert an RGB image (numpy array) into a base64-encoded PNG string useful for embedding images
def get_base64_string(rgb_image):
    # convert NumPy array → PIL Image object
    pil_img = Image.fromarray(rgb_image)
    # Prefix required for displaying Base64 image in web 
    prefix = "C:/Users/My-PC/Downloads/archive/val/routes_30mshortroutes_Town06_Scenario9_route1_11_23_20_29_29/rgb"
    # use an in-memory byte stream instead of saving to disk
    with BytesIO() as stream:
        # save the image into the stream in PNG format
        pil_img.save(stream, format="png")
        # encode the byte data into Base64 and convert to string
        base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
    # return the Base64 image string
    return base64_string

# creates 2D image plot using plotry
def get_image2d_plots(rgb_image):
    # convert the input RGB image (NumPy array) into a Base64-encoded string this allows Plotly to render the image directly in the figure
    return go.Image(
        source=get_base64_string(rgb_image),  # encoded image data
        # Disable hover interaction (useful for cleaner visualization)
        hoverinfo='skip'
    )