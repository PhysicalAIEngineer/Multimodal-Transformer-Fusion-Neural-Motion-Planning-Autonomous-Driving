# Multimodal-Transformer-Fusion-Neural-Motion-Planning-Autonomous-Driving
End-to-end autonomous driving system using multimodal RGB-LiDAR fusion with transformer-based perception and neural motion planning. Performs 3D detection, segmentation, and depth estimation while predicting future trajectories. Includes CARLA pipeline, visualization, and interpretable attention analysis.

# 🚗 Towards Interpretable End-To-End Autonomous Driving

### via Multimodal Transformer Fusion and Neural Motion Planning

---

## 🧠 Overview

This project presents a **unified end-to-end autonomous driving system** that integrates **multimodal perception, transformer-based sensor fusion, and neural motion planning** into a single architecture.

The system leverages **RGB camera + LiDAR data** to understand complex driving environments and predict **future trajectories (waypoints)** for safe navigation.

---

## 🚀 Key Features

* 🔍 **Multimodal Perception** (RGB + LiDAR)
* 🔗 **Transformer-Based Sensor Fusion (TransFuser)**
* 🧱 **3D Object Detection (BEV space)**
* 🧠 **Semantic Segmentation**
* 🌫️ **Depth Estimation**
* 🛣️ **Bird’s Eye View (BEV) Representation**
* 🎯 **Neural Motion Planning (Waypoint Prediction)**
* 📊 **Model Interpretability (CAM / Attention Maps)**
* 🎥 **Visualization + Video Generation Pipeline**

---

## 🏗️ System Architecture
![Image](https://github.com/user-attachments/assets/9687dbb6-7e3c-4efe-9a61-5ebddeb18e4a)

![Image](https://github.com/user-attachments/assets/c134c4fd-35be-4e38-a34c-d1a93aa564f3)

```text
Sensors (RGB + LiDAR)
        ↓
Feature Extraction (CNN Backbones)
        ↓
Multimodal Transformer Fusion
        ↓
Perception Heads:
   - Detection
   - Segmentation
   - Depth
        ↓
Planning Head:
   - Waypoint Prediction (GRU)
```

---

## 📂 Project Structure

```text
├── transfuser_end_to_end_autonomous_driving_pipeline.py
├── global_config_transfuser_autonomous_driving.py
├── carla_multimodal_data_collection_pipeline.py
├── carla_multimodal_dataset_split_generator.py
├── carla_e2e_multimodal_data_exploration_and_preprocessing.py
├── transfuser_model.py
├── utils/
├── assets/
│   ├── images/
│   └── videos/
├── outputs/
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* CARLA Simulator Dataset
* Multi-modal synchronized data:

  * RGB Images
  * LiDAR Point Clouds
  * Depth Maps
  * Semantic Segmentation
  * BEV Maps
  * Vehicle Measurements

---

## 🧪 Model Details

* Backbone: **RegNetY-032**
* Fusion: **Transformer-based cross-modal attention**
* Planning: **GRU-based waypoint prediction**
* Multi-task learning:
  * Detection
  * Segmentation
  * Depth
  * BEV

---

## 🎥 Results & Visualization

### 🔹 RGB Input

<img width="649" height="177" alt="Image" src="https://github.com/user-attachments/assets/1aa2930b-ec2c-4e6b-ae92-84234d87642f" />

### 🔹 Original Drivable & Lane Marking & Cropped BEV View
<img width="1451" height="369" alt="Image" src="https://github.com/user-attachments/assets/a1d19e9a-d675-4782-8f62-8ee0f70122b4" />

### 🔹 LiDAR Points Top-Down View
<img width="554" height="413" alt="Image" src="https://github.com/user-attachments/assets/1a901819-da7d-46e4-a5db-6fdf66a1b218" />

### 🔹 Bird Eye View(BEV) With Boudning Boxes
<img width="430" height="418" alt="Image" src="https://github.com/user-attachments/assets/762090d9-b13d-4641-8bfc-09a1c5feabec" />

### 🔹 Ego Vehicle Future Waypoints
<img width="578" height="455" alt="Image" src="https://github.com/user-attachments/assets/9115a5a5-561c-4a80-bf03-80530dad1811" />

### 🔹 3D LiDAR Point Cloud Visualization Projected Into Bird Eye View (BEV)
<img width="483" height="463" alt="Image" src="https://github.com/user-attachments/assets/c67eee30-10e6-4215-90cb-088790d9359a" />

### 🔹 Class Activation Maps (CAMs) Across Four Layers (Layer1–Layer4) Of an Image Encoder
<img width="960" height="160" alt="Image" src="https://github.com/user-attachments/assets/89cd65e4-dc01-40c1-ac8f-6c2c71e931f4" />

### 🔹 3D LiDAR Point Cloud Visualization Enhanced With Predicted Bounding Boxes From the TransFuser Model
<img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/3b5299d2-61dc-4158-914f-b02287f7c0f2" />

### 🔹 Trajectory Visualization Of Predicted Waypoints For An Autonomous Vehicle
<img width="681" height="327" alt="Image" src="https://github.com/user-attachments/assets/9e3709f8-6ad4-4493-bb61-d1057fabb830" />

### 🔹Visualization Video Output
https://github.com/user-attachments/assets/48e7388d-ccb2-4684-b112-12f558a0da2a

---

## 🧠 Interpretability

The model includes:

* Class Activation Maps (CAM)
* Attention visualization across modalities

👉 Helps understand **where the model is focusing**

---

## ⚙️ Installation

```bash
git clone https://github.com/PhysicalAIEngineer/Multimodal-Transformer-Fusion-Neural-Motion-Planning-Autonomous-Driving.git
cd Multimodal-Transformer-Fusion-Neural-Motion-Planning-Autonomous-Driving

pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python transfuser_end_to_end_autonomous_driving_pipeline.py
```

---

## 📦 Pretrained Model

⚠️ Large file not stored in repo

👉 Download here:
[Download TransFuser Model](https://your-link-here)

File:

```
transfuser_regnet032_seed1_39.pth
```

---

## 📈 Applications

* Autonomous Vehicles
* Robotics & UAV Navigation
* Real-time Perception Systems
* Physical AI Systems

---

## 🧠 Future Work

* Real-world dataset adaptation
* Temporal modeling (video-based learning)
* Reinforcement learning for planning
* Integration with ROS2 and real robots

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📜 License

This project is for research and educational purposes.

---

## 👨‍💻 Author

**Chetan Sonigara**
Perception Engineer | Physical AI & Autonomous Systems

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
