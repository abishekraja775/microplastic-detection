# 🌊 UV-Fluorescence and CNN-Based Autonomous ROV System for Underwater Microplastic Detection and Targeted Retrieval

**SRM MTS TechSurge 2026** — March 16, 2026  
**Institution:** SRM Institute of Science and Technology  
**Event:** Marine Technology Society (MTS) TechSurge — The Underwater Robotics

---

## 👥 Team Members

| Name |
|---|
| Abishek Raja A |
| Lakshman Aadithya R K |
| Sree Sowmi A |

---

## 📌 Project Overview

Microplastic pollution is one of the most critical and underaddressed threats to marine ecosystems. Particles smaller than 5mm are virtually undetectable by conventional underwater imaging systems and cannot be retrieved by traditional collection methods.

This project proposes and demonstrates an **autonomous ROV (Remotely Operated Vehicle)** system that combines:

- **UV Fluorescence Sensing** — for initial microplastic detection
- **CNN-Based Computer Vision** — for particle classification, localization, and size estimation
- **Electrochemical Sensing + TinyML** — for polymer-type identification
- **Robotic Manipulator** — for physical retrieval of recoverable particles
- **Acoustic Communication** — for reporting non-recoverable particles to surface buoys

This is the first proposed single-platform solution combining **real-time in-situ detection AND physical retrieval** autonomously underwater.

---

## 🧠 ML Proof of Concept

As part of this project, we trained and deployed a real object detection model demonstrating the computer vision layer of the system.

### Model
- **Architecture:** SSDLite with MobileNetV3
- **Framework:** PyTorch 2.8
- **Task:** Detect and classify microplastic particles in images
- **Classes:** `plastic`, `organic`
- **Input size:** 320×320
- **Hardware:** NVIDIA RTX 2050, CUDA 11.8

### Training Results

| Metric | Value |
|---|---|
| Training Images | 1,561 |
| Validation Images | 273 |
| Epochs | 30 |
| Initial Loss | 38.47 |
| Final Loss | 7.93 |
| Optimizer | Adam (lr=1e-4) |
| Batch Size | 8 |

Loss reduced by ~80% across 30 epochs — clear evidence of successful learning.

---

## 🔬 System Architecture

```
UV LED (365nm) illuminates water
        ↓
Fluorescence detected by photodiode
        ↓ triggers camera
RGB Camera captures image frame
        ↓
[CNN — SSDLite MobileNetV3]
Detects particle → estimates size
        ↓
[TinyML — Random Forest]
Electrochemical sensor → polymer type (PE / PP / PET / Nylon)
        ↓
Fusion Decision Layer
        ↓
Size > 2mm  ──→  Manipulator triggered → particle retrieved
Size < 2mm  ──→  Acoustic modem → surface buoy → cloud log
```

---

## 🔧 Hardware Components (Proposed System)

### Detection Sensors
| Sensor | Purpose |
|---|---|
| UV LED (365nm) + Filtered Photodiode | Fluorescence-based plastic presence detection |
| Carbon Microwire Electrode (3-electrode cell) | Electrochemical polymer-type identification |
| RGB Camera (wide angle, low-light) | Visual particle detection and localization |
| Turbidity Sensor | Water clarity — gates vision reliability |

### Environmental Sensors
| Sensor | Purpose |
|---|---|
| Temperature Sensor (DS18B20) | Electrochemical baseline correction |
| Conductivity Sensor | Salinity correction |
| pH Sensor | Oxygen reduction rate correction |
| Depth / Pressure Sensor (MS5837) | AUV depth tracking and metadata logging |

### Navigation and Communication
| Component | Purpose |
|---|---|
| IMU (Accelerometer + Gyroscope) | AUV orientation and stability |
| DVL (Doppler Velocity Log) | Underwater positioning |
| Acoustic Modem (e.g. EvoLogics S2CR) | Long-range data transmission to surface buoy |
| Optical Modem (Blue-Green LED) | Short-range high-bandwidth data offload |
| GPS Module (surface buoy) | Particle location tagging |

### Processing Hardware
| Component | Role |
|---|---|
| ESP32 Microcontroller | TinyML inference — electrochemical classification |
| Raspberry Pi 4 / Jetson Nano | CNN inference, fusion decision layer |
| Arduino (manipulator controller) | Gripper actuation and feedback |

### Retrieval System
| Component | Purpose |
|---|---|
| Robotic Manipulator Arm | Physical retrieval of particles > 2mm |
| Proximity Sensor (IR/Ultrasonic) | Guides gripper to target particle |
| Sample Collection Chamber | Stores retrieved particles onboard |

---

## 🤖 ML Models in the System

### Model 1 — TinyML Electrochemical Classifier
- **Algorithm:** Random Forest (100 trees)
- **Input:** Electrochemical signal features — spike amplitude, duration, area under curve, fluorescence intensity, temperature, conductivity, pH
- **Output:** Polymer type — PE / PP / PET / Nylon / Unknown
- **Runs on:** ESP32 microcontroller (~50KB model size)

### Model 2 — Computer Vision CNN
- **Architecture:** SSDLite + MobileNetV3 Large
- **Input:** 320×320 RGB camera frame
- **Output:** Bounding box, class label, confidence score, particle size estimate
- **Runs on:** Raspberry Pi 4 / Jetson Nano (~1.5MB after INT8 quantization)

### Model 3 — Fusion Decision Layer
- **Type:** Rule-based deterministic logic (not ML)
- **Input:** Outputs from Model 1 + Model 2 + depth + turbidity
- **Output:** Trigger manipulator / Send acoustic report / Log only

---

## 📦 Datasets Used

Download all 4 datasets from Roboflow in **COCO JSON** format and extract as follows:

```
datasets/
    dataset1/    ← Sahana Sankar — Microplastics Detection in Water
    dataset2/    ← MicroPlastic Detection (nuga5)
    dataset3/    ← NibbleAI Microplastic Dataset
    dataset4/    ← IAM MicroPlastics
```

| Dataset | Link |
|---|---|
| Sahana Sankar | https://universe.roboflow.com/sahana-sankar-2h6xi/microplastics-detection-in-water-ca6is |
| MicroPlastic nuga5 | https://universe.roboflow.com/microplastic-detection-4dcde/microplastic-nuga5-vw0rb/dataset/5 |
| NibbleAI | https://universe.roboflow.com/nibbleai/microplastic-dataset-7rcef |
| IAM | https://universe.roboflow.com/iam/microplastics-m7mf5 |

For each: click **Use this Dataset** → **Fork Dataset** → download in **COCO JSON** format → extract into the corresponding folder.

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.9
- NVIDIA GPU with CUDA 11.8 or 12.1
- Git

### Step 1 — Clone the repo
```bash
git clone https://github.com/abishekraja775/microplastic-detection.git
cd microplastic-detection
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pycocotools
```

### Step 4 — Download and extract the 4 datasets into datasets/ folder

### Step 5 — Merge datasets
```bash
python merge_datasets.py
```

Expected output:
```
train: 1561 images, 7710 annotations
valid: 273 images, 1855 annotations
```

### Step 6 — Train the model
```bash
python train_mobilenet.py
```
Trains for 30 epochs. Model saved as `mobilenet_ssd_final.pth`.

### Step 7 — Run detection
Place any image in `datasets/merged/test/` and run:
```bash
python detect.py
```
Results with bounding boxes saved to `results/`.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.9 |
| Deep Learning | PyTorch 2.8 |
| Model | SSDLite + MobileNetV3 Large |
| Dataset Tools | Roboflow, pycocotools |
| Image Processing | Pillow, OpenCV |
| GPU | CUDA 11.8, NVIDIA RTX 2050 |
| Environment | Python venv, VSCode |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
microplastic-detection/
    datasets/
        dataset1/
        dataset2/
        dataset3/
        dataset4/
        merged/
            train/
            valid/
            test/
    results/
    merge_datasets.py       ← merges and unifies all 4 datasets
    train_mobilenet.py      ← trains SSDLite MobileNetV3 model
    detect.py               ← runs inference and saves result images
    requirements.txt
    README.md
```

---

## 📚 References

1. Benjamin O. Aves et al., "First evidence of microplastics in Antarctic snow," The Cryosphere, 2022.
2. Prata J.C. et al., "Methods for sampling and detection of microplastics in water and sediment," TrAC Trends in Analytical Chemistry, vol. 110, 2019.
3. Fulton E. et al., "Marine debris detection with deep learning," IEEE Robotics and Automation Letters, vol. 4, 2019.
4. Howard A. et al., "MobileNets: Efficient CNNs for mobile vision applications," arXiv:1704.04861, 2017.
5. Zhang Y. et al., "Hyperspectral imaging for microplastic identification," Journal of Hazardous Materials, vol. 384, 2020.
6. Leslie H.A. et al., "Discovery and quantification of plastic particle pollution in human blood," Environment International, vol. 163, 2022.

---

## 📄 License

This project was developed for academic and competition purposes under SRM Institute of Science and Technology.  
Dataset licenses follow their respective Roboflow terms (CC BY 4.0).
