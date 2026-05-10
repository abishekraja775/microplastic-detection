# Microplastic Detection Project - Complete Documentation

**Project Type:** Object Detection using Deep Learning  
**Framework:** PyTorch with TorchVision  
**Model Architecture:** SSDLite320 with MobileNetV3 Large Backbone  
**Task:** Underwater Microplastic Detection  
**Date Created:** 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Directory Descriptions](#directory-descriptions)
4. [File Descriptions](#file-descriptions)
5. [Setup Instructions](#setup-instructions)
6. [Usage Workflow](#usage-workflow)
7. [Datasets](#datasets)
8. [Model Files](#model-files)
9. [Results](#results)

---

## Project Overview

This project implements an object detection system for identifying microplastics in underwater environments using a lightweight neural network. It uses **SSDLite320 with MobileNetV3 Large** backbone, making it efficient for deployment on resource-constrained devices.

### Key Features

- **Lightweight Model**: MobileNetV3 backbone ensures fast inference
- **Two-Class Detection**: Detects both "plastic" and "organic" materials
- **Multi-Dataset Support**: Merges multiple annotated datasets with different labeling conventions
- **PyTorch Implementation**: Uses torchvision detection models
- **COCO Format**: Works with COCO annotation format
- **GPU Support**: CUDA acceleration available

### Supported Classes

1. **Plastic** - All microplastic variants
2. **Organic** - Non-plastic materials (seaweed, leaves, etc.)

---

## Project Structure

```
microplastic_detection/
├── README.md                          # Original project README
├── PROJECT_DOCUMENTATION.md           # This comprehensive documentation
├── requirements.txt                   # Python dependencies
├── detect.py                          # Inference script
├── train_mobilenet.py                 # Training script
├── merge_datasets.py                  # Dataset merging utility
├── download_dataset.py                # Dataset download script (placeholder)
│
├── Model Files (Checkpoints)
├── mobilenet_ssd_epoch10.pth          # Model checkpoint at epoch 10
├── mobilenet_ssd_epoch20.pth          # Model checkpoint at epoch 20
├── mobilenet_ssd_epoch30.pth          # Model checkpoint at epoch 30
├── mobilenet_ssd_final.pth            # Final trained model (production)
│
├── datasets/                          # Datasets directory
│   ├── dataset1/                      # Roboflow Dataset 1
│   │   ├── README.dataset.txt         # Dataset metadata
│   │   ├── README.roboflow.txt        # Roboflow information
│   │   ├── train/
│   │   │   └── _annotations.coco.json # COCO format annotations
│   │   ├── valid/
│   │   │   └── _annotations.coco.json
│   │   └── test/
│   │       └── _annotations.coco.json
│   │
│   ├── dataset2/                      # Roboflow Dataset 2
│   │   ├── README.roboflow.txt
│   │   └── train/
│   │       └── _annotations.coco.json
│   │
│   ├── dataset3/                      # Roboflow Dataset 3
│   │   ├── README.roboflow.txt
│   │   └── train/
│   │       └── _annotations.coco.json
│   │
│   ├── dataset4/                      # Roboflow Dataset 4
│   │   ├── README.dataset.txt
│   │   ├── README.roboflow.txt
│   │   ├── train/
│   │   │   └── _annotations.coco.json
│   │   └── valid/
│   │       └── _annotations.coco.json
│   │
│   └── merged/                        # Unified merged dataset
│       ├── train/
│       │   ├── _annotations.coco.json # Merged training annotations
│       │   └── images/                # Training images
│       ├── valid/
│       │   ├── _annotations.coco.json # Validation annotations
│       │   └── images/                # Validation images
│       └── test/
│           ├── _annotations.coco.json # Test annotations
│           └── images/                # Test images
│
├── results/                           # Inference results
│   ├── image.png                      # Sample output visualization
│   ├── 21_jpg.rf.bEHEWumM42ouvF5FZNLX.jpg
│   ├── shutterstock_1120655291-800x533_jpg.rf.81fb830de95633b8e6d0cf4134d4a8af.jpg
│   ├── th_jpg.rf.276e505846ede6f7f188e19b182a4021.jpg
│   ├── th_jpg.rf.516859e3e7d5244b2f7e9ab8f68cd7bd.jpg
│   ├── th_jpg.rf.9cc95a396f20907e38bcdc405776a4ae.jpg
│   └── th_jpg.rf.f29bee7a84666179bcd548eb280095e1.jpg
│
├── venv/                              # Python virtual environment
└── .git/                              # Git version control
    └── .gitignore                     # Git ignore rules
```

---

## Directory Descriptions

### Root Directory
Contains main project files including training/inference scripts, model checkpoints, and configuration files.

### datasets/
**Purpose:** Stores all training data organized by dataset source and split (train/valid/test)

#### Individual Datasets (dataset1-4)
- **Source:** Roboflow downloads
- **Format:** COCO JSON annotations with corresponding images
- **Structure:** Each dataset may contain train, valid, and/or test splits
- **Annotations:** COCO format JSON files containing bounding box and category information

#### datasets/merged/
- **Purpose:** Unified dataset combining all individual datasets
- **Created by:** merge_datasets.py script
- **Structure:**
  - `train/`: Training split with merged annotations and images
  - `valid/`: Validation split with merged annotations and images
  - `test/`: Test split with merged annotations and images
  - `_annotations.coco.json`: Unified COCO annotations for each split
  - `images/`: Directory containing actual image files for each split
- **Class Mapping:** Standardizes all class names to two categories:
  - "plastic": plastic, microplastic, water-microplastics, Microplastic, Plastic, mp
  - "organic": leaf waste, sea weed, seaweed, non-plastic, organic

### results/
**Purpose:** Stores inference outputs from the detect.py script

**Contents:**
- Detection visualizations (images with bounding boxes drawn)
- Annotated images showing detected objects and confidence scores
- Used for validating model performance on test images

### venv/
**Purpose:** Python virtual environment with isolated dependencies

**Function:** Keeps project dependencies separate from system Python installation

---

## File Descriptions

### Python Scripts

#### train_mobilenet.py
**Purpose:** Main training script for the object detection model

**Functionality:**
- Implements MicroplasticDataset class for loading COCO-format data
- Uses SSDLite320 with MobileNetV3 Large backbone
- Handles data augmentation via transforms
- Implements training loop with optimizer and learning rate scheduling
- Saves model checkpoints at regular intervals
- Evaluates on validation set
- Supports GPU acceleration via CUDA

**Key Components:**
- `MicroplasticDataset`: Custom PyTorch Dataset class
- Data transforms: Resize to 320x320, ToTensor conversion
- Model: ssdlite320_mobilenet_v3_large with custom classification head
- Training configuration: Epochs, batch size, learning rate

#### detect.py
**Purpose:** Inference script for running detection on images

**Functionality:**
- Loads trained model from checkpoint (mobilenet_ssd_final.pth)
- Reads class names from merged dataset annotations
- Processes test images
- Generates predictions with bounding boxes
- Saves annotated images to results/ directory
- Displays confidence scores and class labels

**Key Components:**
- Model loading with exact training configuration
- Image preprocessing: Resize to 320x320, Tensor conversion
- Inference loop over test dataset
- Visualization with PIL ImageDraw
- Output saving to results directory

#### merge_datasets.py
**Purpose:** Unifies multiple datasets into a single training-ready dataset

**Functionality:**
- Combines dataset1, dataset2, dataset3, and dataset4
- Handles different train/valid/test split configurations
- Remaps class labels to unified naming scheme
- Consolidates COCO annotations
- Copies images to merged directory structure
- Generates unified _annotations.coco.json files

**Class Remapping:**
```
"plastic" -> "plastic"
"microplastic" -> "plastic"
"water-microplastics" -> "plastic"
"Microplastic" -> "plastic"
"Plastic" -> "plastic"
"mp" -> "plastic"
"leaf waste" -> "organic"
"sea weed" -> "organic"
"seaweed" -> "organic"
"organic" -> "organic"
"non-plastic" -> "organic"
```

**Output Structure:**
- Creates datasets/merged/ with train/valid/test subdirectories
- Each split contains:
  - `_annotations.coco.json`: Unified COCO format annotations
  - `images/`: Directory with image files
- Unified class IDs: 1="plastic", 2="organic"

#### download_dataset.py
**Purpose:** Placeholder for dataset download automation

**Status:** Currently contains only whitespace (placeholder file)

**Intended Use:** Would automate downloading datasets from Roboflow using API

### Configuration Files

#### requirements.txt
**Purpose:** Lists Python package dependencies for the project

**Key Dependencies:**
- torch: PyTorch framework
- torchvision: Computer vision models and utilities
- pycocotools: COCO dataset tools
- Pillow (PIL): Image processing
- numpy: Numerical operations

#### README.md
**Purpose:** Original project README with quick start instructions

**Contents:**
- Project title and description
- Setup steps
- Installation instructions
- Training and inference workflow

### Model Checkpoint Files

#### mobilenet_ssd_epoch10.pth
- Model state saved after 10 training epochs
- Checkpoint format: PyTorch .pth file
- Contains weights and biases of trained model

#### mobilenet_ssd_epoch20.pth
- Model state saved after 20 training epochs
- Intermediate checkpoint for monitoring training progress

#### mobilenet_ssd_epoch30.pth
- Model state saved after 30 training epochs
- Late-stage checkpoint

#### mobilenet_ssd_final.pth
- Final trained model after all training epochs
- Used for inference in detect.py
- Contains optimized weights from complete training

**File Format:** PyTorch state dictionary (.pth)

---

## Setup Instructions

### Prerequisites
- Python 3.7+
- CUDA 11.8 (recommended for GPU acceleration)
- pip package manager

### Step 1: Clone Repository
```bash
cd c:\mypython
git clone <repository-url>
cd microplastic_detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Prepare Datasets
1. Download datasets from Roboflow
2. Place each dataset in `datasets/dataset1/`, `datasets/dataset2/`, etc.
3. Each dataset should have COCO format annotations in `_annotations.coco.json` files

### Step 5: Merge Datasets
```bash
python merge_datasets.py
```
This creates `datasets/merged/` with unified train/valid/test splits

### Step 6: Train Model
```bash
python train_mobilenet.py
```
Training output:
- Saves checkpoints: `mobilenet_ssd_epoch10.pth`, `mobilenet_ssd_epoch20.pth`, etc.
- Final model: `mobilenet_ssd_final.pth`

### Step 7: Run Inference
```bash
python detect.py
```
Outputs annotated images to `results/` directory

---

## Usage Workflow

### Complete Workflow (First-time setup)
```
1. Clone repository
2. Create and activate virtual environment
3. Install dependencies
4. Download datasets from Roboflow
5. Run: python merge_datasets.py
6. Run: python train_mobilenet.py
7. Run: python detect.py
```

### Inference on Existing Model
```
1. Activate virtual environment
2. Place test images in datasets/merged/test/images/
3. Run: python detect.py
4. Check results/ directory for annotated output images
```

### Retraining with Different Data
```
1. Place new datasets in datasets/dataset1/, dataset2/, etc.
2. Run: python merge_datasets.py
3. Run: python train_mobilenet.py
4. Use new model checkpoint in detect.py
```

---

## Datasets

### Dataset Sources
All datasets sourced from **Roboflow** platform for computer vision datasets

### Dataset Organization

| Dataset | Splits Included | Purpose | Classes |
|---------|-----------------|---------|---------|
| dataset1 | train, valid, test | Primary training data | Multiple (remapped) |
| dataset2 | train | Training augmentation | Multiple (remapped) |
| dataset3 | train | Training augmentation | Multiple (remapped) |
| dataset4 | train, valid | Additional training data | Multiple (remapped) |

### Merged Dataset

**Location:** `datasets/merged/`

**Statistics:**
- **Train Split:** Consolidated from all dataset train sets
- **Valid Split:** Consolidated from dataset1 and dataset4 valid sets
- **Test Split:** From dataset1 test set
- **Classes:** 2 unified classes (plastic, organic)
- **Annotation Format:** COCO JSON

**Class Distribution After Merge:**
- Class ID 1: "plastic" (all plastic variants)
- Class ID 2: "organic" (non-plastic materials)

### COCO Annotation Format

Each `_annotations.coco.json` file contains:
```json
{
  "images": [
    {
      "id": image_id,
      "file_name": "image_filename.jpg",
      "width": image_width,
      "height": image_height
    }
  ],
  "annotations": [
    {
      "id": annotation_id,
      "image_id": image_id,
      "category_id": class_id,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "plastic"
    },
    {
      "id": 2,
      "name": "organic"
    }
  ]
}
```

---

## Model Files

### Model Architecture

**Base Model:** SSDLite320 (Single Shot Detection Lite with 320x320 input)  
**Backbone:** MobileNetV3 Large (Lightweight efficient CNN)  
**Framework:** PyTorch with TorchVision

**Advantages:**
- Small model size (suitable for edge devices)
- Fast inference speed
- Maintains good accuracy
- Pre-trained weights available

### Model Configuration

**Input Size:** 320 × 320 pixels  
**Number of Classes:** 2 (plastic, organic)  
**Output:** Bounding boxes with class predictions and confidence scores

### Checkpoint Information

| Checkpoint | Epochs | Purpose | Size |
|-----------|--------|---------|------|
| mobilenet_ssd_epoch10.pth | 10 | Early training checkpoint | ~50MB |
| mobilenet_ssd_epoch20.pth | 20 | Mid-training checkpoint | ~50MB |
| mobilenet_ssd_epoch30.pth | 30 | Late-training checkpoint | ~50MB |
| mobilenet_ssd_final.pth | Complete | Production model | ~50MB |

### Loading a Checkpoint

```python
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

# Load checkpoint
checkpoint = torch.load("mobilenet_ssd_final.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
```

---

## Results

### Output Directory: `results/`

**Purpose:** Contains inference outputs and validation visualizations

### Sample Outputs

The results directory contains annotated images from inference:

| File | Type | Purpose |
|------|------|---------|
| image.png | PNG image | Sample detection visualization |
| 21_jpg.rf.*.jpg | JPEG image | Annotated test image with detections |
| shutterstock_*.jpg | JPEG image | Annotated test image with detections |
| th_jpg.rf.*.jpg | JPEG image | Multiple test images with bounding boxes |

### Output Format

Each result image contains:
- Original image
- Detected bounding boxes (red/blue rectangles)
- Class labels (e.g., "plastic", "organic")
- Confidence scores (0-100%)

### Interpreting Results

**Example Output Visualization:**
```
Original Image
      ↓
[Detected bounding box 1 - "plastic" 95%]
[Detected bounding box 2 - "organic" 87%]
[Detected bounding box 3 - "plastic" 92%]
      ↓
Annotated Image saved to results/
```

---

## Quick Reference Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Data Processing
```bash
# Merge all datasets into unified dataset
python merge_datasets.py
```

### Training
```bash
# Train the model
python train_mobilenet.py
```

### Inference
```bash
# Run detection on test images
python detect.py
```

---

## Notes and Tips

### GPU Acceleration
- CUDA 11.8 is recommended for GPU support
- Falls back to CPU if CUDA not available
- GPU significantly speeds up training and inference

### Model Selection
- Use `mobilenet_ssd_final.pth` for production inference
- Intermediate checkpoints useful for studying training progress
- Can load any checkpoint and continue training

### Dataset Best Practices
- Ensure all images are readable and in correct format
- Verify COCO JSON format is correct before training
- Check class ID mappings during merge process
- Balanced dataset improves model performance

### Performance Optimization
- Adjust batch size based on GPU memory
- Use data augmentation for better generalization
- Monitor validation loss to detect overfitting
- Consider class weights if classes are imbalanced

### Troubleshooting
- If CUDA out of memory: Reduce batch size
- If training is slow: Ensure GPU is being used
- If low accuracy: Check dataset quality and class balance
- If inference fails: Verify image format and model checkpoint path

---

## Project Statistics

- **Total Scripts:** 4 Python files
- **Model Checkpoints:** 4 saved states
- **Datasets:** 4 source + 1 merged
- **Classes:** 2 (plastic, organic)
- **Results:** 7 sample outputs
- **Framework:** PyTorch + TorchVision
- **Model Size:** ~50MB per checkpoint

---

## Version Information

- **Python:** 3.7+
- **PyTorch:** Latest (CUDA 11.8 support)
- **TorchVision:** Matches PyTorch version
- **Project Date:** 2026

---

*Documentation generated for Microplastic Detection Project - SSDLite with MobileNetV3*
