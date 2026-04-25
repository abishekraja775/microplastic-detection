# Microplastic Detection — MobileNetV2 + SSDLite

Object detection model for underwater microplastic detection using SSDLite with MobileNetV3 backbone.

## Setup

1. Clone the repo
2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

4. Download datasets from Roboflow and place in datasets/ folder

5. Run merge script:
   python merge_datasets.py

6. Train:
   python train_mobilenet.py

7. Detect:
   python detect.py

   ## Datasets

Download all 4 datasets from Roboflow in COCO JSON format and extract them as follows:

datasets/
    dataset1/    ← Sahana Sankar: https://universe.roboflow.com/sahana-sankar-2h6xi/microplastics-detection-in-water-ca6is
    dataset2/    ← MicroPlastic nuga5: https://universe.roboflow.com/microplastic-detection-4dcde/microplastic-nuga5-vw0rb/dataset/5
    dataset3/    ← NibbleAI: https://universe.roboflow.com/nibbleai/microplastic-dataset-7rcef
    dataset4/    ← IAM: https://universe.roboflow.com/iam/microplastics-m7mf5

For each dataset:
- Click "Use this Dataset"
- Select "Fork Dataset" into your workspace
- Download in COCO JSON format
- Extract into the corresponding folder above

After downloading run:
    python merge_datasets.py
