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