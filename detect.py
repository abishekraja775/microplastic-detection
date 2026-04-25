import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision import transforms
from PIL import Image, ImageDraw
import json
import os

# ── Load class names ──────────────────────────────────────────────────────────

with open("datasets/merged/train/_annotations.coco.json") as f:
    ann_data = json.load(f)

class_names = {cat['id']: cat['name'] for cat in ann_data['categories']}
num_classes = len(ann_data['categories']) + 1

# ── Rebuild model with EXACT same config as training ─────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load with pretrained weights first to get correct backbone size
model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

# then replace classification head exactly as in training
in_channels = [672, 480, 512, 256, 256, 128]
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head.classification_head = SSDLiteClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=num_classes,
    norm_layer=torch.nn.BatchNorm2d
)

# load saved weights
checkpoint = torch.load("mobilenet_ssd_final.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("Model loaded successfully")

# ── Inference ─────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

test_img_dir = "datasets/merged/test"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(test_img_dir)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))
               and not f.endswith('.json')]

print(f"Running inference on {len(image_files)} test images...")

for img_file in image_files:
    img_path = os.path.join(test_img_dir, img_file)
    original = Image.open(img_path).convert("RGB")
    orig_w, orig_h = original.size

    tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(tensor)

    pred = predictions[0]
    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()

    threshold = 0.3
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    scale_x = orig_w / 320
    scale_y = orig_h / 320

    draw = ImageDraw.Draw(original)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        x1 *= scale_x; x2 *= scale_x
        y1 *= scale_y; y2 *= scale_y

        class_name = class_names.get(label.item(), "unknown")
        label_text = f"{class_name} {score:.2f}"
        color = "lime" if class_name == "plastic" else "orange"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1 - 16, x1 + len(label_text) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label_text, fill="black")

    output_path = os.path.join(output_dir, img_file)
    original.save(output_path)
    print(f"  {img_file} — {len(boxes)} detections")

print(f"\nDone. Open results folder for the predictions.")