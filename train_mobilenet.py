import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import os
import json
from PIL import Image

# ── Dataset ───────────────────────────────────────────────────────────────────

class MicroplasticDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


# ── Setup ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

train_dataset = MicroplasticDataset(
    img_dir="datasets/merged/train",
    ann_file="datasets/merged/train/_annotations.coco.json",
    transforms=transform
)

valid_dataset = MicroplasticDataset(
    img_dir="datasets/merged/valid",
    ann_file="datasets/merged/valid/_annotations.coco.json",
    transforms=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

print(f"Train samples: {len(train_dataset)}")
print(f"Valid samples: {len(valid_dataset)}")

# ── Model ─────────────────────────────────────────────────────────────────────

with open("datasets/merged/train/_annotations.coco.json") as f:
    ann_data = json.load(f)

num_classes = len(ann_data['categories']) + 1  # +1 for background
print(f"Number of classes (including background): {num_classes}")

model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

in_channels = [672, 480, 512, 256, 256, 128]
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head.classification_head = SSDLiteClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=num_classes,
    norm_layer=torch.nn.BatchNorm2d
)

model.to(device)
print("Model loaded and moved to GPU")

# ── Training ──────────────────────────────────────────────────────────────────

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            batch_count += 1

        except Exception as e:
            print(f"  Skipping batch: {e}")
            continue

    scheduler.step()
    avg_loss = total_loss / max(batch_count, 1)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"mobilenet_ssd_epoch{epoch+1}.pth")
        print(f"  Checkpoint saved at epoch {epoch+1}")

torch.save(model.state_dict(), "mobilenet_ssd_final.pth")
print("Training complete. Model saved as mobilenet_ssd_final.pth")