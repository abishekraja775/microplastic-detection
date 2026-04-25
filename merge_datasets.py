import os
import json
import shutil
from pathlib import Path

datasets = {
    "dataset1": {"splits": ["train", "valid", "test"]},
    "dataset2": {"splits": ["train"]},
    "dataset3": {"splits": ["train"]},
    "dataset4": {"splits": ["train", "valid"]}
}

output_dir = Path("datasets/merged")
output_splits = ["train", "valid", "test"]

class_remap = {
    "plastic": "plastic",
    "microplastic": "plastic",
    "water-microplastics": "plastic",
    "Microplastic": "plastic",
    "Plastic": "plastic",
    "mp": "plastic",
    "leaf waste": "organic",
    "sea weed": "organic",
    "seaweed": "organic",
    "organic": "organic",
    "non-plastic": "organic",
}

unified_classes = ["plastic", "organic"]
class_to_id = {name: idx + 1 for idx, name in enumerate(unified_classes)}

for split in output_splits:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

merged = {
    split: {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "plastic"},
            {"id": 2, "name": "organic"}
        ]
    }
    for split in output_splits
}

image_id_counter = 1
ann_id_counter = 1

for ds_name, ds_info in datasets.items():
    ds_path = Path(f"datasets/{ds_name}")

    for split in ds_info["splits"]:
        split_path = ds_path / split

        # find annotation file
        ann_file = split_path / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"Skipping {ds_name}/{split} — no annotations found")
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        # images are directly in split folder, not in images subfolder
        img_dir = split_path

        # build category map
        cat_map = {}
        for cat in coco["categories"]:
            unified_name = class_remap.get(cat["name"], None)
            if unified_name:
                cat_map[cat["id"]] = unified_name

        # for train-only datasets, split 80/20
        if split == "train" and ds_info["splits"] == ["train"]:
            total = len(coco["images"])
            split_idx = int(total * 0.8)
            train_img_ids = set(img["id"] for img in coco["images"][:split_idx])
            valid_img_ids = set(img["id"] for img in coco["images"][split_idx:])
        else:
            train_img_ids = set(img["id"] for img in coco["images"])
            valid_img_ids = set()

        old_to_new_id = {}

        for img_info in coco["images"]:
            src = img_dir / img_info["file_name"]
            if not src.exists():
                print(f"  Image not found: {src}")
                continue

            # determine output split
            if split == "test":
                out_split = "test"
            elif img_info["id"] in valid_img_ids:
                out_split = "valid"
            else:
                out_split = "train"

            new_filename = f"{ds_name}_{img_info['file_name']}"
            dst = output_dir / out_split / new_filename
            shutil.copy2(src, dst)

            new_img = {
                "id": image_id_counter,
                "file_name": new_filename,
                "width": img_info.get("width", 640),
                "height": img_info.get("height", 640)
            }
            merged[out_split]["images"].append(new_img)
            old_to_new_id[img_info["id"]] = (image_id_counter, out_split)
            image_id_counter += 1

        for ann in coco["annotations"]:
            if ann["image_id"] not in old_to_new_id:
                continue
            if ann["category_id"] not in cat_map:
                continue

            new_img_id, out_split = old_to_new_id[ann["image_id"]]
            unified_name = cat_map[ann["category_id"]]
            new_cat_id = class_to_id[unified_name]

            new_ann = {
                "id": ann_id_counter,
                "image_id": new_img_id,
                "category_id": new_cat_id,
                "bbox": [float(x) for x in ann["bbox"]],
                "area": ann.get("area", float(ann["bbox"][2]) * float(ann["bbox"][3])),
                "iscrowd": 0
            }
            merged[out_split]["annotations"].append(new_ann)
            ann_id_counter += 1

        print(f"Processed {ds_name}/{split}")

for split in output_splits:
    ann_out = output_dir / split / "_annotations.coco.json"
    with open(ann_out, "w") as f:
        json.dump(merged[split], f)
    print(f"{split}: {len(merged[split]['images'])} images, {len(merged[split]['annotations'])} annotations")

print("Done. Output in datasets/merged/")