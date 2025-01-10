import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def preprocess_image(image):
    transform = F.to_tensor(image)
    transform = F.normalize(transform, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform


def decode_rle(rle):
    width, height = rle["size"]
    counts = rle["counts"]

    mask = torch.zeros(height * width, dtype=torch.long)
    pos = 0
    is_zero = True
    for count in counts:
        if not is_zero:
            mask[pos : pos + count] = 1

        pos += count
        is_zero = not is_zero

    return mask.reshape((height, width)).swapdims(0, 1)


def load_coco_annotations(annotation_file):
    """
    Loads COCO annotations from a JSON file.
    Args:
        annotation_file (str): Path to the filtered.json file.
    Returns:
        dict: Dictionary with image_id as keys and segmentation data as values.
    """
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    annotations = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(ann)

    data = [
        img | {"segmentation": annotations[image_id][0]["segmentation"]}
        for img in coco_data["images"]
        if image_id in annotations
    ]

    return data


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.data = load_coco_annotations(annotation_file)
        self.image_dir = os.path.join(root_dir, "images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_meta = self.data[idx]
        file_name = image_meta["file_name"]
        img_path = os.path.join(self.image_dir, file_name)

        image = preprocess_image(Image.open(img_path))

        mask = decode_rle(image_meta["segmentation"])

        return image, mask
