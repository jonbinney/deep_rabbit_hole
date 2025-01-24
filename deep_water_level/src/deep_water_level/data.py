import functools
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

import mlflow
from deep_water_level.model import ModelNames


class WaterDataset(Dataset):
    def __init__(
        self, annotations_file: Path, images_dir: Path, model_name: ModelNames, transforms=None, normalize_output=False
    ):
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.normalize_output = normalize_output
        self.model_name = model_name
        self.data = self.load_annotations()

    def load_annotations(self):
        suffix = self.annotations_file.suffix
        if suffix == ".json":
            if self.model_name == "DeepLabV3":
                return self.load_coco_for_segmentation()
            return self.load_coco()
        elif suffix == ".csv":
            return self.load_csv()
        else:
            raise ValueError(f"Unsupported annotations file format: {self.annotations_file}")

    def load_coco(self):
        with self.annotations_file.open("r") as f:
            annotations = json.load(f)
        data = []
        # Create a map of image_id to image - IMPORTANT: image_id might not match index in the images array
        images = annotations.get("images", [])
        images_by_id = {}
        for image in images:
            images_by_id[image["id"]] = image
        # Go through annotations to get image_path and depth for each annotation
        for annotation in annotations["annotations"]:
            image_id = annotation["image_id"]
            image_file = images_by_id.get(image_id, {}).get("file_name", None)
            if image_file is not None:
                image_path = os.path.join(self.images_dir, image_file)
                depth = annotation.get("attributes", {})["depth"]
                if self.normalize_output:
                    # See https://docs.google.com/document/d/1_h3u6VsC2pquxDMW0ZL5JwdQC6ZxJw4-CwmKrUR303M/edit?tab=t.0#heading=h.9uh6m7e2o2to
                    depth = (depth - 7.5) / 7.5
                # NOTE: The second item [depth] is the lable. It's an array because it could
                # include many labels
                data.append((image_path, [depth]))
            else:
                print(f"WARN: No image found for annotation. image_id: {image_id}")
        return data

    def load_coco_for_segmentation(self):
        with open(self.annotations_file, "r") as f:
            coco_data = json.load(f)

        annotations = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(ann)

        data = [
            (
                str(Path(self.images_dir) / img["file_name"]),
                img | {"segmentation": annotations[image_id][0]["segmentation"]},
            )
            for img in coco_data["images"]
            if image_id in annotations
        ]

        return data

    def load_csv(self):
        def parse_line(line):
            fields = line.split(",")
            image_path = str(Path(self.images_dir) / fields[0])
            depth = float(fields[2])
            x0, y0, x1, y1 = None, None, None, None

            if len(fields) > 7 and fields[4] != "None":
                x0, y0, x1, y1 = float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])

            if self.model_name == "BasicCnnRegressionWaterLine":
                return (image_path, [depth, x0, y0, x1, y1])

            return (image_path, [depth])

        with open(self.annotations_file, "r") as f:
            lines = f.readlines()
        data = [parse_line(line) for line in lines]

        # Remove any data points with None values
        data = [d for d in data if (None not in d[1])]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.model_name == "DeepLabV3":
            label = self.decode_rle(label["segmentation"])
        else:
            label = torch.tensor(label, dtype=torch.float32)
        return image, label, image_path

    def decode_rle(self, rle):
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


class JitterCrop(torch.nn.Module):
    """
    Module that implements a random jitter of the crop box.

    Args:
        crop_box (list): [top, left, height, width]
        jitter (list): [H, W] How much to jitter the crop box
    """

    def __init__(self, crop_box, jitter=[10, 10]):
        super().__init__()
        self.crop_box = crop_box
        self.jitter = jitter

    def forward(self, img):
        horiz = random.randint(-self.jitter[1], self.jitter[1])
        vert = random.randint(-self.jitter[0], self.jitter[0])
        return v2.functional.crop(
            img, self.crop_box[0] + vert, self.crop_box[1] + horiz, self.crop_box[2], self.crop_box[3]
        )


def get_transforms(
    crop_box=None,
    equalization: bool = True,
    is_training: bool = True,
    crop_box_jitter: List[int] | None = None,
    random_rotation_degrees: int = 0,
    color_jitter: float = 0.0,
):
    transforms_array = []

    # TODO:
    # - Make most effective use of dtypes

    if crop_box is not None:
        if is_training and crop_box_jitter is not None:
            transforms_array.append(JitterCrop(crop_box, crop_box_jitter))
        else:
            top, left, height, width = crop_box
            transforms_array.append(
                functools.partial(v2.functional.crop, top=top, left=left, height=height, width=width)
            )

    if equalization:
        # Apply histogram equalization to each image
        # IMPORTANT: This works with input in the range [0, 255]
        transforms_array.append(v2.functional.equalize)

    transforms_array.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # convert to float32 and normalize to 0, 1
            # TODO(adamantivm) Enable normalization. Make it work with equalization, which requires
            # moving equalization to later and including the necessary dtype conversions back and forth
            # v2.Normalize(mean=[0.10391959, 0.1423446, 0.13618265], std=[0.05138122, 0.04945769, 0.045432]),
        ]
    )

    # Only add data augmentation transforms for training
    if is_training:
        # Add data augmentation
        transforms_array.extend(
            [
                v2.ColorJitter(
                    brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=color_jitter
                ),
                v2.RandomRotation(degrees=random_rotation_degrees),
            ]
        )

    # Pre-calculated mean and std values for water_train_set4:
    # - Without cropping:
    # Mean: [0.18653404 0.20722116 0.19524723], std: [0.15864415 0.13673791 0.12532181]
    # - With Binney cropping:
    # Mean: [0.10391959 0.1423446  0.13618265], std: [0.05138122 0.04945769 0.045432  ]

    return v2.Compose(transforms_array)


def get_data_loaders(
    images_dir: Path,
    annotations_file: Path,
    model_name: ModelNames,
    batch_size: int = 32,
    train_test_split: Tuple[float, float] = (0.8, 0.2),
    normalize_output: bool = False,
    crop_box=None,  # [top, left, height, width]
    equalization: bool = True,
    crop_box_jitter: List[int] = None,
    random_rotation_degrees: int = 0,
    color_jitter: float = 0.0,
):
    transforms = get_transforms(
        crop_box=crop_box,
        equalization=equalization,
        crop_box_jitter=crop_box_jitter,
        random_rotation_degrees=random_rotation_degrees,
        color_jitter=color_jitter,
    )

    # Load dataset from directory
    dataset = WaterDataset(
        annotations_file,
        images_dir,
        model_name=model_name,
        transforms=transforms,
        normalize_output=normalize_output,
    )

    # Split dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, train_test_split)
    train_files = [dataset.data[i][0] for i in train_dataset.indices]
    test_files = [dataset.data[i][0] for i in train_dataset.indices]
    mlflow.log_param("train_files", train_files)
    mlflow.log_param("test_files", test_files)

    mlflow.log_param("transforms", repr(transforms))
    mlflow.log_param("transforms", transforms)

    # Create PyTorch DataLoaders for train and test splits
    return (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        ),
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        ),
    )


def get_data_loader(
    images_dir: Path,
    annotations_file: Path,
    model_name: ModelNames,
    batch_size: int = 32,
    shuffle: bool = True,
    normalize_output: bool = False,
    crop_box=None,  # [top, left, height, width]
    equalization: bool = True,
    crop_box_jitter: List[int] = None,
    random_rotation_degrees: int = 0,
    color_jitter: float = 0.0,
):
    transforms = get_transforms(
        crop_box=crop_box,
        equalization=equalization,
        crop_box_jitter=crop_box_jitter,
        random_rotation_degrees=random_rotation_degrees,
        color_jitter=color_jitter,
    )

    # Load dataset from directory
    dataset = WaterDataset(
        annotations_file,
        images_dir,
        model_name,
        transforms=transforms,
        normalize_output=normalize_output,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
