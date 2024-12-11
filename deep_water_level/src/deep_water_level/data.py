import functools
import json
import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

import mlflow


class WaterDataset(Dataset):
    def __init__(
        self, annotations_file: Path, images_dir: Path, transforms=None, normalize_output=False, use_water_line=None
    ):
        self.annotations_file = Path(annotations_file)
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.normalize_output = normalize_output
        self.data = self.load_annotations(use_water_line)

    def load_annotations(self, use_water_line=None):
        suffix = self.annotations_file.suffix
        if suffix == ".json":
            return self.load_coco()
        elif suffix == ".csv":
            return self.load_csv(use_water_line)
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

    def load_csv(self, use_water_line):
        def parse_line(line):
            fields = line.split(",")
            image_path = str(Path(self.images_dir) / fields[0])
            depth = float(fields[2])
            x0, y0, x1, y1 = None, None, None, None

            if len(fields) > 7 and fields[4] != "None":
                x0, y0, x1, y1 = float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])

            if use_water_line is False:
                return (image_path, [depth])

            return (image_path, [depth, x0, y0, x1, y1])

        with open(self.annotations_file, "r") as f:
            lines = f.readlines()
        data = [parse_line(line) for line in lines]

        # Remove any data points with None values
        data = [d for d in data if (None not in d[1])]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, depth = self.data[index]
        image = Image.open(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        depth = torch.tensor(depth, dtype=torch.float32)
        return image, depth, image_path


def get_transforms(crop_box=None):
    transforms_array = []

    # TODO:
    # - Normalization
    # - Randomize crop (within reason)
    # - Color jitter
    # - Make most effective use of dtypes

    if crop_box is not None:
        top, left, height, width = crop_box
        transforms_array.append(functools.partial(v2.functional.crop, top=top, left=left, height=height, width=width))

    transforms_array.extend(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # convert to float32 and normalize to 0, 1
            v2.Normalize(mean=[0.18653404, 0.20722116, 0.19524723], std=[0.15864415, 0.13673791, 0.12532181]),
        ]
    )

    # From water_train_set4:
    # Mean: [0.18653404 0.20722116 0.19524723], std: [0.15864415 0.13673791 0.12532181]

    return v2.Compose(transforms_array)


def get_data_loaders(
    images_dir: Path,
    annotations_file: Path,
    batch_size: int = 32,
    train_test_split: Tuple[int, int] = [0.8, 0.2],
    normalize_output: bool = False,
    crop_box=None,  # [top, left, height, width]
    use_water_line: bool | None = None,
):
    transforms = get_transforms(crop_box=crop_box)

    # Load dataset from directory
    dataset = WaterDataset(annotations_file, images_dir, transforms=transforms, use_water_line=use_water_line)

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
        ),
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    )


def get_data_loader(
    images_dir: Path,
    annotations_file: Path,
    batch_size: int = 32,
    shuffle: bool = True,
    normalize_output: bool = False,
    crop_box=None,  # [top, left, height, width]
    use_water_line: bool | None = None,
):
    transforms = get_transforms(crop_box=crop_box)

    # Load dataset from directory
    dataset = WaterDataset(
        annotations_file,
        images_dir,
        transforms=transforms,
        normalize_output=normalize_output,
        use_water_line=use_water_line,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
