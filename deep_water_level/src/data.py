import os
import json
import mlflow
from PIL import Image
from torchvision.transforms import v2
import torch
from torch.utils.data import Dataset
from typing import Tuple

class WaterDataset(Dataset):
    def __init__(self, annotations_file, images_dir, transforms=None):
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.transforms = transforms
        self.data = self.load_annotations()

    def load_annotations(self):
        with open(self.annotations_file, 'r') as f:
            annotations = json.load(f)
        data = []
        # Create a map of image_id to image - IMPORTANT: image_id might not match index in the images array
        images = annotations.get('images', [])
        images_by_id = {}
        for image in images:
            images_by_id[image['id']] = image
        # Go through annotations to get image_path and depth for each annotation
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            image_file = images_by_id.get(image_id, {}).get('file_name', None)
            if image_file is not None:
                image_path = os.path.join(self.images_dir, image_file)
                depth = annotation.get('attributes', {})['depth']
                # NOTE: The second item [depth] is the lable. It's an array because it could
                # include many labels
                data.append((image_path, [depth]))
            else:
                print(f"WARN: No image found for annotation. image_id: {image_id}")
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
    
def get_transforms():
    return v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True), # convert to float32 and normalize to 0, 1
    ])

def get_data_loaders(
    images_dir: str,
    annotations_file: str,
    batch_size: int = 32,
    train_test_split: Tuple[int, int] = [0.8, 0.2],
):
    transforms = get_transforms() 

    # Load dataset from directory
    dataset = WaterDataset(annotations_file, images_dir, transforms=transforms)

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
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )

def get_data_loader(
    images_dir: str,
    annotations_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
):
    transforms = get_transforms() 

    # Load dataset from directory
    dataset = WaterDataset(annotations_file, images_dir, transforms=transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)