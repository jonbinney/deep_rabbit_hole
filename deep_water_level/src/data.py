import os
import json
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
        images = annotations.get('images', [])
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            if image_id < len(images):
                image_file = images[image_id]['file_name']
                image_path = os.path.join(self.images_dir, image_file)
                depth = annotation.get('attributes', {})['depth']
                # NOTE: The second item [depth] is the lable. It's an array because it could
                # include many labels
                data.append((image_path, [depth]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, depth = self.data[index]
        image = Image.open(image_path)
        if self.transforms is not None:
            image = self.transforms(image)
        depth = torch.tensor(depth, dtype=torch.float32)
        return image, depth

def get_data_loaders(
    annotations_file: str,
    images_dir: str,
    batch_size: int = 32,
    train_test_split: Tuple[int, int] = [0.8, 0.2],
):
    transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True), # convert to float32 and normalize to 0, 1
    ])

    # Load dataset from directory
    dataset = WaterDataset(annotations_file, images_dir, transforms=transforms)

    # Split dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, train_test_split)

    # Create PyTorch DataLoaders for train and test splits
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
