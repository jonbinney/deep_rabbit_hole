import time
from pathlib import Path
import os
import torch
import torch.nn as nn
from annotation_utils.misc import my_device
from deep_water_level.deeplabv3.utils import SegmentationDataset
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms import functional as F

import mlflow


def do_training(
    # Dataset parameters
    train_dataset_dir: Path,
    test_dataset_dir: Path | None,
    # Training parameters
    n_epochs: int = 40,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
):
    if test_dataset_dir is None or test_dataset_dir == train_dataset_dir:
        # Split the train dataset in test and train datasets
        dataset = SegmentationDataset(train_dataset_dir, train_dataset_dir / "annotations" / "filtered.json")

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    else:
        train_dataset = SegmentationDataset(train_dataset_dir, train_dataset_dir / "annotations" / "filtered.json")
        test_dataset = SegmentationDataset(test_dataset_dir, test_dataset_dir / "annotations" / "filtered.json")

    print(f"Training dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    device = my_device()

    model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier = DeepLabHead(2048, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Print the number of parameters being trained
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for images, masks in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print("Batch", time.time() - t0)

        epoch_train_loss = train_loss / len(train_loader)

        # Test for this epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)["out"]
                loss = criterion(outputs, masks)
                test_loss += loss.item()

        epoch_test_loss = test_loss / len(test_loader)
        print(
            f"Epoch {epoch + 1}, train loss: {epoch_train_loss}, test loss: {epoch_test_loss}, t: {int(time.time() - t0)}s"
        )

        mlflow.log_metric("loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("test_loss", epoch_test_loss, step=epoch)

        torch.save(model.state_dict(), "deeplabv3_pool.pth")


# Couldn't figure out how to import this, facepalm
def start_experiment(experiment_name: str):
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    print(f"Using MLFlow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location="gs://deep-rabbit-hole/mlflow")
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()


start_experiment("Deeplabv3 Training")

with mlflow.start_run():
    do_training(
        Path("datasets/water_2024_10_19_set1/"),
        Path("datasets/water_test_set3/"),
        n_epochs=5,
        batch_size=4,
    )
