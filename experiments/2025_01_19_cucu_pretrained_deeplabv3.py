# TO DO: Implement geometric transformations on the mask so that we can do data augmentation

from pathlib import Path

from utils import my_device, start_experiment

import mlflow
from deep_water_level.train import do_training

annotations_file = Path("filtered.json")
train_dataset_path = Path("datasets/water_2024_10_19_set1")
test_dataset_path = Path("datasets/water_test_set3")
# crop_box = [88, 234, 224, 224]
model_filename = Path("model.pth")
parent_output_dir = Path("../dwl_output")
model_name = "DeepLabV3"

output_dir = parent_output_dir / "deeplabv3"
output_model_path = output_dir / model_filename
output_dir.mkdir(parents=True, exist_ok=True)

args = {
    # Dataset parameters
    "train_dataset_dir": train_dataset_path,
    "test_dataset_dir": test_dataset_path,
    "annotations_file": annotations_file,
    # Preprocessing params
    #   "crop_box": crop_box,
    "normalize_output": False,
    "equalization": True,
    # Training parameters
    "batch_size": 2,
    "n_epochs": 1,  # QQ 5
    "learning_rate": 1e-3,
    # "random_rotation_degrees": 15,
    # "crop_box_jitter": [10, 30],
    "color_jitter": 0.3,
    # Model parameters
    "model_name": model_name,
    # Configuration parameters
    "log_transformed_images": False,
    "output_model_path": output_model_path,
}

start_experiment("Deep Water Level Training - DeepLabV3")
with mlflow.start_run():
    mlflow.log_params(args)
    mlflow.log_param("hardware/gpu", my_device())
    mlflow.log_param("model", "deeplabv3")
    do_training(**args)
