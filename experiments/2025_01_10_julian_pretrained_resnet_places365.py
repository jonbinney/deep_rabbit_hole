# TODO:
#  - Adapt to the proper image input size, probably with cropping
#  - Data augmentation
from pathlib import Path
import mlflow

from deep_water_level.infer import load_model, plot_inference_results, run_dataset_inference
from deep_water_level.train import do_training
from utils import my_device, start_experiment

annotations_file = Path("filtered.csv")
train_dataset_path = Path("datasets/water_train_set4")
test_dataset_path = Path("datasets/water_test_set5")
crop_box = [88, 234, 224, 224]  # Cropping image to Resnet 50 input size. TODO: Verify
model_filename = Path("model.pth")
parent_output_dir = Path("../dwl_output")
model_name = "ResNet50Pretrained"

output_dir = parent_output_dir / "pretrained_resnet_places365"
output_model_path = output_dir / model_filename
output_dir.mkdir(parents=True, exist_ok=True)

args = {
    # Dataset parameters
    "train_dataset_dir": train_dataset_path,
    "test_dataset_dir": test_dataset_path,
    "annotations_file": annotations_file,
    # Preprocessing params
    "crop_box": crop_box,
    "normalize_output": False,
    "equalization": True,
    # Training parameters
    "batch_size": 4,
    "n_epochs": 30,
    "learning_rate": 1e-4,
    "random_rotation_degrees": 10,
    "crop_box_jitter": [10, 30],
    "color_jitter": 0.3,
    # Model parameters
    "model_name": model_name,
    "dropout_p": 0.1,
    "n_conv_layers": 2,
    "channel_multiplier": 4.0,
    "conv_kernel_size": 7,
    "conv_stride": 1,
    "conv_padding": 1,
    "max_pool_kernel_size": 2,
    "max_pool_stride": 1,
    # Configuration parameters
    "log_transformed_images": False,
    "output_model_path": output_model_path,
}

start_experiment("Deep Water Level Training - Pretrained")
with mlflow.start_run():
    mlflow.log_params(args)
    mlflow.log_param("hardware/gpu", my_device())
    mlflow.log_param("model", "pretrained_resnet_places365")
    do_training(**args)

    # Load the model and run inference using it
    model, model_args, preprocessing_args = load_model(output_model_path, model_name)

    train_df = run_dataset_inference(
        model,
        model_name,
        train_dataset_path,
        annotations_file,
        **preprocessing_args,
    )

    test_df = run_dataset_inference(
        model,
        model_name,
        test_dataset_path,
        annotations_file,
        **preprocessing_args,
    )

    plot_inference_results(test_df, train_df, plot_filename="inference_results.png")
    mlflow.log_artifact("inference_results.png")
