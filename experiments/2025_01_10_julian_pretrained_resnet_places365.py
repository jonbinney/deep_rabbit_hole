from pathlib import Path
import mlflow

from matplotlib import pyplot as plt

from deep_water_level.infer import load_model, plot_inference_results, run_dataset_inference
from deep_water_level.train import do_training
from utils import my_device, start_experiment

annotations_file = Path("filtered.csv")
train_dataset_path = Path("datasets/water_train_set4")
test_dataset_path = Path("datasets/water_test_set5")
crop_box = None
train_water_line = False
model_filename = Path("model.pth")
parent_output_dir = Path("../dwl_output")

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
    "n_epochs": 3,
    "learning_rate": 1e-4,
    "random_rotation_degrees": 0,
    "crop_box_jitter": None,
    "color_jitter": 0,
    # Model parameters
    "use_pretrained": True,
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
    "train_water_line": train_water_line,
}

start_experiment("Deep Water Level Training - Pretrained")
with mlflow.start_run():
    mlflow.log_params(args)
    mlflow.log_param("hardware/gpu", my_device())
    mlflow.log_param("model", "pretrained_resnet_places365")
    do_training(**args)

    # Load the model and run inference using it
    model, model_args, preprocessing_args = load_model(output_model_path, train_water_line)

    train_df = run_dataset_inference(
        model,
        train_dataset_path,
        annotations_file,
        **preprocessing_args,
        use_water_line=train_water_line,
    )

    test_df = run_dataset_inference(
        model,
        test_dataset_path,
        annotations_file,
        **preprocessing_args,
        use_water_line=train_water_line,
    )

    plot_inference_results(test_df, train_df, plot_filename="inference_results.png")
    mlflow.log_artifact("inference_results.png")
