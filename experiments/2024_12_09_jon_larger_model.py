from pathlib import Path

from matplotlib import pyplot as plt

from deep_water_level.infer import load_model, plot_inference_results, run_dataset_inference
from deep_water_level.train import do_training

annotations_file = Path("filtered.csv")
train_dataset_path = Path("datasets/water_train_set4")
test_dataset_path = Path("datasets/water_test_set5")
crop_box = [130, 275, 140, 140]
train_water_line = False
model_filename = Path("model.pth")
parent_output_dir = Path("../dwl_output")

output_dir = parent_output_dir / f"large_conv"
output_model_path = output_dir / model_filename
output_dir.mkdir(parents=True, exist_ok=True)

do_training(
    # Dataset parameters
    train_dataset_path,
    test_dataset_path,
    annotations_file,
    # Training parameters
    n_epochs=4,
    learning_rate=1e-3,
    normalize_output=False,
    crop_box=None,
    # Model parameters
    dropout_p=None,
    n_conv_layers=2,
    channel_multiplier=2.0,
    conv_kernel_size=8,
    conv_stride=2,
    conv_padding=1,
    max_pool_kernel_size=2,
    max_pool_stride=1,
    # Configuration parameters
    log_transformed_images=False,
    train_water_line=False,
    output_model_path=output_model_path,
)

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

plot_inference_results(test_df, train_df)
plt.show()
