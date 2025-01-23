# This script is useful to test the deep water level model.
# It can be used to test the model on a single image manually, or on a dataset as a batch.
#
# If the dataset parameters (dataset_dir and annotations_file) are provided, the model will be run on the dataset,
# printing the inferred values vs actuals and the average loss.
#
# If the dataset parameters are omitted, the script will run
# a Gradio interface to run the deep water level model against manually uploaded images.
#
# In order to run it like this:
# - Run training or get a model, save it as model.pth at the root of the project
# - Run this script, then open the Gradio app at http://127.0.0.1:7860/
# - Add and image and click Submit to get the predicted water level
import argparse
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from annotation_utils.misc import filename_to_datetime, my_device
from deep_water_level.data import WaterDataset, get_transforms
from deep_water_level.model import BasicCnnRegression, BasicCnnRegressionWaterLine, ModelNames, DeepLabModel

VERBOSE = False


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, weights_only=False, map_location=my_device())
    model_args = checkpoint["model_args"]
    # Old models won't have this field, so assuming is BasicCnnRegression
    model_name = model_args.pop("model_name", "BasicCnnRegression")
    preprocessing_args = checkpoint.get("preprocessing_args", {})

    if model_name == "ResNet50Pretrained":
        model = torchvision.models.resnet50(num_classes=365, weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "BasicCnnRegressionWaterLine":
        model = BasicCnnRegressionWaterLine(**model_args)
    elif model_name == "BasicCnnRegression":
        model = BasicCnnRegression(**model_args)
    elif model_name == "DeepLabV3":
        model = DeepLabModel()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (model, model_name, model_args, preprocessing_args)


def run_inference(model, input, transforms=None):
    if transforms is not None:
        input = transforms(input)
    with torch.no_grad():
        output = model(input.unsqueeze(0))
    return output


def run_gradio_app(model, crop_box=None):
    transforms = get_transforms(crop_box)

    # Define the Gradio app
    demo = gr.Interface(
        fn=lambda image: run_inference(model, image, transforms),
        inputs=gr.Image(type="pil"),
        outputs=gr.Number(label="Output"),
        title="Image Regression App",
        description="Upload an image to get a scalar output",
    )

    # Launch the app
    demo.launch()


def run_dataset_inference(
    model,
    model_name: ModelNames,
    dataset_dir: Path,
    annotations_file: Path,
    normalize_output,
    crop_box=None,
    equalization: bool = False,
):
    # Convert numpy array to string for printing
    def a2s(a):
        assert len(a.shape) <= 1
        if len(a.shape) == 0 or a.shape[0] == 1:
            return f"{a.item():.4f}"
        else:
            return "[" + ", ".join([f"{x:.4f}" for x in a]) + "]"

    transforms = get_transforms(crop_box=crop_box, equalization=equalization)
    dataset = WaterDataset(
        dataset_dir / "annotations" / annotations_file,
        dataset_dir / "images",
        model_name,
        transforms=transforms,
        normalize_output=normalize_output,
    )

    # Run inference on each image in the dataset
    data = []
    for image, depth, filename in dataset:
        output = run_inference(model, image)

        # Convert tensors to 1d numpy arrays
        depth = depth.cpu().detach().numpy().squeeze()
        output = output.cpu().detach().numpy().squeeze()

        error = abs(output - depth)

        if VERBOSE:
            print(f"Filename: {filename}, Infered: {a2s(output)}, Actual: {a2s(depth)}, Error: {a2s(error)}")
        else:
            print(".", end="", flush=True)

        try:
            timestamp = filename_to_datetime(filename)
        except ValueError:
            # For some synthetic datasets, the filename is not a timestamp
            timestamp = None
        data.append(
            {
                "timestamp": timestamp,
                "filename": filename,
                "predicted": output,
                "actual": depth,
            }
        )
    if not VERBOSE:
        print()

    # Load into a pandas dataframe
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    mse = ((df["predicted"] - df["actual"]) ** 2).mean() / len(df)
    print(f"Dataset: {dataset_dir}, MSE: {a2s(mse)}, Images: {len(dataset)}")

    return df


def plot_inference_results(test_df, train_df=None, plot_filename=None):
    all_test_predictions = np.vstack(test_df["predicted"].values)
    all_test_labels = np.vstack(test_df["actual"].values)
    assert all_test_predictions.shape == all_test_labels.shape
    num_outputs = all_test_predictions.shape[1]

    if train_df is not None:
        all_train_labels = np.vstack(train_df["actual"].values)
        all_train_predictions = np.vstack(train_df["predicted"].values)
        assert all_train_predictions.shape == all_train_labels.shape
        num_outputs = all_train_predictions.shape[1]

    # Sanity check so that we don't try make thousands of plots
    assert num_outputs <= 10

    fig, ax = plt.subplots(3, num_outputs, squeeze=False, figsize=(10, 12))

    for output_i in range(num_outputs):
        test_predictions = all_test_predictions[:, output_i]
        test_labels = all_test_labels[:, output_i]
        values_vs_time_ax = ax[0, output_i]
        values_vs_index_ax = ax[1, output_i]
        scatter_ax = ax[2, output_i]

        test_timestamps = test_df.index
        values_vs_time_ax.plot(test_timestamps, test_predictions, "o", label="predicted", linestyle="None")
        values_vs_time_ax.plot(test_timestamps, test_labels, "o", label="actual", linestyle="None")
        values_vs_time_ax.legend()
        values_vs_time_ax.set_title("Depth vs Time")

        values_vs_index_ax.plot(test_predictions, "o", label="predicted", linestyle="None")
        values_vs_index_ax.plot(test_labels, "o", label="actual", linestyle="None")
        values_vs_index_ax.legend()
        values_vs_index_ax.set_title("Depth vs Index")

        artist_to_df = {}
        test_artist = scatter_ax.plot(test_labels, test_predictions, "o", label="test set")
        artist_to_df[test_artist[0]] = test_df
        min_val = min(test_labels.min(), test_predictions.min())
        max_val = max(test_labels.max(), test_predictions.max())
        if train_df is not None:
            train_predictions = all_train_predictions[:, output_i]
            train_labels = all_train_labels[:, output_i]
            train_artist = scatter_ax.plot(train_labels, train_predictions, "o", label="training set", c="magenta")
            artist_to_df[train_artist[0]] = train_df
            min_val = min(min_val, train_labels.min(), train_predictions.min())
            max_val = max(max_val, train_labels.max(), train_predictions.max())
        scatter_ax.plot([min_val, max_val], [min_val, max_val], color="grey")
        # Add interactive tooltips to show filename on mouseover
        cursor = mplcursors.cursor(ax[2], hover=True)

        @cursor.connect("add")
        def on_add(sel):
            if sel.artist in artist_to_df:
                sel.annotation.set_text(artist_to_df[sel.artist].iloc[sel.index]["filename"])
            else:
                sel.annotation.set_text("")

        scatter_ax.legend()
        scatter_ax.set_title("Predicted vs Actual Depths")
    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)


if __name__ == "__main__":
    # Parse program arguments
    parser = argparse.ArgumentParser(description="Deep Water Level")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="model.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "--normalize_output",
        type=bool,
        default=False,
        help="Set to true if the model was trained with depth values normalize to [-1, 1] range",
    )

    # If these arguments are provided, then the model will be run against the dataset, showing results.
    parser.add_argument(
        "--dataset_dir", type=Path, default="datasets/water_test_set5", help="Path to the dataset directory"
    )
    parser.add_argument(
        "--annotations_file", type=str, default="filtered.csv", help="File name of the JSON file containing annotations"
    )

    # If these arguments are provided, then the model will also be run against the training dataset, and the
    # results will be added to the scatterplot (if scatter_plot is set to True)
    parser.add_argument("--train_dataset_dir", type=Path, default=None, help="Path to the training dataset directory")
    parser.add_argument(
        "--train_annotations_file",
        type=str,
        default="filtered.csv",
        help="File name of the JSON file containing annotations used during training",
    )

    parser.add_argument(
        "--annotations_out_filename",
        type=str,
        default="inferred.csv",
        help="Write a CSV annotations file with the infered values",
    )
    parser.add_argument(
        "--scatter_plot", type=bool, default=True, help="Show a scatter plot of actual vs predicted values"
    )

    parser.add_argument("--verbose", action="store_true", help="Print error for each image in the dataset")

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    (model, model_name, model_args, preprocessing_args) = load_model(args.model_path)

    equalization = preprocessing_args["equalization"] if "equalization" in preprocessing_args else True
    crop_box = preprocessing_args["crop_box"] if "crop_box" in preprocessing_args else None

    # Load the model
    if "dataset_dir" in args and "annotations_file" in args:
        test_df = run_dataset_inference(
            model,
            model_name,
            dataset_dir=args.dataset_dir,
            annotations_file=args.annotations_file,
            normalize_output=args.normalize_output,
            crop_box=crop_box,
            equalization=equalization,
        )
        if args.annotations_out_filename is not None:
            test_df.to_csv(args.annotations_out_filename, index=False)

        train_df = None
        if args.train_dataset_dir is not None and args.train_annotations_file is not None:
            train_df = run_dataset_inference(
                model,
                model_name,
                dataset_dir=args.train_dataset_dir,
                annotations_file=args.train_annotations_file,
                normalize_output=args.normalize_output,
                crop_box=crop_box,
                equalization=equalization,
            )

        if args.scatter_plot:
            plot_inference_results(test_df, train_df)
            plt.show()
    else:
        run_gradio_app(model, crop_box=crop_box)
