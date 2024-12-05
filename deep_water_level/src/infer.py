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
import gradio as gr
import torch
from model import BasicCnnRegression, BasicCnnRegressionWaterLine
from data import get_transforms, get_data_loader
import argparse
from misc import filename_to_datetime
import matplotlib.pyplot as plt
import pandas as pd


def load_model(model_path, train_water_line):
    if model_path is None:
        if train_water_line:
            model_path = BasicCnnRegressionWaterLine.DEFAULT_MODEL_FILENAME
        else:
            model_path = BasicCnnRegression.DEFAULT_MODEL_FILENAME

    checkpoint = torch.load(model_path, weights_only=False)
    model_args = checkpoint["model_args"]
    if train_water_line:
        model = BasicCnnRegressionWaterLine(**model_args)
    else:
        model = BasicCnnRegression(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (model, model_args)


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
    dataset_dir,
    annotations_file,
    normalized_output,
    crop_box=None,
    scatter_plot=None,
    annotations_out_filename=None,
    use_water_line=False,
    **kwargs,
):
    # tensor to string
    def t2s(t):
        t = t.squeeze()
        if len(t.shape) == 0 or t.shape[0] == 1:
            return f"{t.item():.2f}"

        return "[" + ", ".join([f"{x:.2f}" for x in t]) + "]"

    # Load the dataset
    dataset = get_data_loader(
        dataset_dir + "/images",
        dataset_dir + "/annotations/" + annotations_file,
        shuffle=False,
        crop_box=crop_box,
        normalize_output=normalized_output,
        use_water_line=use_water_line,
    )

    # Run inference
    loss = 0
    n_images = 0
    outputs = []
    labeled_depths = []

    data = []

    for i, (images, depths, filenames) in enumerate(dataset):
        mse = 0
        for image, depth, filename in zip(images, depths, filenames):
            output = run_inference(model, image)
            outputs.append(output)
            if not use_water_line:
                labeled_depths.append(depth.item())
            error = abs(output - depth)

            print(f"Filename: {filename}, Infered: {t2s(output)}, Actual: {t2s(depth)}, Error: {t2s(error)}")
            data.append(
                {
                    "timestamp": filename_to_datetime(filename),
                    "filename": filename,
                    "predicted": output.item(),
                    "actual": depth.item(),
                }
            )
            mse += error**2
        mse /= len(images)
        print(f"MSE ({i}): {mse}")
        loss += mse
        n_images += len(images)

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    if annotations_out_filename is not None:
        df.to_csv(annotations_out_filename, index=False)

    if scatter_plot and not use_water_line:
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        ax[0].plot(df.index, df["predicted"], "o", label="predicted", linestyle="None")
        ax[0].plot(df.index, df["actual"], "o", label="actual", linestyle="None")
        ax[0].legend()
        ax[0].set_title("Predicted vs Actual Depths (Time)")

        ax[1].plot(range(len(df)), df["predicted"], "o", label="predicted", linestyle="None")
        ax[1].plot(range(len(df)), df["actual"], "o", label="actual", linestyle="None")
        ax[1].legend()
        ax[1].set_title("Predicted vs Actual Depths (Consecutive)")

        ax[2].scatter(labeled_depths, outputs)
        ax[2].plot([0, max(labeled_depths)], [0, max(labeled_depths)], color="red")
        ax[2].set_title("Predicted vs Actual Depths")
        plt.tight_layout()
        plt.show()

    print(f"Average loss: {loss / len(dataset)}, images: {n_images}, dataset size: {len(dataset)}")


if __name__ == "__main__":
    # Parse program arguments
    parser = argparse.ArgumentParser(description="Deep Water Level")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="Path to the model file")
    parser.add_argument(
        "--normalized_output",
        type=bool,
        default=False,
        help="Set to true if the model was trained with depth values normalized to [-1, 1] range",
    )
    parser.add_argument(
        "--crop_box",
        nargs=4,
        type=int,
        default=[130, 275, 140, 140],
        help="Box with which to crop images, of form: top left height width",
    )

    # If these arguments are provided, then the model will be run against the dataset, showing results.
    parser.add_argument(
        "--dataset_dir", type=str, default="datasets/water_test_set5", help="Path to the dataset directory"
    )
    parser.add_argument(
        "--annotations_file", type=str, default="filtered.csv", help="File name of the JSON file containing annotations"
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
    parser.add_argument(
        "--use_water_line",
        type=bool,
        default=False,
        help="If set, do inference of the water level coordinates as output instead of depth",
    )

    args = parser.parse_args()

    (model, model_args) = load_model(args.model_path, args.use_water_line)

    # Load the model
    if "dataset_dir" in args and "annotations_file" in args:
        run_dataset_inference(model, **vars(args))
    else:
        run_gradio_app(model, model_args["crop_box"])
