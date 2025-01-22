# Script that trains a network model for deep water level detection
import argparse
import signal
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from utils import start_experiment  # From object_tracker_0 package

import mlflow
from annotation_utils.misc import my_device
from deep_water_level.data import get_data_loader, get_data_loaders
from deep_water_level.model import BasicCnnRegression, BasicCnnRegressionWaterLine, ModelNames, DeepLabModel

shutdown_requested = False


def signal_handler(signal, frame):
    global shutdown_requested
    shutdown_requested = True
    print("Interrupt requested. Will finish this epoch, save the model and then exit")


def create_model(model_name: ModelNames, **kwargs):
    # Create the model
    if model_name == "ResNet50Pretrained":
        # Instantiate the model
        model = torchvision.models.resnet50(num_classes=365, weights=None)
        # Load the pretrained weights
        # TODO: Download from the Web if needed
        checkpoint = torch.load(
            "models/weights/resnet50_places365.pth.tar", map_location=my_device(), weights_only=True
        )
        # Load the pretrained weights in the model
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        # Replace the last fully connected layer
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "BasicCnnRegressionWaterLine":
        model = BasicCnnRegressionWaterLine(**kwargs)
    elif model_name == "BasicCnnRegression":
        model = BasicCnnRegression(**kwargs)
    elif model_name == "DeepLabV3":
        model = DeepLabModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    kwargs["model_name"] = model_name
    return (model, kwargs)


def do_training(
    # Dataset parameters
    train_dataset_dir: Path,
    test_dataset_dir: Path,
    annotations_file: Path,
    # Training parameters
    n_epochs: int = 40,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    crop_box_jitter: list = None,
    random_rotation_degrees: int = 5,
    color_jitter: float = 0.2,
    # Pre-processing parameters
    normalize_output: bool = False,
    crop_box: list = None,
    equalization: bool = True,
    # Model parameters
    model_name: ModelNames = "BasicCnnRegression",
    dropout_p: float = None,
    n_conv_layers: int = 2,
    channel_multiplier: float = 2.0,
    conv_kernel_size: int = 4,
    conv_stride: int = 2,
    conv_padding: int = 1,
    max_pool_kernel_size: int = 2,
    max_pool_stride: int = 1,
    # Configuration parameters
    log_transformed_images: bool = False,
    report_fn: callable = lambda *args, **kwargs: None,
    output_model_path: Path = None,
):
    # In case people pass in strings for paths
    train_dataset_dir = Path(train_dataset_dir)
    test_dataset_dir = Path(test_dataset_dir)
    annotations_file = Path(annotations_file)

    device = my_device()
    # torch.set_default_device(device)
    print(f"Using device: {device}")

    preprocessing_args = {
        "normalize_output": normalize_output,
        "crop_box": crop_box,
        "equalization": equalization,
    }

    # Load the data
    if train_dataset_dir is None or test_dataset_dir == train_dataset_dir:
        # Split the train dataset in test and train datasets
        (train_data, test_data) = get_data_loaders(
            train_dataset_dir / "images",
            train_dataset_dir / "annotations" / annotations_file,
            model_name,
            batch_size=batch_size,
            crop_box=crop_box,
            crop_box_jitter=crop_box_jitter,
            equalization=equalization,
            random_rotation_degrees=random_rotation_degrees,
            color_jitter=color_jitter,
        )

    else:
        train_data = get_data_loader(
            train_dataset_dir / "images",
            train_dataset_dir / "annotations" / annotations_file,
            model_name,
            batch_size=batch_size,
            crop_box=crop_box,
            crop_box_jitter=crop_box_jitter,
            equalization=equalization,
            normalize_output=normalize_output,
            random_rotation_degrees=random_rotation_degrees,
            color_jitter=color_jitter,
        )
        test_data = get_data_loader(
            test_dataset_dir / "images",
            test_dataset_dir / "annotations" / annotations_file,
            model_name,
            batch_size=batch_size,
            shuffle=False,
            crop_box=crop_box,
            crop_box_jitter=crop_box_jitter,
            equalization=equalization,
            normalize_output=normalize_output,
            random_rotation_degrees=random_rotation_degrees,
            color_jitter=color_jitter,
        )

    if log_transformed_images:
        log_dir = Path("/tmp/deep_water_level")
        log_dir.mkdir(parents=True, exist_ok=True)
        for image_i in range(len(train_data.dataset)):
            image_filename = log_dir / f"train_data_{image_i}.png"
            image = train_data.dataset[image_i][0]
            torchvision.utils.save_image(image, image_filename)
        for image_i in range(len(test_data.dataset)):
            image_filename = log_dir / f"test_data_{image_i}.png"
            image = test_data.dataset[image_i][0]
            torchvision.utils.save_image(image, image_filename)

    # Grab the first training image to find the resolution, which determines the size of the model
    first_image = train_data.dataset[0][0]

    # Train the model
    (model, model_args) = create_model(
        model_name,
        image_size=first_image.shape,
        dropout_p=dropout_p,
        n_conv_layers=n_conv_layers,
        channel_multiplier=channel_multiplier,
        conv_kernel_size=conv_kernel_size,
        conv_stride=conv_stride,
        conv_padding=conv_padding,
        max_pool_kernel_size=max_pool_kernel_size,
        max_pool_stride=max_pool_stride,
    )
    model.to(device)
    print(f"Model summary: {model}")

    if model_name == "DeepLabV3":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # Train for the epoch
        model.train()
        train_loss = 0
        for i, (inputs, labels, filenames) in enumerate(train_data):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(".", end="", flush=True)
        print("")

        train_loss /= len(train_data)

        # Test for this epoch
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (inputs, labels, filenames) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                print(",", end="", flush=True)
            print("")

        test_loss /= len(test_data)

        # TODO: Abstract this into a reporting function
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        mlflow.log_metric("loss", train_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        report_fn({"epoch": epoch, "loss": train_loss, "test_loss": test_loss})

        global shutdown_requested
        if shutdown_requested:
            print("Stopping now due to interrupt")
            break

    # Save model to disk, locally
    if output_model_path is None:
        if model_name == "BasicCnnRegression":
            output_model_path = "model.pth"
        elif model_name == "BasicCnnRegressionWaterLine":
            output_model_path = "model_waterline.pth"
        elif model_name == "ResNet50Pretrained":
            output_model_path = "model_resnet50_pretrained.pth"
        elif model_name == "DeepLabV3":
            output_model_path = "model_deeplabv3_pretrained.pth"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    if model_name == "DeepLabV3 QQ can I remove?":
        torch.save(model.state_dict(), output_model_path)
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_args": model_args,
                "preprocessing_args": preprocessing_args,
            },
            output_model_path,
        )

    # TODO: Log Model. It's a bit trickier than this, it requires the signature to be inferred or defined properly
    # mlflow.pytorch.log_model(model, "model")

    return {"loss": train_loss, "test_loss": test_loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Deep Water Level dataset")
    parser.add_argument(
        "--train_dataset_dir",
        type=Path,
        default="datasets/water_train_set4",
        help="Path to the train dataset directory",
    )
    parser.add_argument(
        "--test_dataset_dir", type=Path, default="datasets/water_test_set5", help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--annotations_file",
        type=Path,
        default="filtered.csv",
        help="File name of the JSON file containing annotations within a dataset",
    )
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training the model")
    parser.add_argument(
        "--crop_box",
        nargs=4,
        type=int,
        default=[130, 275, 140, 140],
        help="Box with which to crop images, of form: top left height width",
    )
    parser.add_argument(
        "--crop_box_jitter",
        nargs=2,
        type=int,
        default=[4, 20],
        help="Randomize crop box by [H, W] pixels, for data augmentation",
    )
    parser.add_argument(
        "--random_rotation_degrees",
        type=int,
        default=5,
        help="Number of degrees to rotate images randomly for data augmentation",
    )
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.3,
        help="color jittering for data augmentation, 0.0 means disabled",
    )

    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0,
        help="Dropout probability to apply, from 0 to 1. None or 0.0 means disabled.",
    )
    parser.add_argument(
        "--log_transformed_images", type=bool, default=False, help="Log transformed images to /tmp/deep_water_level"
    )
    parser.add_argument("--normalize_output", type=bool, default=False, help="Normalize depth value to [-1, 1] range")
    parser.add_argument(
        "--n_conv_layers", type=int, default=3, help="Number of convolutional layers in the model (2 or 3)"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=float,
        default=3.0,
        help="Multiplier for the number of channels in each convolutional layer",
    )
    parser.add_argument("--conv_kernel_size", type=int, default=7, help="Convolutional kernel size, for all layers")
    parser.add_argument(
        "--equalization",
        type=bool,
        default=True,
        help="Perform histogram equalization on the input images to make them more uniform",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="BasicCnnRegression",
        help="Model to use for training.  See ModelNames for options",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    start_experiment("Deep Water Level Training")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("hardware/gpu", my_device())
        model = do_training(**vars(args))
