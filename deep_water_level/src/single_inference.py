"""
Run inference on a single image
NOTE: This script is mean to be run in the Raspberry Pi that has
access to the live images. As such, it has different versions of
the underlying libraries than what we use elsewhere, e.g.:
  torchvision = 0.8.0a0  (through APT: python3-torchvision 0.8.2-1)
  torch = 1.7.0a0  (through APT: python3-torch 1.7.1-7)
For this reason, much of the code is copied and adapted instead of reused
from the other files.
TODO: Script and improve how deplyment of this file is done
"""

import time
import torch
from PIL import Image, ImageOps

# from model import BasicCnnRegression
import argparse
from torchvision import transforms as t
from torchvision import models
import functools

# Global value to control output verbosity
verbose = False


def load_model(model_path):
    # checkpoint = torch.load(model_path, weights_only=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    preprocessing_args = checkpoint.get("preprocessing_args", {})
    if verbose:
        print(f"Loaded model arguments: {model_args}")
        print(f"Loaded preprocessing arguments: {preprocessing_args}")

    # TODO: Choose model based on arguments, right now it's hardcoded
    # Hand-crafted model
    # model = BasicCnnRegression(**model_args)

    # Fine-tuned resnet50
    model = models.resnet50(pretrained=False, num_classes=365)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (model, model_args, preprocessing_args)


def get_transforms(crop_box=None):
    transforms_array = []

    if crop_box is not None:
        top, left, height, width = crop_box
        transforms_array.append(functools.partial(t.functional.crop, top=top, left=left, height=height, width=width))

    transforms_array.extend(
        [
            t.ToTensor(),
            t.ConvertImageDtype(torch.float32),
        ]
    )

    return t.Compose(transforms_array)


def run_inference(model, input, transforms=None):
    if transforms is not None:
        input = transforms(input)
    with torch.no_grad():
        output = model(input.unsqueeze(0))
    return output.item()


if __name__ == "__main__":
    # Parse program arguments
    parser = argparse.ArgumentParser(description="Deep Water Level")
    parser.add_argument("-m", "--model_path", type=str, default="model.pth", help="Path to the model file")
    parser.add_argument("--image_filename", type=str, default="image.jpg", help="Path to a sample image file")
    parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    # Load the model
    (model, model_args, preprocessing_args) = load_model(args.model_path)

    # Load the transforms
    crop_box = preprocessing_args.get("crop_box", None)
    transforms = get_transforms(crop_box=crop_box)

    # Load the data
    equalization = preprocessing_args.get("equalization", False)
    input = Image.open(args.image_filename)
    if equalization:
        input = ImageOps.equalize(input)

    # Record the current timestamp
    start_time = time.time()
    depth = run_inference(model, input, transforms)
    end_time = time.time()

    if verbose:
        print(f"We got depth = {depth} in {end_time - start_time} seconds")
    print(f"{depth}")
