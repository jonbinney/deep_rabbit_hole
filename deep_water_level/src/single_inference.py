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
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--multi_croppings", action="store_true", default=False, help="Perform and calcucate two more croppings"
    )

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    # Load the model
    (model, model_args, preprocessing_args) = load_model(args.model_path)

    # Load the data
    equalization = preprocessing_args.get("equalization", False)
    input = Image.open(args.image_filename)
    if equalization:
        input = ImageOps.equalize(input)

    # Get crop box(es)
    crop_boxes = [preprocessing_args.get("crop_box", None)]
    if crop_boxes[0] and args.multi_croppings:
        (y, x, h, w) = crop_boxes[0]
        (dy, dx) = [20, -100]
        crop_boxes.append([y + dy, x + dx, h, w])
        crop_boxes.append([y - dy, x - dx, h, w])

    # Calculate the depth for each crop
    depths = []
    for crop_box in crop_boxes:
        transforms = get_transforms(crop_box=crop_box)
        start_time = time.time()
        depth = run_inference(model, input, transforms)
        depths.append(depth)
        end_time = time.time()
        if verbose:
            print(f"We got depth = {depth} in {end_time - start_time} seconds, crop_box={crop_box}")

    # TODO: Do something interesting with the multiple results if there are many
    # That requires also changing the integration scripts that sends this to
    # home assistant
    # Idea: Calculate a standard deviation to understand uncertainty
    # Idea: Remove outlier if there is a clear one
    # Idea: Record everything so we can analyze it later
    for depth in depths:
        print(depth)
