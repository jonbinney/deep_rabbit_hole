import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from deep_water_level.data import get_transforms
from deep_water_level.infer import load_model


def visualize_segmentation(image, mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)

    plt.imshow(mask, cmap="jet", alpha=0.7)
    plt.title("Segmentation Mask")
    plt.axis("off")

    combined = image.clone()
    combined[mask == 1] = torch.tensor([1.0, 0, 0])

    plt.subplot(1, 3, 3)
    plt.imshow(combined)
    plt.title("Image with Segmentation Mask")
    plt.axis("off")
    plt.show()


def run_segmentation(model, image_path):
    image = Image.open(image_path)
    tf = get_transforms(is_training=False)
    preprocessed_image = tf(image).unsqueeze(0)

    output = model(preprocessed_image)
    mask = torch.argmax(output, dim=1).squeeze().numpy()
    displayable_image = transforms.ToTensor()(image).permute(1, 2, 0)

    return displayable_image, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer Segmentation")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="../dwl_output/deeplabv3/model.pth",
        help="Path to the model file",
    )

    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="datasets/water_test_set5/images/2024-10-29/0-1730238001.jpg",
        help="Path to an image file to run segmentation",
    )

    args = parser.parse_args()

    model, model_name, model_args, preprocessing_args = load_model(Path(args.model_path))

    image, mask = run_segmentation(model, args.image_path)

    visualize_segmentation(image, mask)
