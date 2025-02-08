import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from deep_water_level.data import ImageTransforms
from deep_water_level.infer import load_model


def visualize_segmentations(segmentations):
    n = len(segmentations)
    plt.figure(figsize=(12, 6))

    for i, (image, mask) in enumerate(segmentations):
        plt.subplot(3, n, i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(mask, cmap="jet", alpha=0.7)
        plt.axis("off")

        combined = image.clone()
        combined[mask == 1] = torch.tensor([1.0, 0, 0])

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(combined)
        plt.axis("off")

    plt.show()


def run_segmentations(model, image_paths):
    def crop_box_function():
        top = random.randint(50, 150)
        left = random.randint(0, 500)
        width = random.randint(100, 500)
        height = random.randint(width - 20, width + 20)  # make them square-ish
        return [top, left, min(height, 510), min(width, 810)]

    result = []
    for image_path in image_paths:
        image = Image.open(image_path)

        transformation = ImageTransforms(crop_box=crop_box_function, is_training=False)
        preprocessed_image = transformation(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        output = model(preprocessed_image)
        mask = torch.argmax(output, dim=1).squeeze().numpy()
        # displayable_image = transforms.ToTensor()(image).permute(1, 2, 0)

        result.append((preprocessed_image.squeeze(0).permute(1, 2, 0), mask))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer Segmentation")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="../dwl_output/deeplabv3/model_deleteme.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        default="datasets/water_2024_10_19_set1/images/",  # water_2024_11_01_set2
        help="Path to a directory to find random images to segment",
    )
    parser.add_argument(
        "-n",
        "--num_images",
        type=int,
        default=10,
        help="Number of images to segment",
    )

    args = parser.parse_args()

    model, model_name, model_args, preprocessing_args = load_model(Path(args.model_path))

    # Get random jpg from directory
    jpg_files = list(Path(args.image_dir).glob("**/*.jpg"))
    if not jpg_files:
        raise ValueError(f"No jpg files found in {args.image_dir}")

    random_images = random.choices(jpg_files, k=args.num_images)

    segmentations = run_segmentations(model, random_images)

    visualize_segmentations(segmentations)
