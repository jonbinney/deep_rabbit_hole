import csv
import os
from pathlib import Path
import time

import cv2
import numpy as np

IMG_WIDTH = 810
IMG_HEIGHT = 510


def generate_data(
    dir: str,
    levels: list[float],
    min_images_per_level=1,
    max_images_per_level=1,
    pixels_per_level: int = 20,
    max_noise=0,
    max_top_level_offset=0,
    max_water_color_variation=0,
):
    """
    Generates images and annotations and saves them in the given directory.

    Args:
        dir: the base directory,
        levels: A list of water level
        min_images_per_level: the minimum number of images to generate for each level, the exact number will be random
        max_images_per_level: the maximum number of images to generate for each level, the exact number will be random
        pixels_per_level: how many pixels represents a level
        max_noise: the maximum noise to add to the image, or 0 for no noise
        max_top_level_offset: the maximum random offset to add to the top level, or 0 for no offset
        max_water_color_variation: the maximum color variation to add to the water, or 0 for no variation
    """

    os.makedirs(f"{dir}/images", exist_ok=True)
    os.makedirs(f"{dir}/annotations", exist_ok=True)

    for folder in ["images", "annotations"]:
        folder_path = os.path.join(dir, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{dir}/annotations/filtered.csv", "w", newline="") as file:
        count = 0
        writer = csv.writer(file)
        for level in levels:
            num_img = np.random.randint(min_images_per_level, max_images_per_level + 1)
            for _ in range(num_img):
                image = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 255
                top_left = (0, 100 + np.random.randint(0, max_top_level_offset + 1))
                bottom_right = (
                    IMG_WIDTH - 1,
                    top_left[1] + int(level * pixels_per_level),
                )
                blue = np.random.randint(-max_water_color_variation, max_water_color_variation + 1) + 128
                not_blue = np.random.randint(0, max_water_color_variation + 1)
                cv2.rectangle(image, top_left, bottom_right, (blue, not_blue, not_blue), -1)

                if max_noise > 0:
                    noise = np.random.randint(-max_noise, max_noise, (IMG_HEIGHT, IMG_WIDTH, 3))
                    image = np.clip(image + noise, 0, 255)

                cv2.imwrite(f"{dir}/images/img_{count:03}.jpg", image)
                writer.writerow([f"img_{count:03}.jpg", timestamp, level, "n/a"])
                count += 1


if __name__ == "__main__":
    # Very simple training dataset with 50 different levels.  It uses a lot of levels because since
    # there are no noise or other variations, there's only 1 possible image for each level
    # The training set is missing levels, so we can see if the test set can predict them.
    levels_not_to_train = {
        3,
        7,
        11,
        18,
        26,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        43,
        47,
    }
    all_levels = [float(level) for level in range(0, 50)]
    train_levels = [level for i, level in enumerate(all_levels) if i not in levels_not_to_train]
    generate_data("datasets/fake_water_images_train1", train_levels, pixels_per_level=5)
    generate_data("datasets/fake_water_images_test1", all_levels, pixels_per_level=5)

    # More sophisticated data set with noise, color variation, and random number of images per level.
    # The levels also look more like what we actually have.
    train_levels = [4.0, 5.0, 6.0, 6.5, 7.2, 11.0]
    test_levels = [
        4.0,
        4.5,
        5.0,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
        10.5,
        11.0,
    ]
    args = {
        "min_images_per_level": 20,
        "max_images_per_level": 30,
        "max_noise": 20,
        "max_top_level_offset": 10,
        "max_water_color_variation": 30,
    }
    generate_data("datasets/fake_water_images_train2", train_levels, pixels_per_level=5, **args)
    generate_data("datasets/fake_water_images_test2", test_levels, pixels_per_level=5, **args)

    # Similar to the above but with more variability
    args = {
        "min_images_per_level": 5,
        "max_images_per_level": 30,
        "max_noise": 40,
        "max_top_level_offset": 20,
        "max_water_color_variation": 60,
    }
    generate_data("datasets/fake_water_images_train3", train_levels, pixels_per_level=5, **args)
    generate_data("datasets/fake_water_images_test3", test_levels, pixels_per_level=5, **args)

    # Just show an image to see how it looks like
    cv2.imshow("image", cv2.imread("datasets/fake_water_images_train3/images/img_000.jpg"))
    cv2.waitKey(0)
