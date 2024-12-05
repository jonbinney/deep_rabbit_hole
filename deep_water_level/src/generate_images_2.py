import csv
import functools
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_WIDTH = 810
IMAGE_HEIGHT = 510


def generate_mirrored_image(
    level: float,
    box_height: int = 0.1,
    max_top_level_offset: int = 0.25,
    level_scale: float = 0.1 / 11.0,  # Ratio of fraction_of_image/level
):
    box_height_in_pixels = int(round(IMAGE_HEIGHT * box_height))
    box_top_row = int(round(IMAGE_HEIGHT * np.random.uniform(0.0, max_top_level_offset)))
    box_bottom_row = box_top_row + box_height_in_pixels
    mirror_row = int(round(IMAGE_HEIGHT / 2 - IMAGE_HEIGHT * level_scale * level))

    # Make sure we don't accidentally generate an image that doesn't make sense
    assert mirror_row > 0 and mirror_row < IMAGE_HEIGHT - 1
    assert box_top_row >= 0 and box_bottom_row < mirror_row - 1

    # Pick random RGB colors for foreground and background
    background_color = np.random.randint(0, 256, 3)
    foreground_color = np.random.randint(0, 256, 3)

    image = np.tile(np.array(background_color, dtype=np.uint8), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    # Draw first rectangle, which represents some object above the water (like the shed)
    image[box_top_row:box_bottom_row, 0:IMAGE_WIDTH] = foreground_color

    # Draw another rectangle which is the same as the first, but mirrored around mirror_row
    mirror_box_top_row = mirror_row + (mirror_row - box_bottom_row)
    mirror_box_bottom_row = mirror_row + (mirror_row - box_top_row)
    image[mirror_box_top_row:mirror_box_bottom_row, 0:IMAGE_WIDTH] = foreground_color

    return image


def generate_dataset(
    dir: Path,
    num_images: int,
    image_generator: callable,
):
    for folder in ["images", "annotations"]:
        folder_path = dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for filename in folder_path.iterdir():
            if filename.is_file():
                filename.unlink()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (dir / "annotations" / "filtered.csv").open("w", newline="") as file:
        count = 0
        writer = csv.writer(file)
        for count in range(num_images):
            level = np.random.uniform(4.0, 11.0)
            image = image_generator(level)
            cv2.imwrite(f"{dir}/images/img_{count:03}.jpg", image)
            writer.writerow([f"img_{count:03}.jpg", timestamp, level, "n/a"])


if __name__ == "__main__":
    num_train_images = 20000
    num_test_images = 200
    image_generator = functools.partial(generate_mirrored_image, box_height=0.1, max_top_level_offset=0.25)

    generate_dataset(Path("datasets/fake_water_images_train4"), num_train_images, image_generator)

    generate_dataset(Path("datasets/fake_water_images_test4"), num_test_images, image_generator)
