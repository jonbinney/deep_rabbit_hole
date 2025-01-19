"""
This small script takes a date and a known depth and generates
a new dataset of all good images labelled with that depth.
NOTE: These datasets will be lacking the segmentation annotation.

Example:

python make_day.py --date=2024-10-06 --depth=1 --dataset_path=datasets/auto_day_2024-10-06
"""

import argparse
import datetime
import json
from google.cloud.storage import Client
from google.cloud.storage import transfer_manager
import os
from pathlib import Path

from annotation_utils.misc import too_little_contrast


def download_day(date: str, destination_dir: Path, bucket_name: str = "deep-datasets", camera_num: int = 0):
    # Authenticate using Application Default Credentials
    storage_client = Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List all the blobs in the subdirectory
    prefix = Path("pilenew") / "images" / date
    blobs = bucket.list_blobs(prefix=str(prefix / f"{camera_num}-"))
    blob_names = [blob.name for blob in blobs]

    # Download all the blobs
    result = transfer_manager.download_many_to_path(bucket, blob_names, destination_dir)

    # Adjust the directory structure so it's only {dataset_path}/images/{date}
    source_path = destination_dir / prefix
    dataset_images_path = destination_dir / "images" / date
    os.makedirs(dataset_images_path, exist_ok=True)
    os.rename(source_path, dataset_images_path)
    os.removedirs(destination_dir / "pilenew" / "images")

    print(
        f"Downloaded {len(blob_names)} images to {dataset_images_path}. Success = {len(list(filter(lambda r: not isinstance(r, Exception), result)))}"
    )


def create_single_depth_annotations_coco(dataset_path: Path, depth: float):
    data = {}
    data["info"] = {
        "date_created": datetime.datetime.now().isoformat(),
        "description": "Automatically generated with same depth for all images",
    }
    data["categories"] = [{"id": 1, "name": "water", "supercategory": ""}]
    # TODO: Read width and height from image instead of hard-coding it
    image_filenames = [str(file.relative_to(dataset_path / "images")) for file in dataset_path.rglob("*.jpg")]
    data["images"] = [
        {
            "file_name": f,
            "id": i,
            "width": 810,
            "height": 510,
        }
        for i, f in enumerate(image_filenames)
    ]
    data["annotations"] = [
        {
            "id": i,
            "image_id": i,
            "category_id": 1,
            "attributes": {
                "depth": depth,
            },
        }
        for i in range(len(data["images"]))
    ]
    os.makedirs(f"{dataset_path}/annotations", exist_ok=True)
    json.dump(data, open(f"{dataset_path}/annotations/annotations.json", "w"), indent=2)

    print(f"Created annotations file at {dataset_path}/annotations/annotations.json with {len(data['images'])} images")


def filter_images(dataset_path: Path):
    removed = []
    for file in dataset_path.rglob("*.jpg"):
        if too_little_contrast(file.resolve()):
            file.unlink()
            removed.append(file.name)

    print(f"Filtered out {len(removed)} low contrast images")


def make_day(date: str, depth: float, dataset_path: str):
    dataset_path = Path(dataset_path)

    print(f"Creating dataset for day {date} with depth {depth} at {dataset_path}")

    # 1. Download raw data images for the specified day in the specified dataset path
    #    This will be in the form {dataset_path}/images/{date}
    download_day(date, dataset_path)

    # 2. Basic filtering on image quality (minimum amount of lightning and contrast)
    filter_images(dataset_path)

    # 3. Create annotations file (coco)
    create_single_depth_annotations_coco(dataset_path, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    make_day(**vars(args))
