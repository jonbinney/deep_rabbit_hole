# Merge two coco annotation files, joining annotation and images, fixing indices.
# More precisely, the annotations and images from the second file will be appended to the first file,
# Keeping all the metadata from the first file.
# If there are duplicate images, we fail for now.
import argparse
import json


class CocoImage:
    def __init__(self, image):
        self.image = image

    def __hash__(self):
        return hash(self.image["file_name"])


def merge_annotations(coco1, coco2, coco_merged):
    with open(coco1, "r") as f1, open(coco2, "r") as f2, open(coco_merged, "w") as f3:
        coco1 = json.load(f1)
        coco2 = json.load(f2)

        # Merge images
        coco1_image_set = set(map(lambda x: CocoImage(x), coco1["images"]))
        coco2_image_set = set(map(lambda x: CocoImage(x), coco2["images"]))
        images_all = coco1_image_set | coco2_image_set
        images_dups = coco1_image_set & coco2_image_set

        print(f"There are {len(coco1_image_set)} images in coco1 and {len(coco2_image_set)} images in coco2")
        print(f"Overlaps: {len(images_dups)}, total: {len(images_all)}")

        if len(images_dups) > 0:
            print(f"There are {len(images_dups)} duplicate images in coco1 and coco2, I don't know how to proceed")
            return

        # Find the highest image ID in the first dataset then shift all of the images in the second dataset by that much
        max_id = max(map(lambda x: x["id"], coco1["images"]))
        for image in coco2["images"]:
            image["id"] += max_id
        for annotations in coco2["annotations"]:
            annotations["image_id"] += max_id

        # Append shifted images and annotations to the first dataset
        coco1["images"].extend(coco2["images"])
        coco1["annotations"].extend(coco2["annotations"])

        json.dump(coco1, f3, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco1", type=str, required=True)
    parser.add_argument("--coco2", type=str, required=True)
    parser.add_argument("--coco_merged", type=str, required=True)
    args = parser.parse_args()

    merge_annotations(args.coco1, args.coco2, args.coco_merged)
