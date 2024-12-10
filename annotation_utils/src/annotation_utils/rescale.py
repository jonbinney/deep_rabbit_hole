import argparse
import json


def adjust_frame_annotations(data, target_frames):
    original_frames = max((annotation["image_id"] + 1) for annotation in data["annotations"])
    if original_frames == target_frames:
        return data

    frame_ratio = target_frames / original_frames
    new_annotations = {}

    for annotation in data["annotations"]:
        new_image_id = int(annotation["image_id"] * frame_ratio)
        track_id = annotation.get("track_id", None)

        if new_image_id not in new_annotations:
            new_annotations[new_image_id] = []

        # Check if the annotation with the same track_id already exists
        if track_id is not None:
            if any(ann.get("track_id", None) == track_id for ann in new_annotations[new_image_id]):
                continue

        new_annotation = annotation.copy()
        new_annotation["image_id"] = new_image_id
        new_annotations[new_image_id].append(new_annotation)

    # Flatten the dictionary to a list of annotations
    data["annotations"] = [ann for anns in new_annotations.values() for ann in anns]

    # Adjust the images section
    data["images"] = [image for image in data["images"] if image["id"] < target_frames]

    return data


def rescale_annotations(input_file, output_file, scale_factor, target_frames):
    with open(input_file, "r") as f:
        data = json.load(f)

    if target_frames is not None:
        data = adjust_frame_annotations(data, target_frames)

    for annotation in data["annotations"]:
        bbox = annotation["bbox"]
        annotation["bbox"] = [coord * scale_factor for coord in bbox]

        if "area" in annotation:
            annotation["area"] *= scale_factor**2

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescale COCO annotations by a scaling factor.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input COCO annotation file.")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output COCO annotation file.")
    parser.add_argument("-s", "--scale_factor", type=float, required=True, help="Scaling factor for the resolution.")
    parser.add_argument(
        "-f", "--target_frames", type=float, required=True, help="How many frames should the new annotation have."
    )

    args = parser.parse_args()

    rescale_annotations(args.input_file, args.output_file, args.scale_factor, args.target_frames)
