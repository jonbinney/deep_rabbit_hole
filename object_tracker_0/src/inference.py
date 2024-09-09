"""
Script to run inference on a video file and generate annotations for detected objects.
This is the same as inference_gd.py, but with the groundingdino dependency removed and using
the Hugging Face Transformer library instead.
"""
import math
import torch
import argparse
import cv2
import json
# import groundingdino.datasets.transforms as T
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_model():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 1200, "longest_edge": 2000})
    #processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return (model, processor)

# This function is used to convert from the bbox format used
# by the model (percentual cxcywh) to the format used by the COCO dataset (xywh)
# TODO: Use torch.transform functions (see torchvision.ops.box_convert)
def convert_bbox_format(bbox, shape):
    h, w, _ = shape
    cx, cy, cw, ch = bbox.cpu().numpy()
    return [
        (cx * w) - (cw * w) / 2,
        (cy * h) - (ch * h) / 2,
        cw * w,
        ch * h
    ]

# Converts from Pascal VOC bbox format (x_min, y_min, x_max, y_max)
# to COCO bbox format (x, y, width, height)
def convert_bbox_format_2(bbox, shape):
    x0, y0, x1, y1 = bbox
    return [ x0, y0, x1 - x0, y1 - y0 ]

def is_similar(box1, box2, threshold=10):
    return all([abs(box1[i] - box2[i]) < threshold for i in range(4)])

def calculate_croppings(pil_frame, sub_frame_height, sub_frame_width):
    croppings = []
    step_height = (pil_frame.height - sub_frame_height) / (math.ceil(pil_frame.height / sub_frame_height) - 1)
    step_width = (pil_frame.width - sub_frame_width) / (math.ceil(pil_frame.width / sub_frame_width) - 1)
    for top in range(0, int(pil_frame.height - sub_frame_height) + 1, int(step_height)):
        for left in range(0, int(pil_frame.width - sub_frame_width) + 1, int(step_width)):
            croppings.append((left, top))
    return croppings

def detect_objects(model, processor, frame, tiling):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame)

    if tiling:

        # Split frame into subframes that match the default model input size
        sub_frame_height = 800
        sub_frame_width = 1200

        croppings = calculate_croppings(pil_frame, sub_frame_height, sub_frame_width)

        bboxes = []
        for left, top in croppings:
            sub_frame = pil_frame.crop((left, top, left + sub_frame_width, top + sub_frame_height))
            more_boxes = do_detection_on_frame(model, processor, sub_frame)
            more_boxes = list(map(lambda bbox: (bbox[0] + left, bbox[1] + top, bbox[2], bbox[3]), more_boxes))
            # Remove duplicated boxes (from overlapping subframes)
            more_boxes = filter(lambda newbbox: not any(is_similar(newbbox, oldbbox) for oldbbox in bboxes), more_boxes)
            bboxes.extend(more_boxes)

    else:
        bboxes = do_detection_on_frame(model, processor, pil_frame)

    return bboxes

def do_detection_on_frame(model, processor, frame):
    inputs = processor(images=frame, text="rabbit.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.15,
        text_threshold=0.15,
        target_sizes=[frame.size[::-1]],
    )

    return map(lambda bbox: convert_bbox_format_2(bbox, frame.size), results[0].get('boxes', []).cpu())

def test_model(video_path, annotation_path, skip_frame_count=10, tiling=False):
    # Open video file
    video = cv2.VideoCapture(video_path)

    # Open annotation file
    annotations = []

    # Load model
    (model, processor) = load_model()

    # Process frames
    frame_count = 0
    coco_id = 1
    bboxes = []
    propagate_annotations = True  # Whether to propagate annotations to subsequent frames

    while video.isOpened():
        # Read frame
        ret, frame = video.read()
        if not ret:
            break

        print(".", end="")

        keyframe = False

        # Run inference only every skip_frame_count frames (default to 10)
        if frame_count % skip_frame_count == 0:
            # Preprocess frame
            preprocessed_frame = frame

            # Perform inference
            bboxes = list(detect_objects(model, processor, preprocessed_frame, tiling))
            keyframe = True
            print(f"Det 1: {list(bboxes)}")

        # Write annotations
        if propagate_annotations or frame_count % skip_frame_count == 0:
            # NOTE: if propagate_annotations is true, we write the
            # annotation even if no detection was done, using the last produced one
            for bbox in bboxes:
                x, y, width, height = bbox
                coco_annotation = {
                    'image_id': frame_count + 1,
                    'id': coco_id,
                    'category_id': 1,
                    'bbox': [float(x), float(y), float(width), float(height)],
                    'attributes': { 'keyframe': keyframe },
                    'iscrowd': 0,
                    'segmentation': []
                }
                annotations.append(coco_annotation)
                coco_id += 1

        frame_count += 1

    # Save results to JSON file
    coco = {
        'categories': [
            {'id': 1, 'name': 'rabbit'}
        ],
        'images': [{'id': x + 1, 'file_name': f"frame_{x:06d}.PNG" } for x in range(frame_count)],
        'annotations': annotations
    }
    with open(annotation_path, 'w') as f:
        f.write(json.dumps(coco))

    # Release video file
    video.release()

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Object Tracker')

    # Add video path argument
    parser.add_argument('-v', '--video', type=str, required=True, help='Path to the video file')
    # Add annotations json path argument
    parser.add_argument('-a', '--annotations', type=str, required=True, help='Path to the output annotations JSON file')

    # Parse arguments
    args = parser.parse_args()

    # Call test_model function with video_path and json_path arguments
    test_model(args.video, args.annotations)