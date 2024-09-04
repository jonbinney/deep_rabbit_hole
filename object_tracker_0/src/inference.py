"""
Script to run inference on a video file and generate annotations for detected objects.
This is the same as inference_gd.py, but with the groundingdino dependency removed and using
the Hugging Face Transformer library instead.
"""
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

    processor = AutoProcessor.from_pretrained(model_id)
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

def detect_objects(model, processor, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame)

    inputs = processor(images=pil_frame, text="rabbit.", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.15,
        text_threshold=0.15,
        target_sizes=[pil_frame.size[::-1]],
    )

    return map(lambda bbox: convert_bbox_format_2(bbox, frame.shape), results[0].get('boxes', []).cpu())

def test_model(video_path, annotation_path):
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

        # Run inference on one frame every 10
        if frame_count % 10 == 0:
            # Preprocess frame
            preprocessed_frame = frame

            # Perform inference
            bboxes = list(detect_objects(model, processor, preprocessed_frame))
            keyframe = True
            print(f"Detected objects: {list(bboxes)}")

        # Write annotations
        if propagate_annotations or frame_count % 10 == 0:
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