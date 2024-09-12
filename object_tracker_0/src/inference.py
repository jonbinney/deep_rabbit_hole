"""
Script to do object tracking on a video using object detection with Grounding DINO and then Object Tracking with SAM
based on the boxes found.
It takes a video as input.
It first extracts frames from the video in a provided working directory. If the extraction was previously done, it uses the images from there.
Then it goes through the frames until it finds a rabbit. It uses the Grounding DINO model to detect objects in the frames.
When it finds a rabbit, it uses SAM 2 to track the rabbit in the video.

TODO:
 - Make it work
 - Convert the segment results into bboxes and save them in the COCO format
"""
import math
import os
import torch
import argparse
import cv2
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils import Timer
from datetime import datetime
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as np

# Create timers as globals
timer_inference = Timer()
timer_total = Timer()

def load_object_detection_model():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 1200, "longest_edge": 2000})
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    print(f"Using {device} for object detection with model: {model_id}")

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

def do_object_detection(model, processor, pil_frame, tiling = False):
    if tiling:

        # Split frame into subframes that match the default model input size
        sub_frame_height = 800
        sub_frame_width = 1200

        croppings = calculate_croppings(pil_frame, sub_frame_height, sub_frame_width)

        bboxes = []
        for left, top in croppings:
            sub_frame = pil_frame.crop((left, top, left + sub_frame_width, top + sub_frame_height))
            more_boxes = _do_detection_on_frame(model, processor, sub_frame)
            more_boxes = list(map(lambda bbox: (bbox[0] + left, bbox[1] + top, bbox[2], bbox[3]), more_boxes))
            # Remove duplicated boxes (from overlapping subframes)
            more_boxes = filter(lambda newbbox: not any(is_similar(newbbox, oldbbox) for oldbbox in bboxes), more_boxes)
            bboxes.extend(more_boxes)

    else:
        bboxes = _do_detection_on_frame(model, processor, pil_frame)

    return bboxes

def _do_detection_on_frame(model, processor, frame):
    inputs = processor(images=frame, text="rabbit.", return_tensors="pt").to(model.device)

    timer_inference.start()
    with torch.no_grad():
        outputs = model(**inputs)
    timer_inference.stop()

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.15,
        text_threshold=0.15,
        target_sizes=[frame.size[::-1]],
    )

    return map(lambda bbox: convert_bbox_format_2(bbox, frame.size), results[0].get('boxes', []).cpu().numpy())


def do_create_frame_files(video_path, frame_images_dir, force=False, resize=None, frame_end=None):
    # Check that the provided working directory exists and otherwise create it
    if not os.path.exists(frame_images_dir):
        os.makedirs(frame_images_dir)
        force = True

    # If the directory was just created or the force flag is set, create the frame files
    # from the video - otherwise, the existing frame files will be returned
    if force:
        print(f"Creating frame files from video: {video_path} into directory: {frame_images_dir}")
        # Open video file and save each frame as a jpg file
        video = cv2.VideoCapture(video_path)
        frame_count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # SAM 2 requires the frames to be in JPG format and be named <frame_number>.jpg
            file_name = f"{frame_images_dir}/{frame_count:06d}.jpg"
            if resize is not None:
                frame = cv2.resize(frame, resize)
            cv2.imwrite(file_name, frame)
            frame_count += 1
            if frame_end is not None and frame_count >= frame_end:
                break
    else:
        print(f"Reusing existing frame files from directory: {frame_images_dir}")

    # Read all the frame files available in the working directory
    frame_file_names = [f"{frame_images_dir}/{file_name}" for file_name in os.listdir(frame_images_dir) if file_name.endswith(".jpg")]
    # Sort by file name, which should correspond to frame name
    frame_file_names.sort()

    # Return a list of the frame file names
    return frame_file_names

def segments_to_bboxes(segments):
    """
    Converts the segments from the SAM2 model into bounding boxes.
    The return type is an object with the following structure:
    {
        frame_idx: {
            obj_id: bbox (x, y, w, h)
        }
    }
    """
    bboxes = {}
    for frame_idx, frame_segments in segments.items():
        bboxes[frame_idx] = {}
        for obj_id, mask in frame_segments.items():
            # NOTE(adamantivm) I don't know what the first dimension in masks is. I noted by debugging
            # that it had only one value when I was testing, so I'm assuming it's always 0.
            contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            bboxes[frame_idx][obj_id] = (x, y, w, h)
    return bboxes

def do_track_objects(frame_images_dir, bboxes, starting_frame=0, max_frames=10):
    """
    TODO:
     - Try lazy loader
     - Try offload to CPU
     - Peek at and play with image size
    """

    # For now, manually convert bbox to x_min, y_min, x_max, y_max and assume a single object
    (x, y, w, h) = bboxes[0]
    box = [x, y, x + w, y + h]

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-small")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_images_dir)

        _, _, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=starting_frame,
            obj_id=0,
            box=box)

        segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=starting_frame,
                max_frame_num_to_track=starting_frame + max_frames):
            segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        bboxes = segments_to_bboxes(segments)

        return bboxes

def do_create_annotations(bboxes):
    annotations = []
    annotation_id = 1
    for frame_idx, objects in bboxes.items():
        for obj_id, bbox in objects.items():
            x, y, w, h = bbox
            annotations.append({
                'id': annotation_id,
                'image_id': frame_idx + 1,
                'category_id': 1,  # Hardcoded to rabbit
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(w * h),
                'attributes': {
                    'track_id': obj_id,
                    'keyframe': True
                },
                'iscrowd': 0,
                'segmentation': []
            })
            annotation_id += 1
    return annotations

def perform_object_tracking(video_path, annotation_path, working_dir, frame_od_skip=10, tiling=False, frame_end=20):
    print(f"Performing object tracking on video: {video_path}")

    # Turn the input video into a directory with a in image file per frame
    frame_file_names = do_create_frame_files(video_path, working_dir, frame_end=frame_end)
    print(f"We have {len(frame_file_names)} frames to process")

    # Go through frames until we find a rabbit
    bboxes = []
    frame_number = 0
    while len(bboxes) == 0 and frame_number < len(frame_file_names):
        print(f"Looking for rabbits in frame {frame_number}")
        model, processor = load_object_detection_model()
        bboxes = list(do_object_detection(model, processor, Image.open(frame_file_names[frame_number]), tiling))
        if len(bboxes) == 0:
            frame_number += frame_od_skip

    print(f"Starting object tracking with {len(bboxes)} boxes at frame {frame_number}")

    bboxes = do_track_objects(working_dir, bboxes, frame_number)

    print(f"Boxes found: {bboxes}")

    annotations = do_create_annotations(bboxes)

    # Save results to JSON file
    # NOTE: Only to be able to upload to CVAT, we need to name images frame_<number>.png, without any path
    coco = {
        'info': {
            "description": "Object tracking results",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timings": f"Detection: {timer_total}, Inference: {timer_inference}"
        },
        'categories': [
            {'id': 1, 'name': 'rabbit'}
        ],
        'images': [{'id': i + 1, 'file_name': f"frame_{i:06d}.png" } for i, file_name in enumerate(frame_file_names)],
        'annotations': annotations
    }
    with open(annotation_path, 'w') as f:
        f.write(json.dumps(coco))

    print(f"Timings => Detection: {timer_total}, Inference: {timer_inference}")

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Object Tracker')

    # Add video path argument
    parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to the video file')
    # Add annotations json path argument
    parser.add_argument('-a', '--annotation_path', type=str, required=True, help='Path to the output annotations JSON file')
    # Add working directory argument
    parser.add_argument('-w', '--working_dir', type=str, default='.', help='Path to the working directory')

    # Parse arguments
    args = parser.parse_args()

    # Call test_model function with video_path and json_path arguments
    perform_object_tracking(**vars(args))