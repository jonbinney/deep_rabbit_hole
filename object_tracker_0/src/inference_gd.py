"""
Script to run inference on a video file and generate annotations for detected objects.
It uses Grounding DINO code directly.
See inference.py for a version that uses the Hugging Face Transformer library instead.

The following is working:
 - It runs on Julian's VSCode using a python debug config
 - It successfully loads Grounding DINO and executes on Julian's CUDA
 - It detects some rabbits!
 - The written annotations file is correct and can be successfully imported in CVAT.ai
 - The annotation bbox format is correctly converted from the model's format to COCO's format

Bonus:
 - There's a test for the bbox conversion function and Julian could run it from VSCode

Next up:
 - Perform fine-tuning?
 - Implement a tracker?
 - Clean-up / organize the code properly?
 - Likewise organize datasets properly?
 - Figure out how to benchmark detection results vs. manual labels?
"""
import argparse
import cv2
import json
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
from PIL import Image


# This function is used to convert from the bbox format used
# by the model (percentual cxcywh) to the format used by the COCO dataset (xywh)
# TODO: Use torch.transform functions (see torchvision.ops.box_convert)
def convert_bbox_format(bbox, shape):
    h, w, _ = shape
    cx, cy, cw, ch = bbox.numpy()
    return [
        (cx * w) - (cw * w) / 2,
        (cy * h) - (ch * h) / 2,
        cw * w,
        ch * h
    ]

def prepare_frame(frame):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    prepared_frame, _ = transform(frame, None)
    return prepared_frame

def detect_objects(model, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame)
    frame_tensor = prepare_frame(pil_frame)
    boxes, logits, phrases = predict(
        model=model,
        image=frame_tensor,
        caption="rabbit",
        box_threshold=0.15,
        text_threshold=0.15
    )
    return map(lambda bbox: convert_bbox_format(bbox, frame.shape), boxes)

def test_model(video_path, annotation_path):
    # Open video file
    video = cv2.VideoCapture(video_path)

    # Open annotation file
    annotations = []

    # Load model
    model = load_model("external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "models/weights/groundingdino_swint_ogc.pth")

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
            bboxes = list(detect_objects(model, preprocessed_frame))
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
        'images': list(map(lambda x: {'id': x['image_id'], 'file_name': f"frame_{x['image_id'] - 1:06d}.PNG" }, annotations)),
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