"""
Script to run inference on a video file and generate annotations for detected objects.
"""
import argparse
import cv2
import json

def detect_objects(model, frame):
    # Hardcoded for now
    return ((0, 0, 100, 100),)

def test_model(video_path, annotation_path):
    print("running")
    # Open video file
    video = cv2.VideoCapture(video_path)

    # Open annotation file
    annotations = []

    # Load model
    model = None
    #model = Model()
    #model.load_state_dict(torch.load('/path/to/model/weights.pth'))
    #model.eval()

    # Process frames
    frame_count = 0
    coco_id = 0
    bboxes = []
    propagate_annotations = True  # Whether to propagate annotations to subsequent frames

    while video.isOpened():
        # Read frame
        ret, frame = video.read()
        if not ret:
            break

        print(".", end="")

        keyframe = False

        # Run inference on one frame every 250
        if frame_count % 10 == 0:
            # Preprocess frame
            preprocessed_frame = frame

            # Perform inference
            bboxes = detect_objects(model, preprocessed_frame)
            keyframe = True

        # Write annotations
        if propagate_annotations or frame_count % 10 == 0:
            # NOTE: if propagate_annotations is true, we write the
            # annotation even if no detection was done, using the last produced one
            for bbox in bboxes:
                x, y, width, height = bbox
                coco_annotation = {
                    'image_id': frame_count,
                    'id': coco_id,
                    'category_id': 1,
                    'bbox': [x, y, width, height],
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
        'images': list(map(lambda x: {'id': x['image_id'], 'file_name': f"frame_{x['image_id']}.PNG" }, annotations)),
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