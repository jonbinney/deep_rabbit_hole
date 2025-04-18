"""
This file is used to take a video file and an annotations file in JSON format
and convert it into an annotated video file. The annotated video file will
contain visualizations of the annotations for human visualization.

Usage:
  python visualize.py -v <video_path> -a <annotations_path>

Arguments:
  -v, --video: Path to the video file
  -a, --annotations: Path to the annotation file in JSON format.  Appears in green in the video
  -g, --ground_truth_annotations: Path to the ground truth annotation file in JSON format (optional). Appears in red in the video
  -d, --description: Path to the description file in text format (optional)

Output:
  An annotated video file named 'annotated_video.mp4' will be created in the
  current directory.

Example:
  python visualize.py -v video.mp4 -a annotations.json
"""
from pycocotools.coco import COCO
import cv2
import argparse
from collections import defaultdict

def draw_annotations(frame, frame_annotations, score_per_track, color):
  for annotation in frame_annotations:
    # Extract the bounding box coordinates
    x, y, w, h = [int(val) for val in annotation['bbox']]
    track_id = annotation['attributes']['track_id']

    # Get the score for the track_id, and if not found (since we don't get it when doing tracking, only detection), use the last known one
    score = annotation['attributes'].get('score', score_per_track.get(track_id, None))
    score_per_track[track_id] = score

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), color, 2)
    cv2.putText(frame, f"{track_id} {score if score else ''}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Create an argument parser
parser = argparse.ArgumentParser(description='Object Tracker Visualizer')
# Add the video path argument
parser.add_argument('-v', '--video', type=str, required=True, help='Path to the video file')
# Add the annotation path argument
parser.add_argument('-a', '--annotations', type=str, required=True, help='Path to the annotation file in JSON format')
parser.add_argument('-g', '--ground_truth_annotations', type=str, required=False, help='Path to the ground truth annotation file in JSON format')
# Parse the command line arguments
parser.add_argument('-d', '--description', type=str, required=False, help='Path to the description file in text format')
args = parser.parse_args()

# Get the video path from the command line arguments
video_path = args.video
# Get the annotation path from the command line arguments
annotation_path = args.annotations

gt_annotation_path = args.ground_truth_annotations


# Define the output video path
# TODO: Make this also an argument
output_path = 'annotated_video.mp4'
SUBTITLE_DURATION = 90

# Load the video
video = cv2.VideoCapture(video_path)

# Load the annotations from the JSON file
coco = COCO(annotation_path)

coco_gt = None
if gt_annotation_path:
  coco_gt = COCO(gt_annotation_path)

print(f"Video Path: {video_path}")

# get the fps from video
fps = int(video.get(cv2.CAP_PROP_FPS))

# Get the video width and height
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create the VideoWriter object
output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

descriptions_per_frame = defaultdict(lambda: [])
if args.description:
  with open(args.description, 'r') as f:
    for line in f.read().split("\n"):
      if line.strip() == "":
        continue

      frame, actor, action = line.split(";")
      descriptions_per_frame[int(frame)].append((actor, action))

current_descriptions_per_actor = {}
score_per_track = {}

# Iterate over each frame in the video
while True:
  # Read the next frame
  ret, frame = video.read()

  # Break if no more frames are available
  if not ret:
    break

  # Get the frame number
  frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

  # Get the annotations for the current frame
  frame_annotations = coco.loadAnns(coco.getAnnIds(imgIds=frame_number))

  draw_annotations(frame, score_per_track, frame_annotations, (0, 255, 0)) # green

  if coco_gt:
    frame_annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=frame_number))
    draw_annotations(frame, frame_annotations, {}, (0, 0, 255)) # red

  # Add new descriptions (may override existing one if it's the same actor)
  for actor, action in descriptions_per_frame[frame_number]:
    current_descriptions_per_actor[actor] = (f"{actor} {action}", frame_number + SUBTITLE_DURATION)

  # Remove expired descriptions
  current_descriptions_per_actor = dict(filter(lambda x: x[1][1] > frame_number, current_descriptions_per_actor.items()))

  description = ". ".join(map(lambda x: x[0], current_descriptions_per_actor.values()))
  if description:
    cv2.putText(frame, f"{description}", (100, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

  # Write the annotated frame to the output video
  output_video.write(frame)

  if frame_number % 100 == 0:
    print(".", end="")

print("DONE")

# Release the video capture and close all windows
video.release()
output_video.release()
cv2.destroyAllWindows()