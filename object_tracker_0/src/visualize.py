"""
This file is used to take a video file and an annotations file in JSON format
and convert it into an annotated video file. The annotated video file will
contain visualizations of the annotations for human visualization.

Usage:
  python visualize.py -v <video_path> -a <annotations_path>

Arguments:
  -v, --video: Path to the video file
  -a, --annotations: Path to the annotation file in JSON format

Output:
  An annotated video file named 'annotated_video.mp4' will be created in the
  current directory.

Example:
  python visualize.py -v video.mp4 -a annotations.json
"""
from pycocotools.coco import COCO
import cv2
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Object Tracker Visualizer')
# Add the video path argument
parser.add_argument('-v', '--video', type=str, required=True, help='Path to the video file')
# Add the annotation path argument
parser.add_argument('-a', '--annotations', type=str, required=True, help='Path to the annotation file in JSON format')
# Parse the command line arguments
args = parser.parse_args()

# Get the video path from the command line arguments
video_path = args.video
# Get the annotation path from the command line arguments
annotation_path = args.annotations
# Define the output video path
# TODO: Make this also an argument
output_path = 'annotated_video.mp4'

# Load the video
video = cv2.VideoCapture(video_path)

# Load the annotations from the JSON file
coco = COCO(annotation_path)

print(f"Video Path: {video_path}")


# Get the video width and height
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create the VideoWriter object
output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

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

  # Draw the annotations on the frame
  for annotation in frame_annotations:
    # Extract the bounding box coordinates
    x, y, w, h = annotation['bbox']

    # Cast x and y to integer
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)

  # Write the annotated frame to the output video
  output_video.write(frame)

  if frame_number % 100 == 0:
    print(".", end="")

print("DONE")

# Release the video capture and close all windows
video.release()
output_video.release()
cv2.destroyAllWindows()