import argparse
from io import TextIOWrapper
import json
from collections import defaultdict
import mlflow
import datetime

# Helper class that converts from an image ID to a timestamp in ISO format
# It is instantiated with a start_time and fps values and then
# the method to_timestamp can be called passing the image ID as an argument
# to get the timestamp in ISO format
class TimestampConverter:
    def __init__(self, start_time, fps):
        self.start_time = start_time
        self.fps = fps
        self.last_time = None

    def to_timestamp(self, image_id):
        this_time = self.start_time + datetime.timedelta(seconds=(image_id / self.fps))
        # If the date is different or it's the first time, then print the whole date
        timestamp_string = ""
        if self.last_time is None or self.last_time.date() != this_time.date():
            timestamp_string = this_time.strftime("%B %d at ")
        # But always print the time
        timestamp_string += this_time.strftime("%H:%M:%S")
        self.last_time = this_time
        return timestamp_string

def directions_of_movement(prev_x, prev_y, x, y, THRESHOLD = 1):
    dx = x - prev_x
    dy = y - prev_y

    adx = abs(dx)
    ady = abs(dy)

    horizontal = ""
    if (adx) > THRESHOLD:
        horizontal = "R" if dx > 0 else "L"

    vertical = ""
    if (ady) > THRESHOLD:
        vertical = "D" if dy > 0 else "U"

    # TODO: we want to show diagonal movements, but need to tune it so that we're not too sensitive to changes in direction
    # if adx > 2 * ady:
    #     vertical = ""

    # if ady > 2 * adx:
    #     horizontal = ""

    if adx > ady:
        vertical = ""
    else:
        horizontal = ""

    return (horizontal, vertical)

def describe_movement(h, v):
    d = []
    if h == "R":
        d.append("right")
    elif h == "L":
        d.append("left")

    if v == "D":
        d.append("down")
    elif v == "U":
        d.append("up")

    return " and ".join(d)

def describe_annotations(filename: str, output_filename: str):
    params = {f'describe_annotations/{param}': value for param, value in locals().items()}
    mlflow.log_params(params)
    mlflow.set_tag("Inference Info", "Find rabbits in video and track them using Grounding DINO and SAM2")

    with open(filename, 'r') as f:
        data = json.load(f)

    categories_map = {category['id']: category['name'] for category in data['categories']}

    annotation_map = defaultdict(lambda: {})

    for annotation in data['annotations']:
        track_id = annotation.get('attributes', {}).get('track_id', 0)
        annotation_map[annotation['image_id']][track_id] = {'category_id': annotation['category_id'], 'bbox': annotation['bbox']}

    min_image_id = min(annotation_map.keys())
    max_image_id = max(annotation_map.keys())

    position_map = {}
    last_seen = {}

    # Get video information and create timestamp converter with it
    video_timestap = data['info'].get('video_timestamp', None)
    if video_timestap is None:
        print(f"WARNING: No video_timestamp found in {filename}")
        video_timestap = datetime.datetime.now()
    fps = data['info'].get('fps', None)
    if fps is None:
        print(f"WARNING: No fps found in {filename}")
        fps = 30
    tsc = TimestampConverter(video_timestap, fps)

    def log(file: TextIOWrapper, image_id: int, actor: str, message: str):
        #file.write(f"{image_id:6};{tsc.to_timestamp(image_id)};{actor};{message}\n")
        file.write(f"At {tsc.to_timestamp(image_id)}, {actor} {message}\n")

    with open(output_filename, 'w') as output_file:

        for image_id in range(min_image_id, max_image_id + 1):
            for track_id, annotation in annotation_map[image_id].items():
                x = int(annotation['bbox'][0] + annotation['bbox'][2] / 2)
                y = int(annotation['bbox'][1] + annotation['bbox'][3] / 2)
                last_seen[track_id] = image_id

                name = f"{categories_map[annotation['category_id']]}_{track_id}"
                if track_id in position_map:
                    prev_x, prev_y, prev_image_id, prev_state = position_map[track_id]
                    distance = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
                    if distance > 20:
                        h, v = directions_of_movement(prev_x, prev_y, x, y)
                        movement = describe_movement(h, v)

                        state = f"MOVING_{h}{v}"
                        if prev_state != state:
                            log(output_file, image_id, name, f"is moving {movement}")
                        position_map[track_id] = (x,y, image_id, state)
                    else:
                        if image_id - prev_image_id > 32 and prev_state[:6] == "MOVING":
                            # log(image_id, f"{name} stopped moving")
                            position_map[track_id] = (x,y, image_id, "STILL")

                else:
                    position_map[track_id] = (x,y, image_id, "STILL")
                    # TODO: give more information of where it appeared, e.g. "appeared in the top left corner" or if it just popped up in the middle of the screen
                    log(output_file, image_id, name, "appeared")

            removed_tracks = []
            for track_id, image_id_last_seen in last_seen.items():
                if image_id - image_id_last_seen > 32:
                    name = f"{categories_map[annotation_map[image_id_last_seen][track_id]['category_id']]}_{track_id}"
                    # TODO: give more information of where it dissapeared
                    log(output_file, image_id, name,"disappeared")
                    removed_tracks.append(track_id)

            for removed_track in removed_tracks:
                last_seen.pop(removed_track)

    mlflow.log_artifact(output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Tracker')

    parser.add_argument('-a', '--annotations', type=str, required=True, help='Path to the annotations JSON file')
    parser.add_argument('-d', '--description', type=str, required=True, help='Path to the output description annotations text file')
    args = parser.parse_args()

    describe_annotations(args.annotations, args.description)
