import argparse
from io import TextIOWrapper
import json
from collections import defaultdict

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

    def log(file: TextIOWrapper, image_id: int, actor: str, message: str):
        file.write(f"{image_id:5};{actor};{message}\n")
    

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

            for track_id, image_id_last_seen in last_seen.items():
                if image_id - image_id_last_seen > 32:
                    name = f"{categories_map[annotation_map[image_id][track_id]['category_id']]}_{track_id}"
                    # TODO: give more information of where it dissapeared
                    log(output_file, image_id, name,"disappeared")
                    last_seen.pop(track_id)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Tracker')

    parser.add_argument('-a', '--annotations', type=str, required=True, help='Path to the annotations JSON file')
    parser.add_argument('-d', '--description', type=str, required=True, help='Path to the output description annotations text file')
    args = parser.parse_args()

    describe_annotations(args.annotations, args.description)