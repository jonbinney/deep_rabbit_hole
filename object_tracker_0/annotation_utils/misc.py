# Various utility functions used to manipulate the pilecam annotations
import json
import datetime

# Removes images with names starting in 0-1728254877-
# IDK why those images ended up in the dataset
def remove_bad_images():
    with open('deep_water_level/data/annotations.json', 'r') as f:
        data = json.load(f)

    delete_ids = []
    for image in data['images']:
        if image['file_name'].startswith('pilenew/images/2024-10-06/0-1728254877-'):
            delete_ids.append(image['id'])
    
    # Remove the images if the ID is in delete_ids
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in delete_ids]
    data['images'] = [image for image in data['images'] if image['id'] not in delete_ids]
    
    with open('annotations_fixed.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Removed {len(delete_ids)} images")

# Dumps some annotations details to a CSV for easy inspection.
# Includes: file_name, timestamp, depth and transparency
def to_csv():
    with open('deep_water_level/data/annotations_filtered.json', 'r') as f:
        data = json.load(f)
    
    # Create a map of image_id to annotation
    image_id_to_annotation = {annotation['image_id']: annotation for annotation in data['annotations']}

    with open('deep_water_level/data/annotations.csv', 'w') as f:
        for image in data['images']:
            filename = image['file_name']
            timestamp_epoch = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
            # Convert to a date string in ISO format
            timestamp = datetime.datetime.fromtimestamp(timestamp_epoch).isoformat(sep=" ")
            attr = image_id_to_annotation.get(image['id'],{}).get('attributes', {})

            f.write(f"{filename},{timestamp},{attr.get('depth',-1)},{attr.get('transparency','n/a')}\n")

# Filters a COCO annotation file keeping only the images that have annotations
def filter_annotations():
    with open('deep_water_level/data/annotations.json', 'r') as f:
        data = json.load(f)

    print(f"Total images: {len(data['images'])}")

    # Create a map of image_id to annotation
    image_id_to_annotation = {annotation['image_id']: annotation for annotation in data['annotations']}

    # Remove the images if the ID is not in image_id_to_annotation
    data['images'] = [image for image in data['images'] if image['id'] in image_id_to_annotation]

    with open('deep_water_level/data/annotations_filtered.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Filtered images: {len(data['images'])}")

if __name__ == '__main__':
    # remove_bad_images()
    to_csv()
    # filter_annotations()