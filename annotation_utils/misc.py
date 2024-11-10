# Various utility functions used to manipulate the pilecam annotations
import json
import datetime
import cv2

def bad_name(dataset_dir, image):
    """
    Function for filter_images
    Removes images with names starting in 0-1728254877-
    IDK why those images ended up in the dataset
    """
    return image['file_name'].startswith('pilenew/images/2024-10-06/0-1728254877-')

def too_little_contrast(dataset_dir, image):
    """
    Function for filter_images
    Removes images with low contrast
    """
    image_filename = image.get('file_name', None)
    if image_filename is not None:
        # use OpenCV to display the image and wait for a key
        img = cv2.imread(f'{dataset_dir}/images/{image_filename}')
        # Convert to grayscale and normalize
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray / 255.0
        # Calculate mean and standard deviation
        (means, stds) = cv2.meanStdDev(img_gray)
    return stds[0] < 0.071

def filter_images(eval_fn: callable=bad_name):
    """
    Removes images from a dataset according to the provided filter function
    """
    dataset_dir = 'datasets/water_2024_10_19_set1'
    input_annotations = f'{dataset_dir}/annotations/manual_annotations.json'
    output_annotations = f'{dataset_dir}/annotations/filtered.json'

    with open(input_annotations, 'r') as f:
        data = json.load(f)

    delete_ids = []
    for image in data['images']:
        if eval_fn(dataset_dir,image):
            delete_ids.append(image['id'])
    
    # Remove the images if the ID is in delete_ids
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in delete_ids]
    data['images'] = [image for image in data['images'] if image['id'] not in delete_ids]
    
    with open(output_annotations, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Filtered {len(delete_ids)} images")

def to_csv(
        input_json_path: str = 'deep_water_level/data/annotations.json',
        output_csv_path: str = 'deep_water_level/data/annotations.csv'
    ):
    """
    Dumps some annotations details to a CSV for easy inspection.
    Includes: file_name, timestamp, depth and transparency
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Create a map of image_id to annotation
    image_id_to_annotation = {annotation['image_id']: annotation for annotation in data['annotations']}

    with open(output_csv_path, 'w') as f:
        for image in data['images']:
            filename = image['file_name']
            timestamp_epoch = int(filename.split('/')[-1].split('.')[0].split('-')[-1])
            # Convert to a date string in ISO format
            timestamp = datetime.datetime.fromtimestamp(timestamp_epoch).isoformat(sep=" ")
            attr = image_id_to_annotation.get(image['id'],{}).get('attributes', {})

            f.write(f"{filename},{timestamp},{attr.get('depth',-1)},{attr.get('transparency','n/a')}\n")

def filter_annotations():
    """
    Filters a COCO annotation file keeping only the images that have annotations
    """
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

def explore_images():
    """
    Allows inspecting mean and standard deviation of images while displaying
    for visual inspection.
    """
    dataset_dir = 'datasets/water_test_set3'
    with open(f'{dataset_dir}/annotations/manual_annotations.json', 'r') as f:
        data = json.load(f)
    
    # Create a map of image_id to filename
    image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        image_filename = image_id_to_filename.get(image_id, None)
        if image_filename is not None:
            # use OpenCV to display the image and wait for a key
            img = cv2.imread(f'{dataset_dir}/images/{image_filename}')
            # Convert to grayscale and normalize
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = img_gray / 255.0
            # Calculate mean and standard deviation
            (means, stds) = cv2.meanStdDev(img_gray)

            print(f"Image ID: {image_id}, mean: {means[0]}, std: {stds[0]}")
            cv2.imshow('Image', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

    print(f"Total images: {len(data['images'])}")

if __name__ == '__main__':
    # remove_bad_images()
    to_csv(
        'datasets/water_test_set3/annotations/filtered.json',
        'datasets/water_test_set3/annotations/filtered.csv'
        )
    # filter_images(too_little_contrast)
    #explore_images()