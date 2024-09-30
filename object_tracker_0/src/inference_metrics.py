import argparse
from collections import defaultdict
import json
import numpy as np

def compute_confusion_matrix(ground_truth: str, predictions: str, iou_threshold: float):
    def group_annotations(annotations):
        annotations_per_image = defaultdict(lambda: [])
        for annotation in annotations:
            image_id = annotation['image_id']
            annotations_per_image[image_id].append(annotation)
        return annotations_per_image


    with open(ground_truth, 'r') as f:
        ground_truth_data = json.load(f)

    with open(predictions, 'r') as f:
        predictions_data = json.load(f)

    if len(ground_truth_data['images']) != len(predictions_data['images']):
        raise ValueError("Number of images in ground truth and predictions do not match")


    ground_truth_per_image = group_annotations(ground_truth_data['annotations'])
    print(f"Total annotations in predictions: {len(predictions_data['annotations'])}")

    predictions_per_image = group_annotations(predictions_data['annotations'])
    print(f"Total annotations in ground truth: {len(ground_truth_data['annotations'])}")

    ids = [gt['id'] for gt in ground_truth_data['images']]
    cm = np.zeros((2, 2), np.int32)

    # this is for debugging purposes
    images_with_fp = []
    images_with_fn = []

    for id in ids:
        cm_image = compute_confusion_matrix_per_image(ground_truth_per_image[id], predictions_per_image[id], iou_threshold)
        cm += cm_image

        # For debugging purposes
        if cm_image[0, 1] > 0:
            images_with_fp.append(id)

        if cm_image[1, 0] > 0:
            images_with_fn.append(id)


    print(f"Images with FP: {images_with_fp}")
    print(f"Images with FN: {images_with_fn}")

    return cm

def compute_confusion_matrix_per_image(ground_truth: list, predictions: list, iou_threshold: float):
    def group_by_category(annotations):
        annotations_per_category = defaultdict(lambda: [])
        for annotation in annotations:
            category_id = annotation['category_id']
            annotations_per_category[category_id].append(annotation)
        return annotations_per_category


    ground_truth_per_category = group_by_category(ground_truth)
    predictions_per_category = group_by_category(predictions)

    all_categories = set(ground_truth_per_category.keys()) | set(predictions_per_category.keys())
    cm = np.zeros((2, 2), np.int32)

    for category in all_categories:
        cm += compute_confusion_matrix_per_image_one_category(ground_truth_per_category[category], predictions_per_category[category], iou_threshold)

    return cm

# Boxes must be in the format [x, y, width, height]
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_end = x1 + w1
    y1_end = y1 + h1
    x2_end = x2 + w2
    y2_end = y2 + h2

    intersection = max(0, min(x1_end, x2_end) - max(x1, x2)) * max(0, min(y1_end, y2_end) - max(y1, y2))

    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - intersection
    return intersection / union

# returns a confusion matrix in the form: [[TP, FP], [FN, TN]] (TN is always 0)
def compute_confusion_matrix_per_image_one_category(ground_truths: list, predictions: list, iou_threshold: float):
    # Compute the IoU for every pair of ground truth and prediction, and store it in a map from (gt_id, pred_id) to IoU
    pair_to_iou = {}
    for gt in ground_truths:
        for pred in predictions:
            iou = compute_iou(gt['bbox'], pred['bbox'])
            if iou > iou_threshold:
                pair_to_iou[(gt['id'], pred['id'])] = iou

    gt_ids = set(gt['id'] for gt in ground_truths)
    pred_ids = set(pred['id'] for pred in predictions)

    # Iterate through the pairs in descending order of IoU, and if both elements are still in the set of ids, it's a true positive.
    tp = 0
    sorted_pairs = sorted(pair_to_iou.keys(), key=lambda x: x[1], reverse=True)
    for gt, pred in sorted_pairs:
        if gt in gt_ids and pred in pred_ids:
            tp += 1

        gt_ids.discard(gt)
        pred_ids.discard(pred)


    # The elements left in pred_ids are false positives, and the elements left in gt_ids are false negatives
    return np.array([[tp, len(pred_ids)], [len(gt_ids), 0]], np.int32)

def compute_metrics(confusion_matrix):
    precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    f1 = 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Tracker')

    parser.add_argument('-g', '--ground_truth', type=str, required=True, help='Path to the ground truth JSON file')
    parser.add_argument('-p', '--predictions', type=str, required=True, help='Path to the predictions JSON file')
    parser.add_argument('-t', '--iou_thresholds', type=float, default=0.5, help='IOU threshold for computing confusion matrix')

    args = parser.parse_args()

    cm = compute_confusion_matrix(args.ground_truth, args.predictions, args.iou_thresholds)
    print(f"Confusion matrix: \n{cm}")
    precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1       : {f1:.2f}")
