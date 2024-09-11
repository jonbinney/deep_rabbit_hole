import pytest
import numpy as np
from inference_metrics import compute_iou, compute_confusion_matrix_per_image_one_category, compute_confusion_matrix_per_image, compute_confusion_matrix
from collections import defaultdict
import json
import tempfile

@pytest.fixture
def mock_data():
    ground_truth = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]}, 
        {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [10, 10, 10, 10]},
        {'id': 3, 'image_id': 1, 'category_id': 3, 'bbox': [20, 20, 10, 10]} 
    ]
    predictions = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [1, 1, 9, 10]},
        {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [10, 12, 10, 10]},
        {'id': 3, 'image_id': 1, 'category_id': 1, 'bbox': [20, 20, 10, 10]}
    ]
    return ground_truth, predictions

def test_compute_iou():
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 10, 10]
    iou = compute_iou(box1, box2)
    expected_iou = 25 / 175  # Intersection area / Union area
    assert np.isclose(iou, expected_iou)

    # No intersection
    box2 = [20, 20, 10, 10]
    iou = compute_iou(box1, box2)
    expected_iou = 0  # Intersection area / Union area
    assert np.isclose(iou, expected_iou)

    box2 = [1, 0, 10, 10]
    iou = compute_iou(box1, box2)
    expected_iou = 90 / 110  # Intersection area / Union area
    assert np.isclose(iou, expected_iou)
   
    box2 = [9, 9, 10, 10]
    iou = compute_iou(box1, box2)
    expected_iou = 1 / 199  # Intersection area / Union area
    assert np.isclose(iou, expected_iou)

def test_compute_confusion_matrix_per_image_one_category(mock_data):
    ground_truth, predictions = mock_data

    expected_cms = [
        np.array([[1, 1], [0, 0]], np.int32),
        np.array([[1, 0], [0, 0]], np.int32),
        np.array([[0, 0], [1, 0]], np.int32)
    ]
    for category_id in [1,2,3]:
        gt = [gt for gt in ground_truth if gt['category_id'] == category_id]
        pred = [pred for pred in predictions if pred['category_id'] == category_id]
        cm = compute_confusion_matrix_per_image_one_category(gt, pred, 0.5)

        expected_cm = expected_cms.pop(0)
        np.testing.assert_array_equal(cm, expected_cm)

def test_compute_confusion_matrix_per_image(mock_data):
    ground_truth, predictions = mock_data
    cm = compute_confusion_matrix_per_image(ground_truth, predictions, 0.5)
    expected_cm = np.array([[2, 1], [1, 0]], np.int32) 
    np.testing.assert_array_equal(cm, expected_cm)

def test_compute_confusion_matrix(mock_data):
    ground_truth, predictions = mock_data
    with tempfile.NamedTemporaryFile('w', delete=False) as gt_file, tempfile.NamedTemporaryFile('w', delete=False) as pred_file:
        json.dump({'images': [{'id': 1}], 'annotations': ground_truth}, gt_file)
        json.dump({'images': [{'id': 1}], 'annotations': predictions}, pred_file)
        gt_file.flush()
        pred_file.flush()
        cm = compute_confusion_matrix(gt_file.name, pred_file.name, 0.5)
        expected_cm = np.array([[2, 1], [1, 0]], np.int32) 
        np.testing.assert_array_equal(cm, expected_cm)