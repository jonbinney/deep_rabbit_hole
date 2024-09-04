from inference import convert_bbox_format
import torch

def test_convert_bbox_format():
    shape = (1280, 720, 3)
    bbox = torch.tensor([0.4906, 0.4586, 0.7900, 0.5276])
    expected_result = [69.54,243.86,560.76,680.64]
    result = convert_bbox_format(bbox, shape)
    # Assert all coordinates are within 5% of the expected result (manually labeled)
    threshold = max(shape) * 0.05
    assert all(abs(a - b) < threshold for a, b in zip(result, expected_result)), f"Expected {expected_result}, got {result}"
