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

def test_calculate_croppings():
    from inference import calculate_croppings
    from PIL import Image
    import numpy as np
    pil_frame = Image.fromarray(np.zeros((2048, 3072, 3), dtype=np.uint8))
    sub_frame_height = 800
    sub_frame_width = 1200
    croppings = calculate_croppings(pil_frame, sub_frame_height, sub_frame_width)
    assert len(croppings) == 9, f"Expected 9 croppings, got {len(croppings)}"

def test_is_similar():
    from inference import is_similar
    box1 = [1135.3022, 401.4587, 50.4141, 73.8211]
    box2 = [1135.0023, 399.2978, 50.8004, 76.1039]
    assert is_similar(box1, box2), f"Expected {box1} and {box2} to be similar"