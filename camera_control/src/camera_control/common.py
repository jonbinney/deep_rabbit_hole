import cv2
from dataclasses import dataclass
from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QImage

@dataclass
class DetectionParameters:
    min_area: int
    blur_size: int
    background_subtraction_method: str
    background_subtraction_learning_rate: float

class CameraControlSignals(QObject):
    frame_updated = Signal(QImage)
    frame_number_updated = Signal(int)
    detection_parameters_updated = Signal(DetectionParameters)
    detection_completed = Signal(QImage)
    detection_mask_updated = Signal(QImage)
    annotations_updated = Signal(list, list)

def cv_image_to_q_image(cv_image):
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    height, width, channels = cv_image_rgb.shape
    bytes_per_line = channels * width
    q_image = QImage(cv_image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return q_image
