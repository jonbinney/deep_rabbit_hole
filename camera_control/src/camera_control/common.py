import cv2
from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QImage

class CameraControlSignals(QObject):
    frame_updated = Signal(QImage)
    frame_number_updated = Signal(int)
    detection_completed = Signal(QImage)
    detection_mask_updated = Signal(QImage)

def cv_image_to_q_image(cv_image):
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    height, width, channels = cv_image_rgb.shape
    bytes_per_line = channels * width
    q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return q_image
