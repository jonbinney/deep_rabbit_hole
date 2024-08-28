from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QImage

class CameraControlSignals(QObject):
    frame_updated = Signal(QImage)
    frame_number_updated = Signal(int)
    detection_completed = Signal(QImage)
    detection_mask_updated = Signal(QImage)