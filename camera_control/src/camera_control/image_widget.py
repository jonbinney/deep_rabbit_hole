from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtCore import Qt

class ImageWidget(QLabel):
    """
    A widget to display an image, resizing it as the window size changes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_pixmap = None

    def display_image(self, q_img):
        self.raw_pixmap = QPixmap.fromImage(q_img)

    def paintEvent(self, event):
        if self.raw_pixmap is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(self.rect(), self.raw_pixmap)
            painter.end()
        super().paintEvent(event)
