from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import QLabel


class ImageWidget(QLabel):
    """
    A widget to display an image, resizing it as the window size changes.
    """
    image_clicked_signal = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_pixmap = None

    def display_image(self, q_img):
        self.raw_pixmap = QPixmap.fromImage(q_img)
        # trigger a paintevent
        self.update()

    def paintEvent(self, event):
        if self.raw_pixmap is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(self.rect(), self.raw_pixmap)
            painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event):
            if self.raw_pixmap is not None and event.button() == Qt.LeftButton:
                # Calculate the coordinates in the image
                x_ratio = self.raw_pixmap.width() / self.width()
                y_ratio = self.raw_pixmap.height() / self.height()
                x = int(event.pos().x() * x_ratio)
                y = int(event.pos().y() * y_ratio)
                self.image_clicked_signal.emit(x, y)
            super().mousePressEvent(event)