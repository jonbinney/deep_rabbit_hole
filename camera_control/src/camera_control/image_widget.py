from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPixmap, QPainter, QPen
from PySide6.QtWidgets import QLabel


class ImageWidget(QLabel):
    """
    A widget to display an image, resizing it as the window size changes.
    """
    image_clicked_signal = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_pixmap = None

        # bboxes are of the form (x, y, width, height)
        self.box_annotations = []
        # Circles are of the form (x, y, radius)
        self.circle_annotations = []

    def display_image(self, q_img):
        self.raw_pixmap = QPixmap.fromImage(q_img)
        # trigger a paintevent
        self.update()
    
    def set_annotations(self, box_annotations, circle_annotations):
        self.box_annotations = box_annotations
        self.circle_annotations = circle_annotations
        self.update()

    def paintEvent(self, event):
        if self.raw_pixmap is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(self.rect(), self.raw_pixmap)    
            
            # Draw bounding boxes
            x_scale_factor = self.width() / self.raw_pixmap.width()
            y_scale_factor = self.height() / self.raw_pixmap.height()
            painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
            for bbox in self.box_annotations:
                x, y, width, height = bbox
                # Draw the bounding box
                painter.drawRect(x * x_scale_factor, y * y_scale_factor, width * x_scale_factor, height * y_scale_factor)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            for circle in self.circle_annotations:
                x, y, radius = circle
                painter.drawEllipse(QPointF(x * x_scale_factor, y * y_scale_factor), radius * x_scale_factor, radius * y_scale_factor)
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