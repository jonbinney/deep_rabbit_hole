import argparse
import cv2
import sys
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QSlider,
    QTabWidget,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from camera_control.camera_control_signals import CameraControlSignals
from camera_control.detection_control_widget import DetectionControlWidget, DetectionParameters

class CameraControlApp(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

        self.app_signals = CameraControlSignals()

        self.setWindowTitle("Animal finder")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # This label will display the video frames
        self.frame_display = QLabel(self)
        self.detection_mask_display = QLabel(self)

        self.seek_slider = QSlider(Qt.Horizontal, self)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(100)
        self.seek_slider.setValue(0)
        self.seek_slider.valueChanged.connect(self.seek_frame)

        default_detection_parameters = DetectionParameters(10, 11, "MOG2", 0.001)
        self.detection_control_widget = DetectionControlWidget(self.app_signals, self.video_path, default_detection_parameters)

        self.app_signals.frame_updated.connect(
            lambda q_img: self.frame_display.setPixmap(QPixmap.fromImage(q_img))
        )
        self.app_signals.detection_completed.connect(
            lambda q_img: self.frame_display.setPixmap(QPixmap.fromImage(q_img))
        )
        self.app_signals.detection_mask_updated.connect(
            lambda q_img: self.detection_mask_display.setPixmap(QPixmap.fromImage(q_img))
        )
        self.app_signals.frame_number_updated.connect(self.detection_control_widget.set_frame_number)

        self.running = True
                
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.layout = QHBoxLayout(self.central_widget)
        self.video_layout = QVBoxLayout()
        self.video_tabs = QTabWidget(self)
        self.video_tabs.addTab(self.frame_display, "Input Video")
        self.video_tabs.addTab(self.detection_mask_display, "Detection Mask")
        self.video_layout.addWidget(self.video_tabs)
        self.video_layout.addWidget(self.seek_slider)
        self.layout.addLayout(self.video_layout)
        self.side_panel_layout = QVBoxLayout()
        self.side_panel_layout.addWidget(self.detection_control_widget)
        self.layout.addLayout(self.side_panel_layout)

        self.seek_slider.setValue(0)
        # Manually trigger the first frame update.
        self.seek_slider.valueChanged.emit(0)

    def seek_frame(self, value):
        frame_number = int((value / 100) * self.total_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, num_channels = frame.shape
            bytes_per_line = num_channels * width
            q_img = QImage(
                frame.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            self.app_signals.frame_updated.emit(q_img)
            self.app_signals.frame_number_updated.emit(frame_number)
        else:
            print(f"Error reading frame {frame_number}")

    def closeEvent(self, event):
        self.running = False
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Animal finder application")
    parser.add_argument("video_path", help="Path to the video file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    main_window = CameraControlApp(args.video_path)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
