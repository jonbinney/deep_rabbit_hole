import argparse
import cv2
from pathlib import Path
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QTabWidget,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt
import sys
import yaml

from camera_control.common import CameraControlSignals, cv_image_to_q_image
from camera_control.detection_control_widget import DetectionControlWidget, DetectionParameters
from camera_control.image_widget import ImageWidget
from camera_control.gstreamer_camera_capture import GStreamerCameraCapture

class CameraControlApp(QMainWindow):
    def __init__(self, camera_config, video_path):
        super().__init__()
        self.camera_uri = f"rtsp://{camera_config['camera_username']}:{camera_config['camera_password']}@" + \
            f"{camera_config['camera_address']}:{camera_config['camera_rtsp_port']}" + \
            f"/{camera_config['camera_rtsp_path']}"
        self.video_path = video_path

        if self.video_path is not None:
            self.from_video_file = True
        elif self.camera_uri is not None:
            self.from_video_file = False
        else:
            raise ValueError("Must provide either a camera URI or a video path")

        self.app_signals = CameraControlSignals()

        self.setWindowTitle("Camera Control")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # This label will display the video frames
        self.frame_display = ImageWidget(self)
        self.detection_mask_display = ImageWidget(self)

        default_detection_parameters = DetectionParameters(10, 11, "MOG2", 0.001)
        self.detection_control_widget = DetectionControlWidget(self.app_signals, self.video_path, default_detection_parameters)

        self.app_signals.frame_updated.connect(self.frame_display.display_image)
        self.app_signals.detection_completed.connect(self.frame_display.display_image)
        self.app_signals.detection_mask_updated.connect(self.detection_mask_display.display_image)
        self.app_signals.frame_number_updated.connect(self.detection_control_widget.set_frame_number)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.video_layout = QVBoxLayout()
        self.video_tabs = QTabWidget(self)
        self.video_tabs.addTab(self.frame_display, "Input Video")
        self.video_tabs.addTab(self.detection_mask_display, "Detection Mask")
        self.video_layout.addWidget(self.video_tabs)

        self.main_layout.addLayout(self.video_layout)
        self.side_panel_layout = QVBoxLayout()
        self.side_panel_layout.addWidget(self.detection_control_widget)
        self.main_layout.addLayout(self.side_panel_layout)

        if self.from_video_file:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.video_path}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.seek_slider = QSlider(Qt.Horizontal, self)
            self.seek_slider.setMinimum(0)
            self.seek_slider.setMaximum(self.total_frames-1)
            self.seek_slider.setValue(0)
            self.seek_slider.valueChanged.connect(self.seek_frame)
            self.video_layout.addWidget(self.seek_slider)

            self.seek_slider.setValue(0)
            # Manually trigger the first frame update.
            self.seek_slider.valueChanged.emit(0)
        else:
            self.camera_capture_thread = CameraCaptureThread(self.app_signals, self.camera_uri)
            self.camera_capture_thread.start()

    def seek_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        if ret:
            self.app_signals.frame_updated.emit(cv_image_to_q_image(frame))
            self.app_signals.frame_number_updated.emit(frame_number)
        else:
            print(f"Error reading frame {frame_number}")

    def closeEvent(self, event):
        self.camera_capture_thread.requestInterruption()
        self.camera_capture_thread.wait()
        super().closeEvent(event)


class CameraCaptureThread(QThread):
    def __init__(self, app_signals, camera_uri):
        super().__init__()
        self.app_signals = app_signals
        self.camera_streamer = GStreamerCameraCapture(camera_uri)

    def run(self):
        while not self.isInterruptionRequested():
            # Blocks until a new frame is available (I think).
            frame = self.camera_streamer.read_next_frame()
            self.app_signals.frame_updated.emit(cv_image_to_q_image(frame))

        self.camera_streamer.release()
           

def main():
    parser = argparse.ArgumentParser(description="Camera control application")
    parser.add_argument("--video-path", help="Path to the video file (if not using a camera)")
    args = parser.parse_args()

    camera_config = None
    if args.video_path is None:
        # Load camera configuration from a YAML file.
        config_file_path = Path.home() / ".config/camera_control.yml"
        try:
            with open(config_file_path) as yaml_file:
                camera_config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"No config file found at {config_file_path}.")
            sys.exit(-1)

    app = QApplication(sys.argv)
    main_window = CameraControlApp(camera_config, args.video_path)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
