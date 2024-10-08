import argparse
import cv2
from enum import Enum
from pathlib import Path
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QMainWindow,
    QSlider,
    QTabWidget,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt
import random
import sys
import yaml
import time

from camera_control.common import CameraControlSignals, cv_image_to_q_image
from camera_control.detection_control_widget import DetectionControlWidget, DetectionParameters, ThingsOfInterestDetector
from camera_control.image_widget import ImageWidget
from camera_control.onvif_camera_controller import OnvifCameraController

class CameraControlApp(QMainWindow):
    def __init__(self, camera_config, video_path):
        super().__init__()
        self.app_signals = CameraControlSignals()

        self.video_path = video_path
        self.live_mode = True
        if self.video_path is not None:
            self.live_mode = False

        self.setWindowTitle("Camera Control")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # This label will display the video frames
        self.frame_display = ImageWidget(self)
        self.detection_mask_display = ImageWidget(self)

        default_detection_parameters = DetectionParameters(200, 11, "MOG2", 0.0001)
        self.detection_control_widget = DetectionControlWidget(self.app_signals, self.video_path, default_detection_parameters)

        self.app_signals.frame_updated.connect(self.frame_display.display_image)
        self.app_signals.detection_completed.connect(self.frame_display.display_image)
        self.app_signals.detection_mask_updated.connect(self.detection_mask_display.display_image)
        self.app_signals.frame_number_updated.connect(self.detection_control_widget.set_frame_number)
        self.app_signals.annotations_updated.connect(self.frame_display.set_annotations)

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

        if not self.live_mode:
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
    
        self.camera_capture_thread = None
        if self.live_mode:
            self.camera_capture_thread = CameraCaptureThread(self.app_signals, camera_config, default_detection_parameters)
            self.autozoom_button = QCheckBox("Autozoom", self)
            self.autozoom_button.stateChanged.connect(self.camera_capture_thread.update_autozoom)
            self.side_panel_layout.addWidget(self.autozoom_button)
            self.frame_display.image_clicked_signal.connect(self.camera_capture_thread.focus_on_point)
            self.app_signals.detection_parameters_updated.connect(self.camera_capture_thread.update_detection_parameters)
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
        if self.camera_capture_thread is not None:
            self.camera_capture_thread.requestInterruption()
            self.camera_capture_thread.wait()
        super().closeEvent(event)


class CameraCaptureThread(QThread):
    class PTZState(Enum):
        IDLE = 1
        TURNING = 2
        STOPPING_TURN = 3
        ZOOMING = 4
        FILMING_CLOSEUP = 5
        RETURNING_TO_HOME = 6

    def __init__(self, app_signals, camera_config, detection_parameters):
        super().__init__()
        self.app_signals = app_signals
        self.detection_parameters = detection_parameters
        self.autozoom_on = False
        camera_uri = f"rtsp://{camera_config['camera_username']}:{camera_config['camera_password']}@" + \
            f"{camera_config['camera_address']}:{camera_config['camera_rtsp_port']}" + \
            f"/{camera_config['camera_rtsp_path']}"
        self.camera_capture = cv2.VideoCapture(camera_uri)
        if not self.camera_capture.isOpened():
            raise ValueError(f"Failed to open camera stream: {camera_uri}")
        self.camera_controller = OnvifCameraController(
            camera_config["camera_address"],
            camera_config["camera_onvif_port"],
            camera_config["camera_username"],
            camera_config["camera_password"])
        
        self.ptz_state = self.PTZState.IDLE
        self.ptz_state_start_time = time.time()
        self.focus_point = None
        self.toi_detector = ThingsOfInterestDetector(self.detection_parameters)

    def run(self):
        idle_duration = 10.0  # seconds (detector needs time to learn the background)
        ptz_turn_duration = 2.0  # seconds
        ptz_stop_turn_duration = 0.5  # seconds (camera takes a moment to stop turning)
        ptz_zoom_duration = 1.5  # seconds
        ptz_filming_closeup_duration = 5.0  # seconds
        ptz_returning_to_home_duration = 10.0  # seconds
        x_speed_factor = 0.0008
        y_speed_factor = -0.0008

        previous_state = None
        frame_number = -1
        while not self.isInterruptionRequested():
            # Blocks until a new frame is available (I think).
            ret, frame = self.camera_capture.read()
            if not ret:
                raise ValueError("Failed to read frame from camera stream")
            frame_number += 1
            self.app_signals.frame_number_updated.emit(frame_number)
            image_height, image_width, _ = frame.shape
            self.app_signals.frame_updated.emit(cv_image_to_q_image(frame))

            if frame_number % 10 == 0:
                toi_bboxes = self.toi_detector.detect(frame)
                focus_circles = []
                if self.focus_point is not None:
                    focus_circles.append((self.focus_point[0], self.focus_point[1], 40))
                self.app_signals.annotations_updated.emit(toi_bboxes, focus_circles)

            if self.ptz_state != previous_state:
                print(f"PTZ State: {self.ptz_state.name}")
                previous_state = self.ptz_state
            if self.ptz_state == self.PTZState.IDLE:
                if self.autozoom_on:
                    idle_time = time.time() - self.ptz_state_start_time
                    if idle_time > idle_duration and len(toi_bboxes) > 0:
                        x, y, width, height = random.choice(toi_bboxes)
                        focus_x, focus_y = (x + width / 2, y + height / 2)
                        focus_circles.append((x + width / 2, y + height / 2, 40))
                        self.focus_on_point(focus_x, focus_y)
                        
                if self.focus_point is not None:
                    self.app_signals.annotations_updated.emit([], [])
                    x, y = self.focus_point
                    x_speed = ((x - image_width / 2) / ptz_turn_duration) * x_speed_factor
                    y_speed = ((y - image_height / 2) / ptz_turn_duration ) * y_speed_factor
                    print("Setting speed to ", x_speed, y_speed, 0.0)
                    self.camera_controller.relative_move(x_speed, y_speed, 0.0)
                    self.ptz_state = self.PTZState.TURNING
                    self.ptz_state_start_time = time.time()

            elif self.ptz_state == self.PTZState.TURNING:
                if time.time() - self.ptz_state_start_time >= ptz_turn_duration:
                    self.camera_controller.stop()
                    self.ptz_state = self.PTZState.STOPPING_TURN
                    self.ptz_state_start_time = time.time()

            elif self.ptz_state == self.PTZState.STOPPING_TURN:
                if time.time() - self.ptz_state_start_time >= ptz_stop_turn_duration:
                    self.camera_controller.relative_move(0.0, 0.0, 1.0)
                    self.ptz_state = self.PTZState.ZOOMING
                    self.ptz_state_start_time = time.time()

            elif self.ptz_state == self.PTZState.ZOOMING:
                if time.time() - self.ptz_state_start_time >= ptz_zoom_duration:
                    self.camera_controller.stop()
                    self.ptz_state = self.PTZState.FILMING_CLOSEUP
                    self.ptz_state_start_time = time.time()

            elif self.ptz_state == self.PTZState.FILMING_CLOSEUP:
                if time.time() - self.ptz_state_start_time >= ptz_filming_closeup_duration:
                    self.camera_controller.goto_preset("1")
                    self.ptz_state = self.PTZState.RETURNING_TO_HOME
                    self.ptz_state_start_time = time.time()

            elif self.ptz_state == self.PTZState.RETURNING_TO_HOME:
                if time.time() - self.ptz_state_start_time >= ptz_returning_to_home_duration:
                    self.toi_detector.reset()
                    self.focus_point = None
                    self.ptz_state = self.PTZState.IDLE
                    self.ptz_state_start_time = time.time()

        self.camera_capture.release()

    def focus_on_point(self, x, y):
        if self.focus_point is None:
            print(f"Setting focus point: ({x}, {y})")
            self.focus_point = (x, y)
        else:
            print("Ignoring focus point, already working on one.")

    def update_detection_parameters(self, detection_parameters):
        self.detection_parameters = detection_parameters
        self.toi_detector = ThingsOfInterestDetector(self.detection_parameters)

    def update_autozoom(self, autozoom_on):
        self.autozoom_on = autozoom_on

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
