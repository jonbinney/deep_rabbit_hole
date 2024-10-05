import cv2
import imutils
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
    QVBoxLayout,
)

from common import cv_image_to_q_image, DetectionParameters

class DetectionControlWidget(QWidget):
    def __init__(self, app_signals, video_path, default_parameters):
        super().__init__()
        self.app_signals = app_signals
        self.video_path = video_path
        self.frame_number = 0
        self.detection_thread = None

        self.main_layout = QVBoxLayout(self)
        self.frame_number_label = QLabel(f"Frame: {self.frame_number:9d}", self)
        self.main_layout.addWidget(self.frame_number_label)
        self.min_area_label = QLabel("Minimum area")
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(0, 1000000)
        self.min_area_spinbox.setValue(default_parameters.min_area)
        self.main_layout.addWidget(self.min_area_label)
        self.main_layout.addWidget(self.min_area_spinbox)
        self.blur_size_label = QLabel("Blur size")
        self.blur_size_spinbox = QSpinBox()
        self.blur_size_spinbox.setRange(1, 1000)
        self.blur_size_spinbox.setValue(default_parameters.blur_size)
        self.main_layout.addWidget(self.blur_size_label)
        self.main_layout.addWidget(self.blur_size_spinbox)
        self.bgs_method_label = QLabel("Background subtraction method")
        self.bgs_method_combobox = QComboBox()
        self.bgs_method_combobox.addItems(["MOG2", "KNN"])
        self.main_layout.addWidget(self.bgs_method_label)
        self.main_layout.addWidget(self.bgs_method_combobox)
        self.bgs_learning_rate_label = QLabel("Background subtraction learning rate")
        self.bgs_learning_rate_spinbox = QDoubleSpinBox()
        self.bgs_learning_rate_spinbox.setRange(0.0, 1.0)
        self.bgs_learning_rate_spinbox.setSingleStep(0.0001)
        self.bgs_learning_rate_spinbox.setDecimals(4)
        self.bgs_learning_rate_spinbox.setValue(default_parameters.background_subtraction_learning_rate)
        self.main_layout.addWidget(self.bgs_learning_rate_label)
        self.main_layout.addWidget(self.bgs_learning_rate_spinbox)
        self.run_detection_button = QPushButton("Set parameters", self)
        self.run_detection_button.clicked.connect(self.emit_detection_parameters)
        self.main_layout.addWidget(self.run_detection_button)

    def emit_detection_parameters(self):
        self.app_signals.detection_parameters_updated.emit(self.get_detection_parameters())
    
    def get_detection_parameters(self):
        return DetectionParameters(
            self.min_area_spinbox.value(),
            self.blur_size_spinbox.value(),
            self.bgs_method_combobox.currentText(),
            self.bgs_learning_rate_spinbox.value())
    
    def run_detection(self):
        if self.detection_thread is not None:
            self.detection_thread.requestInterruption()
            self.detection_thread.wait()
        self.detection_thread = DetectionThread(self.app_signals, self.video_path, self.frame_number, self.get_detection_parameters())
        self.detection_thread.start()

    def set_frame_number(self, frame_number):
        self.frame_number = frame_number
        self.frame_number_label.setText(f"Frame: {self.frame_number:9d}")

class ThingsOfInterestDetector:
    def __init__(self, detection_parameters):
        self.detection_parameters = detection_parameters

        if self.detection_parameters.background_subtraction_method == "MOG2":
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        elif self.detection_parameters.background_subtraction_method == "KNN":
            self.background_subtractor = cv2.createBackgroundSubtractorKNN()

    def detect(self, frame):
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.GaussianBlur(
            grayscale_image, (self.detection_parameters.blur_size, self.detection_parameters.blur_size), 0
        )
        mask = self.background_subtractor.apply(grayscale_image, learningRate=self.detection_parameters.background_subtraction_learning_rate)


        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        big_enough_contours = [c for c in contours if cv2.contourArea(c) > self.detection_parameters.min_area]

        # Draw bounding boxes for each contour on the frame.
        bounding_boxes = []
        for c in big_enough_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))

        return bounding_boxes
    
    def reset(self):
        self.background_subtractor.clear()

class DetectionThread(QThread):
    def __init__(self, app_signals, video_path, frame_number, detection_parameters):
        super().__init__()
        self.app_signals = app_signals
        self.video_path = video_path
        self.frame_number = frame_number
        self.detection_parameters = detection_parameters

    def run(self):
        window_size = 100 # FIXME: should depend on the history parameter in some way.
        start_frame = max(0, self.frame_number - window_size)

        if start_frame == self.frame_number:
            print("Not enough previous frames to create a background model")
            return

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if self.detection_parameters.background_subtraction_method == "MOG2":
            self.background_subtracter = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        elif self.detection_parameters.background_subtraction_method == "KNN":
            self.background_subtracter = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError(f"Unknown background subtraction method: {self.detection_parameters.background_subtraction_method}")

        for frame_i in range(start_frame, self.frame_number):
            if self.isInterruptionRequested():
                cap.release()
                return
            
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_i}")
            
            grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayscale_image = cv2.GaussianBlur(
                grayscale_image, (self.detection_parameters.blur_size, self.detection_parameters.blur_size), 0
            )
            mask = self.background_subtracter.apply(grayscale_image, learningRate=self.detection_parameters.background_subtraction_learning_rate)


        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        big_enough_contours = [c for c in contours if cv2.contourArea(c) > self.detection_parameters.min_area]

        # Draw bounding boxes for each contour on the frame.
        for c in big_enough_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

        print(f"Ran detection with parameters: {self.detection_parameters}")
        print(f"Found {len(contours)} contours and {len(big_enough_contours)} contours with area > {self.detection_parameters.min_area}")
        self.app_signals.detection_completed.emit(cv_image_to_q_image(frame))
        self.app_signals.detection_mask_updated.emit(cv_image_to_q_image(mask))
