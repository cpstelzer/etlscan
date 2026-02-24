import os
import sys
import time
import cv2
import numpy as np
from pypylon import pylon
from collections import deque

# UI Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QLineEdit,
                             QPushButton, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

# --- Analysis Logic ---

def get_well_background_peak_single(image_uint8, well_mask, min_intensity=75):
    """Calculates the dominant brightness peak in the masked area."""
    well_pixels = image_uint8[well_mask]
    if well_pixels.size == 0:
        return None
        
    hist, bin_edges = np.histogram(well_pixels, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    search_start_idx = np.searchsorted(bin_centers, min_intensity)
    if search_start_idx >= len(hist):
        return float(np.argmax(hist))
    
    peak_idx_in_range = np.argmax(hist[search_start_idx:])
    peak_val = bin_centers[search_start_idx + peak_idx_in_range]
    return float(peak_val)

# --- Camera & Analysis Thread ---

class CameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, float) 
    error_occurred = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.refresh_rate = config['analysis']['initial_refresh_hz']
        self.radius_fraction = config['analysis']['radius_fraction']
        self.min_intensity = config['analysis']['min_intensity']
        self.camera = None

    def setup_camera(self):
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if not devices:
                raise Exception("No Basler camera detected.")
            
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.camera.Open()
            
            # ROI Configuration
            max_w = self.camera.WidthMax.Value
            max_h = self.camera.HeightMax.Value
            req_w = min(self.config['roi']['width'], max_w)
            req_h = min(self.config['roi']['height'], max_h)

            self.camera.Width.Value = req_w
            self.camera.Height.Value = req_h
            self.camera.OffsetX.Value = (max_w - req_w) // 2
            self.camera.OffsetY.Value = (max_h - req_h) // 2
            
            # Settings
            self.camera.ExposureTime.SetValue(self.config['camera']['exposure_us'])
            
            if "Mono8" in self.camera.PixelFormat.Symbolics:
                self.camera.PixelFormat.SetValue("Mono8")
            
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            return True
        except Exception as e:
            self.error_occurred.emit(str(e))
            return False

    def run(self):
        if not self.setup_camera():
            return

        self.running = True
        while self.running:
            loop_start = time.perf_counter()
            try:
                res = self.camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    img = res.Array
                    h, w = img.shape
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    radius = int(min(h, w) * self.radius_fraction)
                    cv2.circle(mask, (w // 2, h // 2), radius, 1, -1)
                    
                    peak = get_well_background_peak_single(img, mask.astype(bool), self.min_intensity)
                    self.frame_ready.emit(img.copy(), peak if peak is not None else 0.0)
                res.Release()
            except Exception as e:
                print(f"Acquisition Error: {e}")

            elapsed = time.perf_counter() - loop_start
            wait_time = max(0.001, (1.0 / self.refresh_rate) - elapsed)
            time.sleep(wait_time)

    def stop(self):
        self.running = False
        self.wait()
        if self.camera and self.camera.IsOpen():
            self.camera.StopGrabbing()
            self.camera.Close()

# --- Main User Interface ---

class WellAnalyzerUI(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Live Well Intensity Monitor (Smoothed)")
        self.setMinimumSize(1000, 750)
        self.latest_frame = None

        # History buffer for moving average (Mean of samples per 1Hz)
        # We initialize with a maxlen based on the starting Hz
        initial_hz = config['analysis']['initial_refresh_hz']
        self.peak_history = deque(maxlen=int(max(1, initial_hz)))

        # --- UI Setup ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; border: 2px solid #444;")
        layout.addWidget(self.video_label, stretch=4)

        controls_layout = QVBoxLayout()
        layout.addLayout(controls_layout, stretch=1)

        self.peak_label = QLabel("Peak (Mean): --")
        self.peak_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00FF00;")
        controls_layout.addWidget(self.peak_label)
        
        self.raw_label = QLabel("Instantaneous: --")
        self.raw_label.setStyleSheet("font-size: 14px; color: #888; margin-bottom: 20px;")
        controls_layout.addWidget(self.raw_label)

        form = QFormLayout()
        self.spin_refresh = QDoubleSpinBox()
        self.spin_refresh.setRange(1.0, 30.0)
        self.spin_refresh.setValue(initial_hz)
        self.spin_refresh.setSuffix(" Hz")
        self.spin_refresh.valueChanged.connect(self.update_params)
        form.addRow("Refresh Rate:", self.spin_refresh)

        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.05, 0.50)
        self.spin_radius.setSingleStep(0.01)
        self.spin_radius.setValue(config['analysis']['radius_fraction'])
        self.spin_radius.valueChanged.connect(self.update_params)
        form.addRow("Mask Radius:", self.spin_radius)

        self.input_image_name = QLineEdit()
        self.input_image_name.setPlaceholderText("e.g. well_background_01")
        form.addRow("Image Name:", self.input_image_name)

        controls_layout.addLayout(form)

        self.btn_capture = QPushButton("Capture Image")
        self.btn_capture.clicked.connect(self.capture_image)
        controls_layout.addWidget(self.btn_capture)

        self.capture_status = QLabel("")
        self.capture_status.setStyleSheet("font-size: 12px; color: #AAA;")
        controls_layout.addWidget(self.capture_status)

        controls_layout.addStretch()

        self.btn_quit = QPushButton("Exit Program")
        self.btn_quit.clicked.connect(self.close)
        controls_layout.addWidget(self.btn_quit)

        self.worker = CameraWorker(config)
        self.worker.frame_ready.connect(self.update_display)
        self.worker.error_occurred.connect(self.on_hardware_error)
        self.worker.start()

    def update_params(self):
        new_hz = self.spin_refresh.value()
        self.worker.refresh_rate = new_hz
        self.worker.radius_fraction = self.spin_radius.value()
        
        # Adjust history buffer size to match 1 second of data at new Hz
        new_maxlen = int(max(1, new_hz))
        if new_maxlen != self.peak_history.maxlen:
            self.peak_history = deque(self.peak_history, maxlen=new_maxlen)

    @pyqtSlot(np.ndarray, float)
    def update_display(self, frame, raw_peak):
        # 1. Update smoothed calculation
        self.peak_history.append(raw_peak)
        smoothed_peak = sum(self.peak_history) / len(self.peak_history)
        self.latest_frame = frame

        # 2. Update UI text
        self.peak_label.setText(f"Peak (Mean): {smoothed_peak:.2f}")
        self.raw_label.setText(f"Instantaneous: {raw_peak:.1f}")

        # 3. Handle Image Overlay
        h, w = frame.shape
        display_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        radius = int(min(h, w) * self.worker.radius_fraction)
        cv2.circle(display_img, (w // 2, h // 2), radius, (255, 100, 0), 2)
        
        # 4. Render to Label
        q_img = QImage(display_img.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

    def capture_image(self):
        image_name = self.input_image_name.text().strip()
        if not image_name:
            self.capture_status.setText("Enter an image name before capturing.")
            return
        if self.latest_frame is None:
            self.capture_status.setText("No frame available yet.")
            return

        output_path = f"C:/wellscanner/background_brightness/raw/{image_name}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, self.latest_frame)
        if success:
            self.capture_status.setText(f"Saved: {output_path}")
        else:
            self.capture_status.setText("Failed to save image.")

    def on_hardware_error(self, msg):
        self.video_label.setText(f"Hardware Error:\n{msg}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

# --- Entry Point ---

if __name__ == "__main__":
    # !!! RE-INSERT YOUR UPDATED CAMERA VALUES HERE !!!
    configuration = {
        'camera': {
            'exposure_us': 750, 
        },
        'roi': {
            'width': 4512, 
            'height': 4512
        },
        'analysis': {
            'initial_refresh_hz': 10.0,  # Default rate
            'radius_fraction': 0.33, 
            'min_intensity': 75      
        }
    }

    app = QApplication(sys.argv)
    window = WellAnalyzerUI(configuration)
    window.show()
    sys.exit(app.exec())
