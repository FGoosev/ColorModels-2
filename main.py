import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
                             QSlider, QWidget, QCheckBox, QScrollArea)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная_2")
        self.setGeometry(100, 100, 1200, 800)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.container = QWidget()
        self.scroll_area.setWidget(self.container)

        self.image_label = QLabel(self.container)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.hist_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.ax_hist = self.hist_canvas.figure.subplots()

        self.load_button = QPushButton("Загрузить")
        self.load_button.clicked.connect(self.load_image)

        self.bw_button = QPushButton("Черно-белое")
        self.bw_button.clicked.connect(self.convert_to_bw)

        self.linear_corr_button = QPushButton("Линейная")
        self.linear_corr_button.clicked.connect(self.linear_correction)

        self.nonlinear_corr_button = QPushButton("Нелинейная")
        self.nonlinear_corr_button.clicked.connect(self.nonlinear_correction)

        self.show_hist_checkbox = QCheckBox("Гистограмма")
        self.show_hist_checkbox.stateChanged.connect(self.toggle_histogram)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(10)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)

        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setMinimum(0)
        self.saturation_slider.setMaximum(100)
        self.saturation_slider.setValue(10)
        self.saturation_slider.valueChanged.connect(self.adjust_saturation)

        layout = QVBoxLayout(self.container)
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_button)
        layout.addWidget(self.bw_button)
        layout.addWidget(self.linear_corr_button)
        layout.addWidget(self.nonlinear_corr_button)
        layout.addWidget(self.show_hist_checkbox)
        layout.addWidget(self.hist_canvas)

        sliders_layout = QHBoxLayout()
        sliders_layout.addWidget(QLabel("Яркость"))
        sliders_layout.addWidget(self.brightness_slider)
        sliders_layout.addWidget(QLabel("Контрастность"))
        sliders_layout.addWidget(self.contrast_slider)
        sliders_layout.addWidget(QLabel("Насыщенность"))
        sliders_layout.addWidget(self.saturation_slider)

        layout.addLayout(sliders_layout)

        self.setCentralWidget(self.scroll_area)

        self.image = None
        self.original_image = None
        self.show_histogram = False

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.original_image = self.image.copy()
            self.display_image()

    def display_image(self):
        if self.image is not None:
            qimage = self.convert_cv_qt(self.image)
            self.image_label.setPixmap(QPixmap.fromImage(qimage).scaledToHeight(500))
            if self.show_histogram:
                self.plot_histogram(self.image)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_qt_format

    def plot_histogram(self, image):
        self.ax_hist.clear()
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            self.ax_hist.plot(hist, color=col)
        self.ax_hist.set_xlim([0, 256])
        self.hist_canvas.draw()

    def convert_to_bw(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            self.display_image()

    def adjust_brightness(self, value):
        if self.image is not None:
            value = value / 100.0
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[..., 2] = hsv[..., 2] * (1 + value)
            hsv[..., 2][hsv[..., 2] > 255] = 255
            self.image = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
            self.display_image()

    def adjust_contrast(self, value):
        if self.image is not None:
            value = value / 10.0
            self.image = cv2.convertScaleAbs(self.original_image, alpha=value, beta=0)
            self.display_image()

    def adjust_saturation(self, value):
        if self.image is not None:
            value = value / 10.0
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[..., 1] = hsv[..., 1] * value
            hsv[..., 1][hsv[..., 1] > 255] = 255
            self.image = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
            self.display_image()

    def linear_correction(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equ = cv2.equalizeHist(gray)
            self.image = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
            self.display_image()

    def nonlinear_correction(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            self.image = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
            self.display_image()

    def toggle_histogram(self, state):
        self.show_histogram = state == Qt.Checked
        if self.image is not None:
            self.display_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    sys.exit(app.exec_())
