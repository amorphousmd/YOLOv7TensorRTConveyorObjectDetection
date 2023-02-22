import math
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from Ui_Design import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMessageBox
import cv2, imutils, threading
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import numpy as np

import os
import time

# pypylon
# plotting for graphs and display of image
import matplotlib.pyplot as plt
# linear algebra and basic math on image matrices
import numpy as np
# OpenCV for image processing functions
import cv2

from YOLOv7TensorRT import _COLORS
from seg.segment.inferences import SegmentInference, parse_opt
from YOLOv7TensorRT import BaseEngine
import YOLOv7TensorRT as yolov7
from YOLOv7TensorRTv2 import BaseEngineVer2
import time
from Haytraochoanh import Yolov7


class BaseEngineCracker(BaseEngine):
    def __init__(self, engine_path, imgsz=(640, 640)):
        super().__init__(engine_path, imgsz=(640, 640))
        self.class_names = ['BAD', 'GOOD']
        self.coord_list = []


    def direct_inference(self, captured_image, conf=0.25):
        self.coord_list = [] # Reset the coord list every time
        origin_img = captured_image
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img, ratio = yolov7.preproc(origin_img, self.imgsz, self.mean, self.std)
        num, final_boxes, final_scores, final_cls_inds = self.infer(img)
        # num: number of object detected
        # final_boxes: Coordinates of the bounding boxes
        # final scores: Confidence score of each object
        # final_cls_inds: The position (index) of class in the list above (80 classes, count start at 0)
        final_boxes = np.reshape(final_boxes, (-1, 4))  # Unknown number of rows and 4 columns
        num = num[0]
        if num > 0:
            final_boxes, final_scores, final_cls_inds = final_boxes[:num] / ratio, final_scores[:num], final_cls_inds[
                                                                                                       :num]
            # final_boxes_bad = []
            # final_scores_bad = []
            # final_cls_inds_bad = []
            # final_boxes_good = []
            # final_scores_good = []
            # final_cls_inds_good = []
            for i in range(num):
                if final_cls_inds[i] == 0:
                    score_array = np.array([final_scores[i]])
                    concatenated_array = np.concatenate((final_boxes[i], score_array))
                    self.coord_list.append(concatenated_array)
            #
            #     if final_cls_inds[i] == 0:
            #         final_boxes_good.append(final_boxes[i])
            #         final_scores_good.append(final_scores[i])
            #         final_cls_inds_good.append(final_cls_inds[i])
            #
            final_boxes_cracker = final_boxes
            final_scores_cracker = final_scores
            final_cls_inds_cracker = final_cls_inds

            origin_img = yolov7.vis(origin_img, final_boxes_cracker, final_scores_cracker, final_cls_inds_cracker,
                             conf=conf, class_names=self.class_names)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
        return origin_img

def coord_list_to_center_list(coord_list, confidence):
    centers = []
    for coord in coord_list:
        if coord[4] >= confidence:
            x_left = coord[0]
            y_left = coord[1]
            x_right = coord[2]
            y_right = coord[3]
            centers.append([(x_left + x_right)/2, (y_left + y_right)/2])
    return centers

class Logic(QMainWindow, Ui_MainWindow):
    def overwriteLogic(self):
        self.model_name = 'solov2'  # Default model
        self.model = 0 # Default model index
        self.run = False
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.trigger_once)
        self.ignoreEvent = False
        self.polygonMask = True
        self.ignoreFrame = False
        self.close_event_called = False
        self.allowCapture = True
        self.confidence = 0.5
        # self.cam = cv2.VideoCapture(0)
        self.comboModels.setItemText(0, "YOLOv7")
        self.comboModels.setItemText(1, 'YOLOv7Segmentation')
        self.comboModels.setItemText(2, 'YOLOv7Tiny')
        self.btnEnum.clicked.connect(self.enum_devices)
        self.btnOpen.clicked.connect(self.open_device)
        self.btnClose.clicked.connect(self.close_device)
        self.bnStart.clicked.connect(self.start_grabbing)
        self.bnSingle.clicked.connect(self.single_grab)
        self.bnStop.clicked.connect(self.stop_grabbing)
        self.bnSave.clicked.connect(self.saveImage)
        self.btnGetParam.clicked.connect(self.get_param)
        self.btnSetParam.clicked.connect(self.set_param)
        self.radioContinueMode.clicked.connect(self.set_continue_mode)
        self.radioTriggerMode.clicked.connect(self.set_software_trigger_mode)
        self.btnLoadCheckpoint.clicked.connect(self.load_model)
        self.btnRunInferenceVideo.clicked.connect(self.run_model)
        self.btnRunCalib.clicked.connect(self.runCameraCalib)
        self.btnLoadCalib.clicked.connect(self.loadCameraCalib)
        self.btnRunInference.clicked.connect(self.runInferenceImage)
        self.btnLoadImage.clicked.connect(self.loadImage)
        self.comboModels.currentIndexChanged.connect(self.select_model)
        self.btnStartCamCalibTest.clicked.connect(self.startCamCalibTest)
        self.btnStartServer.clicked.connect(self.start_server)
        self.btnConnAddr.clicked.connect(self.print_values)
        self.btnSendTCPIP.clicked.connect(self.send_data)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        app.aboutToQuit.connect(self.closeEvent)

    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

    def closeEvent(self):
        if self.close_event_called:
            return
        self.close_event_called = True
        # rest of your close event logic here
        self.allowCapture = False
        print('closeEvent called')

    def start_server(self):
        pass
        # serverUtilities.establish_connection()

    def print_values(self):
        pass
        # print('')
        # print(serverUtilities.conn)
        # print(serverUtilities.addr)

    def send_data(self):
        pass

    def startCamCalibTest(self):
        pass

    def load_model(self):
        pass
    def getPos(self, event):
        pass

    def loadCameraCalib(self):
        pass
        # filename = CameraUtils.loadCalibration()

    def runCameraCalib(self):
        pass
        # CameraUtils.runCalibration((10, 7), (2592, 1944), 25)

    def saveImage(self):
        pass

    def select_model(self, i):
        if i == 0:
            self.model = 0
            print("[MODEL]: YOLOv7")
        elif i == 1:
            self.model = 1
            print("[MODEL]: YOLOv7Segmentation")
        elif i == 2:
            self.model = 2
            print("[MODEL]: YOLOv7Tiny")

    def run_model(self):
        print(self.editScoreThreshold.toPlainText())
        self.confidence = float(self.editScoreThreshold.toPlainText())
    def detect(self, image, score_thr_value, center=True):
        pass

    def set_img_show(self, image):
        pass

    def xFunc(event):
        pass

    def enum_devices(self):
        pass

    def open_device(self):
        pass

    def start_grabbing(self):
        # Old code (bounding boxes inferences)
        # thread = threading.Thread(target=self.cuda_context2)
        # thread.start()

        # New code (Segmentation)
        # segment_object = SegmentInference()
        # segment_object.start(**vars(opt))
        # segment_object.infer()
        # segment_object.getcentermask
        # segment_object.getcenterbox
        if self.model == 0:
            self.cuda_contextYOLO()
        elif self.model == 1:
            self.cuda_contextYOLOSegmentation()
        elif self.model == 2:
            self.cuda_contextYOLOTiny()
        else:
            print("Model not implemented yet")
            return

    def cuda_contextYOLO(self):
        cuda.init()
        cuda_context = cuda.Device(0).make_context()
        pred = BaseEngineCracker(engine_path='./tensorrt-python/YOLOv7.trt')

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        while True:
            fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                      "All Files (*);;Images (*.png *.xpm *.jpg *.bmp *.gif)",
                                                      options=options)
            if fileName:
                image = cv2.imread(fileName)
                # image = cv2.resize(image, (640, 640))
                confidence = self.confidence
                self.time_start = time.time()
                origin_img = pred.direct_inference(image, conf=confidence)
                self.time_detect = time.time() - self.time_start
                center_list = coord_list_to_center_list(pred.coord_list, self.confidence)
                for center in center_list:
                    center = (int(center[0]), int(center[1]))
                    origin_img = cv2.circle(origin_img, center, radius=10, color=(0, 0, 255), thickness=-1)
                self.set_image(origin_img)
                self.label_4.setText(str(self.time_detect))  # Has to use a label, editbox just freezes the GUI

            else:
                break
        cuda_context.pop()


    def cuda_contextYOLOTiny(self):
        pred2 = BaseEngineCracker(engine_path='./tensorrt-python/YOLOv7TinyVer5.trt')

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        while True:
            fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                      "All Files (*);;Images (*.png *.xpm *.jpg *.bmp *.gif)",
                                                      options=options)
            if fileName:
                image = cv2.imread(fileName)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                confidence = self.confidence
                self.time_start = time.time()
                origin_img = pred2.direct_inference(image, conf=confidence)
                origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
                self.time_detect = time.time() - self.time_start
                center_list = coord_list_to_center_list(pred2.coord_list, self.confidence)
                for center in center_list:
                    center = (int(center[0]), int(center[1]))
                    origin_img = cv2.circle(origin_img, center, radius=10, color=(0, 0, 255), thickness=-1)
                self.set_image(origin_img)
                self.label_4.setText(str(self.time_detect))  # Has to use a label, editbox just freezes the GUI

            else:
                break

    def cuda_contextYOLOSegmentation(self):
        cuda.init()
        cuda_context = cuda.Device(0).make_context()
        opt = parse_opt()
        opt.nosave = True
        segment_object = SegmentInference()
        segment_object.start(**vars(opt))
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        while True:
            fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                      "All Files (*);;Images (*.png *.xpm *.jpg *.bmp *.gif)",
                                                      options=options)
            if fileName:
                opt.source = fileName
                self.time_start = time.time()
                segment_object.infer(**vars(opt))
                self.time_detect = time.time() - self.time_start
                self.set_image(segment_object.get_inferred_image_with_MaskCentroid())
                self.label_4.setText(str(self.time_detect))  # Has to use a label, editbox just freezes the GUI
            else:
                break
        cuda_context.pop()

    def stop_grabbing(self):
        pass

    def close_device(self):
        pass
    def set_continue_mode(self):
        pass

    def set_software_trigger_mode(self):
        pass
    def trigger_once(self):
        pass
    def save_bmp(self):
        pass

        # en:get param

    def get_param(self):
        pass

    def set_param(self):
        pass

    def get_image(self):
        pass

        # en:set enable status

    def enable_controls(self):
        pass

    def loadImage(self):
        pass

    def set_image(self, image):
        """ This function will take image input and resize it
                    only for display purpose and convert it to QImage
                    to set at the label.
                """
        self.tmp = image
        image = cv2.resize(image, (1000, 750))
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.displayLabel.setPixmap(QtGui.QPixmap.fromImage(image))

    def runInferenceImage(self):
        pass

    def single_grab(self):
        pass

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Logic()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())