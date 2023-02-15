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
from YOLOv7TensorRT import BaseEngine
import YOLOv7TensorRT as yolov7
import time


class BaseEngineHumans(BaseEngine):
    # def __init__(self, engine_path, imgsz=(640, 640)):
    #     super().__init__(engine_path, imgsz=(640, 640))
    #     self.class_names = ['person']

    def direct_inference(self, captured_image, conf=0.25):
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
            final_boxes_person = []
            final_scores_person = []
            final_cls_inds_person = []
            for i in range(num):
                if final_cls_inds[i] == 0:
                    final_boxes_person.append(final_boxes[i])
                    final_scores_person.append(final_scores[i])
                    final_cls_inds_person.append(final_cls_inds[i])

            origin_img = yolov7.vis(origin_img, final_boxes_person, final_scores_person, final_cls_inds_person,
                             conf=conf, class_names=self.class_names)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
        return origin_img
class Logic(QMainWindow, Ui_MainWindow):
    def overwriteLogic(self):
        self.model_name = 'solov2'  # Default model
        self.run = False
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.trigger_once)
        self.ignoreEvent = False
        self.polygonMask = True
        self.ignoreFrame = False
        self.close_event_called = False
        self.allowCapture = True
        self.confidence = 0.25
        self.cam = cv2.VideoCapture(0)
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
        pass

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
        thread = threading.Thread(target=self.cuda_context)
        thread.start()

    def cuda_context(self):
        cuda.init()
        cuda_context = cuda.Device(0).make_context()
        pred = BaseEngineHumans(engine_path='./tensorrt-python/yolov7-tiny-nms.trt')


        while True:
            if self.allowCapture:
                self.time_start = time.time()
                ret, frame = self.cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                confidence = self.confidence
                origin_img = pred.direct_inference(frame, conf=confidence)
                self.set_image(origin_img)
                self.time_detect = time.time() - self.time_start
                self.label_4.setText(str(self.time_detect)) # Has to use a label, editbox just freezes the GUI

                k = cv2.waitKey(1)
            else:
                break

        self.cam.release()

        cv2.destroyAllWindows()
        cuda_context.pop()

    # def cuda_context2(self):
    #     cuda.init()
    #     cuda_context = cuda.Device(0).make_context()
    #     pred = BaseEngineHumans(engine_path='./tensorrt-python/YOLOv7X.trt')
    #
    #     options = QFileDialog.Options()
    #     options |= QFileDialog.ReadOnly
    #     fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
    #                                               "All Files (*);;Images (*.png *.xpm *.jpg *.bmp *.gif)",
    #                                               options=options)
    #     while True:
    #         if fileName:
    #             image = cv2.imread(fileName)
    #             self.time_start = time.time()
    #             confidence = self.confidence
    #             origin_img = pred.direct_inference(image, conf=confidence)
    #             self.set_image(origin_img)
    #             self.time_detect = time.time() - self.time_start
    #             self.label_4.setText(str(self.time_detect))  # Has to use a label, editbox just freezes the GUI
    #         else:
    #             break



        cuda_context.pop()
        # en:Stop grab image

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
        image = imutils.resize(image, width=640)
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