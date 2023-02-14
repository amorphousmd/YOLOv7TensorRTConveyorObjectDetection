# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode


# class LoadImageNoPath(LoadImages):
#     def __init__(self, image, img_size=640, stride=32, auto=True, transforms=None):
#         super().__init__(self, path=None, img_size=640, stride=32, auto=True, transforms=None)
#
#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         path = self.files[self.count]
#
#         if self.video_flag[self.count]:
#             # Read video
#             self.mode = 'video'
#             ret_val, im0 = self.cap.read()
#             while not ret_val:
#                 self.count += 1
#                 self.cap.release()
#                 if self.count == self.nf:  # last video
#                     raise StopIteration
#                 path = self.files[self.count]
#                 self.new_video(path)
#                 ret_val, im0 = self.cap.read()
#
#             self.frame += 1
#             s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
#
#         else:
#             # Read image
#             self.count += 1
#             im0 = cv2.imread(path)  # BGR
#             assert im0 is not None, f'Image Not Found {path}'
#             s = f'image {self.count}/{self.nf} {path}: '
#
#         if self.transforms:
#             im = self.transforms(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))  # transforms
#         else:
#             im = LoadImages.letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
#             im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#             im = np.ascontiguousarray(im)  # contiguous
#
#         return path, im, im0, self.cap, s

class SegmentInference(object):
    def __init__(self):
        self.bboxes = []
        self.masks = None
        self.source = None
        self.save_img = None
        self.webcam = None
        self.save_dir = None
        self.device = None
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.imgsz = None
        self.inferred_image = None

    @smart_inference_mode()
    def start(
            self,
            weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
            # weights=ROOT / 'YOLOv7Segmentation.engine',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            # source='./42.jpg',
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.2,  # confidence threshold
            iou_thres=0.5,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/predict-seg',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        self.source = str(source)
        self.save_img = not nosave and not self.source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(source)  # download

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size


    def infer(
            self,
            weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
            # weights=ROOT / 'YOLOv7Segmentation.engine',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            # source='./42.jpg',
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.2,  # confidence threshold
            iou_thres=0.5,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/predict-seg',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        # Dataloader
        if self.webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=self.pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, out = self.model(im, augment=augment, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                # pred = [x * 1.0005 for x in pred] # Testing the mask function

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    self.masks = masks
                    self.bboxes = det[:, :4]
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    self.bboxes = det[:, :4]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                    # Mask plotting ----------------------------------------------------------------------------------------

                    # Write results
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_img or save_crop or view_img:  # Add bbox to image
                            pass
                            # c = int(cls)  # integer class
                            # label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            # annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
                # Stream results

                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

                self.inferred_image = im0
            # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        self.inference_time = t[0] + t[1] + t[2]
        if save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    def get_inferred_image(self):
        return self.inferred_image

    def get_inferred_image_with_MaskCentroid(self):
        temp = self.inferred_image
        if self.masks is None:
            return temp
        for i in range(self.masks.shape[0]):
            # Grab a single mask, each mask has only 1 detection
            img = self.masks[i].cpu().numpy()
            gray = img.astype(np.uint8)

            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop through the contour found in the mask (1 contour only)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300: # A threshold to filter out bad contours
                    # Compute the moments of the contour
                    moments = cv2.moments(cnt)

                    # Compute the centroid of the object
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])

                    # Draw the centroid on the image
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

                    # Resize the centroid to fit on the original image, a small modification as
                    # the image is collapsed from the middle not the edge
                    cx2 = int(cx * 2592 / 640)
                    cy2 = int(cy * 1944 / 640)
                    midline = 1944 / 2
                    cy3 = int(midline - (midline - cy2) * 2592 / 1944)

                    cv2.circle(temp, (cx2, cy3), 10, (0, 0, 255), -1)
        self.masks = None
        return temp

    def get_inferred_image_with_BoxCentroid(self):
        temp = self.inferred_image
        for i in range(self.bboxes.shape[0]):
            center = (int((self.bboxes[i][0] + self.bboxes[i][2])/2),
                      int((self.bboxes[i][1] + self.bboxes[i][3])/2))
            temp = cv2.circle(temp, center, radius=10, color=(0, 0, 255), thickness=-1)
        return temp

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'YOLOv7Segmentation.engine', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / '63.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    opt.nosave = True
    check_requirements(exclude=('tensorboard', 'thop'))
    segment_object = SegmentInference()
    segment_object.start(**vars(opt))
    segment_object.infer(**vars(opt))
    cv2.imshow('Annotated Image 1', resize_image(segment_object.get_inferred_image_with_MaskCentroid(), 0.4))
    cv2.waitKey(0)
    # Changing the image to infer on
    opt.conf_thres = 0.99
    opt.source = ROOT / '69.jpg'
    segment_object.infer(**vars(opt))
    cv2.imshow('Annotated Image 2', resize_image(segment_object.get_inferred_image_with_MaskCentroid(), 0.4))
    cv2.waitKey(0)


def resize_image(old_image, percentage):
    # Load the image

    # Get the original size
    scale_percent = percentage  # percent of original size
    width = int(old_image.shape[1] * scale_percent)
    height = int(old_image.shape[0] * scale_percent)
    dim = (width, height)

    # resize image
    resized = cv2.resize(old_image, dim, interpolation=cv2.INTER_AREA)
    return resized


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
