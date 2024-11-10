import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch

from ultralytics.utils.plotting import Annotator, colors, save_one_box

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    scale_segments,
    strip_optimizer,
)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

class YOLOv5():
    def __init__(self, weight_file, data_file):
        self.weights = weight_file
        self.data = data_file
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.augment = False
        self.device = ""
        self.view_img = True
        self.classes = None
        self.agnostic_nms = False
        self.line_thickness = 2
        self.half = False
        self.dnn = False

        self.s = str()

        self.load_model()

        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))  # warmup

    def load_model(self):
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    def predict(self, image):
        im = letterbox(image, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred, proto = self.model(im, augment=self.augment, visualize=False)[:2]

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=32)

        for i, det in enumerate(pred):  # per image
            self.s += f'{i}: '
            self.s += '%gx%g ' % im.shape[2:]
            annotator = Annotator(image, line_width=self.line_thickness, example=str(self.names))

            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=im[i]
                )

                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    c = int(cls)  # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            im0 = annotator.result()
            cv2.imshow('window', im0)
            if cv2.waitKey(1) == ord("q"):  # 1 millisecond
                exit()
        return im0

class YOLOv5Node(Node):

    def __init__(self):
        super().__init__('YOLOv5Node')

        self.declare_parameter('weight_file', rclpy.Parameter.Type.STRING)
        self.declare_parameter('data_file', rclpy.Parameter.Type.STRING)

        weight_file = self.get_parameter('weight_file')
        data_file = self.get_parameter('data_file')
        
        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(Image, 'image_raw', self.image_callback,10)
        self.pub_image = self.create_publisher(Image, 'yolov5/pred', 10)

        self.yolov5 = YOLOv5(str(weight_file.value), str(data_file.value))

    def image_callback(self, image):
        image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")

        annotated_image = self.yolov5.predict(image_raw)

        ros_image = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
        self.pub_image.publish(ros_image)


def main(args=None):
    rclpy.init(args=args) 

    yolov5_node = YOLOv5Node()

    rclpy.spin(yolov5_node)

    yolov5_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()