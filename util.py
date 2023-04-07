"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import numpy as np
import os
import sys
import cv2
import torch


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def string_contains_any(string, substring_list):
    for substring in substring_list:
        if substring in string:
            return True
    return False


class Reporter(object):
    def __init__(self, reporter_log_root, epoch):
        self.log_file = os.path.join(reporter_log_root, str(epoch))
        self.epoch = epoch
        self.report_dict = {
            'summary': True,
            'step': self.epoch,
        }

    def add(self, key, val):
        self.report_dict.update({key: val})

    def write(self):
        log_file = self.log_file
        while os.path.isfile(log_file):
            log_file += '_'
        with open(log_file, 'w') as f:
            f.write(json.dumps(self.report_dict))


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))


def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[0, :, :] = x[0, :, :] * std[0] + mean[0]
    x[1, :, :] = x[1, :, :] * std[1] + mean[1]
    x[2, :, :] = x[2, :, :] * std[2] + mean[2]
    return x


def visualize(image_tensor, cam = [], bounding_box = [], bounding_box_pre = []):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (3, H, W)(0~1)
        cam: (array) shape => (H, W)(0~1)
        bounding_box: (array) shape => (4,)[x1, y1, x2, y2]
    Return:
        synthesized image (array): shape =>(H, W, 3)
    """
    if len(image_tensor.shape) == 2:    # 临时调试边界框阈值划分添加
        img_array = image_tensor
        img_array = cv2.applyColorMap(np.uint8(image_tensor*255), cv2.COLORMAP_JET) / 255
    else:
        img = reverse_normalize(image_tensor.clone().detach().squeeze())
        # 去除img冗余维度，通道转换，转numpy，维度调整
        r, g, b = img.split(1)
        img_array = torch.cat([b, g, r]).cpu().numpy().transpose(1, 2, 0)
        

    if len(cam):
        # 去除cam冗余维度，生成热图，注意opencv默认bgr, heatmap: (224, 224, 3)
        heatmap = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET) / 255
        result = heatmap + img_array
        result = result / result.max()
    else:
        result = img_array.copy()

    if len(bounding_box):
        for box in bounding_box:
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    if len(bounding_box_pre):
        for box in bounding_box_pre:
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    return result


def showImg(name, img):
    img = img.squeeze()
    cv2.imshow(name, img)
    cv2.waitKey(0)