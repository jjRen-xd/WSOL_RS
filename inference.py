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

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd

from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from util import t2n, visualize, showImg

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

class SaveValues():
    """
        后期新增，记录中间反传梯度
    """
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAMComputer(object):
    def __init__(self, model, architecture, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.architecture = architecture

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))  # cam到bounding box的分割阈值

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)


    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")
        for images, targets, image_ids in self.loader:
            # targets: 一个batch图像的标签(tensor), image_ids: 一个batch图像的文件名
            image_size = images.shape[2:]
            images = images.cuda()
            # 前向传播计算cam
            cams = t2n(self.model(images, targets, return_cam=True))    # (b, 14, 14)
            for cam, image_id in zip(cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)
        return self.evaluator.compute()

    def compute_and_evaluate_gradcams(self, top1 = False, gt_known = True):
        """
            计算gradcampp需要计算梯度，即反向传播，与之前方法不同，需要额外程序
        """ 
        print("Computing and evaluating gradcams.")
        for images, targets, image_ids in self.loader:
            # targets: 一个batch图像的label(tensor), image_ids: 一个batch图像的文件名
            _, _, h, w = images.shape
            images = images.cuda()
            # 指定需要可视化的一层
            if self.architecture == 'resnet50':
                target_layer = self.model.layer4[2].bn3
            elif self.architecture == 'vgg16':
                target_layer = self.model.relu
            else:                       # inception
                target_layer = self.model.SPG_A3_2b[1]
                # target_layer = self.model.SPG_A4
            # print("target_layer:", target_layer)
            
            hook_values = SaveValues(target_layer)
            # 前向传播计算每个类别的score,并hook特征图
            logits = self.model(images)['logits']
            if top1:
                pred_scores = logits.max(dim = 1)[0]
            elif gt_known:
                # GT-Known指标
                batch_size, _ = logits.shape
                _range = torch.arange(batch_size)
                pred_scores = logits[_range, targets]
                # print(pred_scores)
            else: 
                print("Error in indicator designation!!!")
                exit()

            # TODO:FIXED 对一个batch的得分进行反向传播，并hoock倒数, 计算一个batch的gradcampp
            ''' 修复不能计算batch的问题 '''
            # 1. 反向传播计算并hook梯度
            self.model.zero_grad()                          
            pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)
            activations = hook_values.activations           # ([bz, 1024, 14, 14])
            gradients = hook_values.gradients               # ([bz, 1024, 14, 14])
            bz, nc, _, _ = activations.shape                # (batch_size, num_channel, height, width)
            # 2. 计算梯度图中每个梯度的权重alpha
            numerator = gradients.pow(2)
            denominator = 2 * gradients.pow(2)
            ag = activations * gradients.pow(3)
            denominator += ag.view(nc, -1).sum(-1, keepdim=True).view(nc, 1, 1)
            denominator = torch.where(
                denominator != 0.0, denominator, torch.ones_like(denominator)
            )
            alpha = numerator / (denominator + 1e-7)        # ([bz, 1024, 14, 14])
            # 3. 计算梯度图权重weights
            relu_grad = gradients.clone()                   # ([bz, 1024, 14, 14])
            for idx, (score, grad) in enumerate(zip(pred_scores, gradients)):
                relu_grad[idx] = F.relu(score.exp() * grad)
            weights = (alpha * relu_grad).view(bz, nc, -1).sum(-1).view(bz, nc, 1, 1)   # ([bz, 1024, 1, 1])
            # 4. 计算一组batch的CAMs
            cams = (weights * activations).sum(1)            
            cams = t2n(F.relu(cams))                         # ([bz, 14, 14])
            for cam, image, image_id in zip(cams, images, image_ids):
                cam = cv2.resize(cam, (h, w),               # 上采样，基于4x4像素邻域的3次插值法
                                        interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam)    # (224, 224)
            

            # for idx, (image, image_label, image_id, score) in enumerate(zip(images, targets, image_ids, pred_scores)):
            #     # 反向传播计算并hook梯度
            #     self.model.zero_grad()
            #     score.backward(retain_graph=True)
            #     activations = hook_values.activations[idx]  # ([2048, 14, 14])
            #     gradients = hook_values.gradients[idx]      # ([2048, 14, 14])
            #     # 计算gradcampp
            #     nc, h, w = activations.shape                # (num_channel, height, width)  
            #     # 计算alpha
            #     numerator = gradients.pow(2)
            #     denominator = 2 * gradients.pow(2)
            #     ag = activations * gradients.pow(3)
            #     denominator += ag.view(nc, -1).sum(-1, keepdim=True).view(nc, 1, 1)
            #     denominator = torch.where(
            #         denominator != 0.0, denominator, torch.ones_like(denominator)
            #     ) 
            #     alpha = numerator / (denominator + 1e-7)

            #     relu_grad = F.relu(score.exp() * gradients)
            #     weights = (alpha * relu_grad).view(nc, -1).sum(-1).view(nc, 1, 1)   # ([2048, 1, 1])
            #     # shape => (H', W')
            #     cam = (weights * activations).sum(0)
            #     cam = t2n(F.relu(cam))                              # (14, 14)
                # cam_resized = cv2.resize(cam, image_size,
            #                              interpolation=cv2.INTER_CUBIC)
            #     cam_normalized = normalize_scoremap(cam_resized)    # (224, 224)
            

                # boxes_pre, box_gt = self.evaluator.accumulate(cam_normalized, image_id, return_bb = True)
                # print(boxes_pre, box_gt)
                # masked_image = mask_image(cam_normalized, 166= image).unsqueeze(dim=0).float()
                # showImg("test", visualize(masked_image))
                # showImg("test", visualize(image, cam = cam_normalized))
                # showImg("test", visualize(image, cam_normalized, box_gt[0], boxes_pre))


                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)
        return self.evaluator.compute()


def compute_gradcampp(images, targets, model, architecture, top1 = False, gt_known = True):
    _, _, h, w = images.shape

    # 指定需要可视化的一层，并hook参数及倒数
    if architecture == 'resnet50':
        target_layer = model.layer4[2].bn3
    elif architecture == 'vgg16':
        target_layer = model.relu
    else:                       # inception
        target_layer = model.SPG_A3_2b[1]
        # target_layer = self.model.SPG_A4
    # print("target_layer:", target_layer)
    hook_values = SaveValues(target_layer)

    # 前向传播计算每个类别的score,并hook特征图,计算分类损失
    logits = model(images)['logits']
    cross_entropy_loss = nn.CrossEntropyLoss().cuda()
    loss_cl = cross_entropy_loss(logits, targets)

    if top1:
        pred_scores = logits.max(dim = 1)[0]
    elif gt_known:
        # GT-Known指标
        batch_size, _ = logits.shape
        _range = torch.arange(batch_size)
        pred_scores = logits[_range, targets]
    else: 
        print("Error in indicator designation!!!")
        exit()

    # TODO:FIXED 对一个batch的得分进行反向传播，并hoock倒数, 计算一个batch的gradcampp
    ''' 修复不能计算batch的问题 '''
    # 1. 反向传播计算并hook梯度
    model.zero_grad()                          
    pred_scores.backward(torch.ones_like(pred_scores), retain_graph=True)
    activations = hook_values.activations           # ([bz, 1024, 14, 14])
    gradients = hook_values.gradients               # ([bz, 1024, 14, 14])
    bz, nc, _, _ = activations.shape                # (batch_size, num_channel, height, width)
    # 2. 计算梯度图中每个梯度的权重alpha
    numerator = gradients.pow(2)
    denominator = 2 * gradients.pow(2)
    ag = activations * gradients.pow(3)
    denominator += ag.view(nc, -1).sum(-1, keepdim=True).view(nc, 1, 1)
    denominator = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator)
    )
    alpha = numerator / (denominator + 1e-7)        # ([bz, 1024, 14, 14])
    # 3. 计算梯度图权重weights
    relu_grad = gradients.clone()                   # ([bz, 1024, 14, 14])
    for idx, (score, grad) in enumerate(zip(pred_scores, gradients)):
        relu_grad[idx] = F.relu(score.exp() * grad)
    weights = (alpha * relu_grad).view(bz, nc, -1).sum(-1).view(bz, nc, 1, 1)   # ([bz, 1024, 1, 1])
    # 4. 计算一组batch的CAMs
    cams = (weights * activations).sum(1)            
    cams = t2n(F.relu(cams))                        # ([bz, 14, 14])
    CAMs = []
    for cam, image in zip(cams, images):
        cam = cv2.resize(cam, (h, w),               # 上采样，基于4x4像素邻域的3次插值法
                                interpolation=cv2.INTER_CUBIC)
        cam_normalized = normalize_scoremap(cam)    # (224, 224)
        # showImg("test", visualize(image, cam_normalized))

    # 对batch中的每一个图像进行反传
    # CAMs = []
    # for idx, (image, score) in enumerate(zip(images, pred_scores)):
    #     # 反向传播计算并hook梯度
    #     model.zero_grad()
    #     score.backward(retain_graph=True)
    #     activations = hook_values.activations[idx]  # ([2048, 14, 14])
    #     gradients = hook_values.gradients[idx]      # ([2048, 14, 14])
    #     # 计算gradcampp
    #     nc, h, w = activations.shape                # (num_channel, height, width)  
    #     # 计算alpha
    #     numerator = gradients.pow(2)
    #     denominator = 2 * gradients.pow(2)
    #     ag = activations * gradients.pow(3)
    #     denominator += ag.view(nc, -1).sum(-1, keepdim=True).view(nc, 1, 1)
    #     denominator = torch.where(
    #         denominator != 0.0, denominator, torch.ones_like(denominator))
    #     alpha = numerator / (denominator + 1e-7)

    #     relu_grad = F.relu(score.exp() * gradients)
    #     weights = (alpha * relu_grad).view(nc, -1).sum(-1).view(nc, 1, 1)   # ([2048, 1, 1])
    #     # shape => (H', W')
    #     cam = (weights * activations).sum(0)
    #     cam = t2n(F.relu(cam))                              # (14, 14)
    #     cam_resized = cv2.resize(cam, image_size,
    #                                 interpolation=cv2.INTER_CUBIC)
    #     cam_normalized = normalize_scoremap(cam_resized)    # (224, 224)
    #     # showImg("test", visualize(image, cam_normalized))
        CAMs.append(cam_normalized)

    return CAMs, logits, loss_cl


def mask_image(cam, image):
    omega = 10   # 该参数从GAIN论文中获取
    sigma = 0.5
    cam = torch.from_numpy(cam).cuda()
    mask = torch.sigmoid(omega * (cam - sigma)).squeeze()
    masked_image = image - (image * mask)
    return masked_image
    