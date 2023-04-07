# -*- coding: utf-8 -*- #
'''
# ------------------------------------------------------------------------
# File Name:        WSOL_RS/main.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/10/19
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
#                           --> 遥感图像, 弱监督目标定位项目代码 <--        
#                   -- 主程序 TODO
#                   — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    <0> None
# Function List:    <0> None
# Class List:       <0> None
#                   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
#      |  <author>  | <version> |   <time>   |         <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#  <0> |    rjj     |   v1.0    | 2021/10/19 |          copy
# ------------------------------------------------------------------------
'''

import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim


from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer, compute_gradcampp, mask_image
from evaluation import BoxEvaluator, MaskEvaluator, configure_metadata
from util import string_contains_any, visualize, showImg, t2n
import wsol
import wsol.method


def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    """
    性能指标设置
    """
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None 
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
        "PN2": 11,
        "C45V2": 16
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        # print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter           = self.args.reporter
        self.model              = self._set_model()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.optimizer          = self._set_optimizer()
        self.loaders            = get_data_loader(
            data_roots              = self.args.data_paths,
            metadata_root           = self.args.metadata_root,
            batch_size              = self.args.batch_size,
            workers                 = self.args.workers,
            resize_size             = self.args.resize_size,
            crop_size               = self.args.crop_size,
            proxy_training_set      = self.args.proxy_training_set,
            num_val_sample_per_class= self.args.num_val_sample_per_class
        )

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        # 直接调取相应wsol中各个.py中的函数，如resnet50()
        model = wsol.__dict__[self.args.architecture](
            dataset_name        = self.args.dataset_name,
            architecture_type   = self.args.architecture_type,  # cam
            pretrained          = self.args.pretrained,
            num_classes         = num_classes,
            large_feature_map   = self.args.large_feature_map,
            pretrained_path     = self.args.pretrained_path,
            adl_drop_rate       = self.args.adl_drop_rate,
            adl_drop_threshold  = self.args.adl_threshold,
            acol_drop_threshold = self.args.acol_threshold
        )
        model = model.cuda()
        print(model)
        return model

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_features.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_classifiers.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _wsol_training(self, images, target):
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsol.method.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.model(images)
            logits = output_dict['logits']
            loss = (self.cross_entropy_loss(logits, target_a) * lam +
                    self.cross_entropy_loss(logits, target_b) * (1. - lam))
            return logits, loss

        if self.args.wsol_method == 'has':
            images = wsol.method.has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)

        output_dict = self.model(images, target)
        logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
                output_dict, target, spg_thresholds=self.args.spg_thresholds)
        else:
            loss = self.cross_entropy_loss(logits, target)

        return logits, loss

    def train(self, split):
        self.model.train()
        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, _) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            logits, loss = self._wsol_training(images, target)
            # print(batch_idx,": ", loss)
            # # if loss > 1000:
            # print(target)
            # print(logits)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['classification'].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def train_guide(self, epoch, split):
        self.model.train()
        loader = self.loaders[split]
        # 少量监督样本，记录
        if self.args.dataset_name == 'CUB':
            supervise_log = [0 for i in range(200)]  # 每类5个监督样本,200类
        elif self.args.dataset_name == 'OpenImages':
            supervise_log = [0 for i in range(100)]
        elif self.args.dataset_name == 'ILSVRC':
            supervise_log = [0 for i in range(1000)]
        elif self.args.dataset_name == 'PN2':
            supervise_log = [0 for i in range(11)]
        elif self.args.dataset_name == 'C45V2':
            supervise_log = [0 for i in range(16)]
        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()                                  # ([4])

            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            logits, loss_cl = self._wsol_training(images, targets)  # ([4, 200]), ([1])
            preds = logits.argmax(dim=1)                             # ([4])
            # print("Before:", loss_cl, pred)
            # 分阶段训练
            # if epoch >= (0.5*self.args.epochs):
            if 0:
                # print("train self loss")
                ################### 计算gradcampp，并获取抹去目标损失L_am->Loss_self ######################
                loss_am = 0
                ext_loss = 0
                # Eq1 transfer
                cams, logits, loss_cl = compute_gradcampp(images, targets, self.model, self.args.architecture)
                preds = logits.argmax(dim=1)

                for image, label, cam, image_id in zip(images, targets, cams, image_ids):
                    # Eq3, Eq4
                    # ([1, 3, 224, 224])
                    masked_image = mask_image(cam, image).unsqueeze(dim=0).float()
                    
                    # showImg("test", visualize(masked_image))
                    logits_masked = self.model(masked_image)['logits']      # 抹去目标后的图像经过网络输出
                    # TODO Eq5 sigmoid到底需要不，论文中没有，代码中有
                    logits_masked = torch.sigmoid(logits_masked)
                    # loss_am_temp = logits_masked.sum() / logits_masked.shape[1]
                    # 其他代码中的另一种思路
                    loss_am_temp = logits_masked[0][label]
                    loss_am += loss_am_temp

                    ################## 加入少量监督样本，计算L_ext ########################
                    # if supervise_log[int(t2n(label))] < 10:
                    #     supervise_log[int(t2n(label))] += 1
                    #     iou_computer = CAMComputer(
                    #         model=self.model,
                    #         architecture = self.args.architecture,
                    #         loader=self.loaders[split],
                    #         metadata_root=os.path.join(self.args.metadata_root, split),
                    #         mask_root=self.args.mask_root,
                    #         iou_threshold_list=self.args.iou_threshold_list,    # 30,50,70
                    #         dataset_name=self.args.dataset_name,
                    #         split=split,
                    #         cam_curve_interval=self.args.cam_curve_interval,    # cam阈值0-1间选取的间距
                    #         multi_contour_eval=self.args.multi_contour_eval,    # 多轮廓评估
                    #         log_folder=self.args.log_folder,
                    #     ).evaluator
                    #     multiple_iou = iou_computer.accumulate(cam, image_id, is_return = True)
                    #     ext_loss += 1 - np.max(multiple_iou)

                # Eq 6
                alpha = 5
                omega = 20
                loss_am /= images.size(0)
                ext_loss /= images.size(0)
                self_loss = loss_cl + alpha*loss_am
                # loss_sum = self_loss + omega*ext_loss
                loss_sum = self_loss

            else:
                loss_sum = loss_cl


            total_loss += loss_sum.item() * images.size(0)
            num_correct += (preds == targets).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100

        self.performance_meters[split]['classification'].update(classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            output_dict = self.model(images)
            pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, epoch, split):
        print("Evaluate epoch {}, split {}".format(epoch, split))   # split: tarin,val,test
        self.model.eval()

        # 计算分类准确率
        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        # 计算CAM
        cam_computer = CAMComputer(
            model=self.model,
            architecture = self.args.architecture,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,    # 30,50,70
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,    # cam阈值0-1间选取的间距
            multi_contour_eval=self.args.multi_contour_eval,    # 多轮廓评估
            log_folder=self.args.log_folder,
        )
        if self.args.wsol_method == 'gradcampp':
            cam_performance = cam_computer.compute_and_evaluate_gradcams(gt_known=True)  # 计算需要梯度cam
        else:
            cam_performance = cam_computer.compute_and_evaluate_cams()      # 计算传统的cam
        

        if self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split]['localization'].update(loc_score)

        if self.args.dataset_name in ('CUB', 'ILSVRC', 'PN2', 'C45V2'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance[idx])

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    trainer = Trainer()
    
    # # 训练前初始化性能评估
    # print("===========================================================")
    # print("Start epoch 0 ...")
    # trainer.evaluate(epoch=0, split='val')
    # trainer.print_performances()
    # trainer.report(epoch=0, split='val')
    # trainer.save_checkpoint(epoch=0, split='val')
    # print("Epoch 0 done.")
    
    # # 训练
    # for epoch in range(trainer.args.epochs):
    #     print("===========================================================")
    #     print("Start epoch {} ...".format(epoch + 1))
    #     trainer.adjust_learning_rate(epoch + 1)
    #     # train_performance = trainer.train(split='train')
    #     train_performance = trainer.train_guide(epoch, split='train')
    #     trainer.report_train(train_performance, epoch + 1, split='train')
    #     trainer.evaluate(epoch + 1, split='val')
    #     trainer.print_performances()
    #     trainer.report(epoch + 1, split='val')
    #     trainer.save_checkpoint(epoch + 1, split='val')
    #     print("Epoch {} done.".format(epoch + 1))
    
    # 测试
    print("===========================================================")
    print("Final epoch evaluation on test set ...")
    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    trainer.evaluate(trainer.args.epochs, split='test')
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()


if __name__ == '__main__':
    main()
