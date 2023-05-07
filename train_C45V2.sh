#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name C45V2 \
#                --architecture resnet50 \
#                --wsol_method gradcampp \
#                --experiment_name c17.C45V2_resnet50_Gpp_Lself \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 16 \
#                --epochs 25 \
#                --lr 0.00023222617 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best

# CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method gradcampp \
#                --experiment_name c20.C45V2_vgg16_Gpp \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 10 \
#                --lr 0.00005268269 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best
CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name C45V2 \
               --architecture vgg16 \
               --wsol_method gradcampp \
               --experiment_name test.tmp \
               --pretrained TRUE \
               --num_val_sample_per_class 5 \
               --large_feature_map FALSE \
               --batch_size 32 \
               --epochs 10 \
               --lr 0.00005268269 \
               --lr_decay_frequency 15 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 4 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type best

# CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method gradcampp \
#                --experiment_name c18.C45V2_vgg16_Gpp_Lself \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00005268269 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best

# CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method gradcampp \
#                --experiment_name c19.C45V2_inception_v3_Gpp_Lself \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 16 \
#                --epochs 25 \
#                --lr 0.00224844746 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best

               
################################# CAM ##################################
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name C45V2 \
#                --architecture resnet50 \
#                --wsol_method cam \
#                --experiment_name c10.C45V2_resnet50_cam \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 16 \
#                --epochs 50 \
#                --lr 0.00023222617 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method cam \
#                --experiment_name c11.C45V2_vgg16_cam \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 16 \
#                --epochs 50 \
#                --lr 0.00001268269 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method cam \
#                --experiment_name c12.C45V2_inception_v3_cam \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 16 \
#                --epochs 50 \
#                --lr 0.00224844746 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best