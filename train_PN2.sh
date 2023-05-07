# CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name PN2 \
#                --architecture resnet50 \
#                --wsol_method gradcampp \
#                --experiment_name 1.PN2_resnet50_Gpp_Lext \
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

# CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name PN2 \
#                --architecture vgg16 \
#                --wsol_method gradcampp \
#                --experiment_name 9.PN2_vgg16_Gpp_Lext \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 16 \
#                --epochs 50 \
#                --lr 0.0002268269 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best
               
CUDA_VISIBLE_DEVICES=0 python main_modify.py --dataset_name PN2 \
               --architecture inception_v3 \
               --wsol_method gradcampp \
               --experiment_name 10.PN2_inception_v3_Gpp_Lext \
               --pretrained TRUE \
               --num_val_sample_per_class 5 \
               --large_feature_map TRUE \
               --batch_size 16 \
               --epochs 50 \
               --lr 0.00124844746 \
               --lr_decay_frequency 15 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 4 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type last


################################# CAM ##################################
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name PN2 \
#                --architecture resnet50 \
#                --wsol_method cam \
#                --experiment_name c10.PN2_resnet50_cam \
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
#                --eval_checkpoint_type last

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name PN2 \
#                --architecture vgg16 \
#                --wsol_method cam \
#                --experiment_name 7.PN2_vgg16_cam \
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
#                --eval_checkpoint_type last

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset_name PN2 \
#                --architecture inception_v3 \
#                --wsol_method cam \
#                --experiment_name c12.PN2_inception_v3_cam \
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
#                --eval_checkpoint_type last