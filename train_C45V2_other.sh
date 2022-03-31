# ######################################### Has ##############################################
# python main.py --dataset_name C45V2 \
#                --architecture resnet50 \
#                --wsol_method has \
#                --experiment_name B16.C45V2_resnet50_Has \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00361996526 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --has_grid_size 120 \
#                --has_drop_rate 0.66

python main.py --dataset_name C45V2 \
               --architecture vgg16 \
               --wsol_method has \
               --experiment_name B17.C45V2_vgg16_Has \
               --pretrained TRUE \
               --num_val_sample_per_class 5 \
               --large_feature_map FALSE \
               --batch_size 32 \
               --epochs 50 \
               --lr 0.00342891848 \
               --lr_decay_frequency 15 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 4 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type best \
               --has_grid_size 65 \
               --has_drop_rate 0.47

# python main.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method gradcampp \
#                --experiment_name B18.C45V2_inception_v3_Has \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.01400558066 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --has_grid_size 170 \
#                --has_drop_rate 0.26

# ######################################### ACoL ##############################################
python main.py --dataset_name C45V2 \
               --architecture resnet50 \
               --wsol_method acol \
               --experiment_name B19.C45V2_resnet50_acol \
               --pretrained TRUE \
               --num_val_sample_per_class 5 \
               --large_feature_map TRUE \
               --batch_size 32 \
               --epochs 50 \
               --lr 0.00001700073 \
               --lr_decay_frequency 15 \
               --weight_decay 1.00E-04 \
               --override_cache FALSE \
               --workers 4 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type best \
               --acol_threshold 0.57

# python main.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method acol \
#                --experiment_name B20.C45V2_vgg16_acol \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00010569582 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --acol_threshold 0.89

# python main.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method acol \
#                --experiment_name B21.C45V2_inception_v3_acol \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00658832165 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --acol_threshold 0.79

# ######################################### SPG ##############################################
# python main.py --dataset_name C45V2 \
#                --architecture resnet50 \
#                --wsol_method spg \
#                --experiment_name B22.C45V2_resnet50_spg \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00004521947 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --spg_threshold_1h 0.41 \
#                --spg_threshold_1l 0.35 \
#                --spg_threshold_2h 0.24 \
#                --spg_threshold_2l 0.21 \
#                --spg_threshold_3h 0.12 \
#                --spg_threshold_3l 0.08

# python main.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method spg \
#                --experiment_name B23.C45V2_vgg16_spg \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00012558923 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --spg_threshold_1h 0.29 \
#                --spg_threshold_1l 0.23 \
#                --spg_threshold_2h 0.03 \
#                --spg_threshold_2l 0.02 \
#                --spg_threshold_3h 0.82 \
#                --spg_threshold_3l 0.59

# python main.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method spg \
#                --experiment_name B24.C45V2_inception_v3_spg \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.0003360682 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --spg_threshold_1h 0.28 \
#                --spg_threshold_1l 0.02 \
#                --spg_threshold_2h 0.17 \
#                --spg_threshold_2l 0.03 \
#                --spg_threshold_3h 0.47 \
#                --spg_threshold_3l 0.31
               
# ######################################### ADL ##############################################
# python main.py --dataset_name C45V2 \
#                --architecture resnet50 \
#                --wsol_method adl \
#                --experiment_name B25.C45V2_resnet50_adl \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.01253438325 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --adl_threshold 0.76 \
#                --adl_drop_rate 0.24

# python main.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method adl \
#                --experiment_name B26.C45V2_vgg16_adl \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00002430601 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --adl_threshold 0.72 \
#                --adl_drop_rate 0.33

# python main.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method adl \
#                --experiment_name B27.C45V2_inception_v3_adl \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00038655054 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --adl_threshold 0.86 \
#                --adl_drop_rate 0.69

# ######################################### CUTMIX ##############################################
# python main.py --dataset_name C45V2 \
#                --architecture resnet50 \
#                --wsol_method cutmix \
#                --experiment_name B28.C45V2_resnet50_cutmix \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map TRUE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.0003333722 \
#                --lr_decay_frequency 15 \
#                --weight_decay 1.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --cutmix_beta 1.35 \
#                --cutmix_prob 0.24

# python main.py --dataset_name C45V2 \
#                --architecture vgg16 \
#                --wsol_method cutmix \
#                --experiment_name B29.C45V2_vgg16_cutmix \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00003220934 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --cutmix_beta 0.85 \
#                --cutmix_prob 0.79

# python main.py --dataset_name C45V2 \
#                --architecture inception_v3 \
#                --wsol_method cutmix \
#                --experiment_name B30.C45V2_inception_v3_cutmix \
#                --pretrained TRUE \
#                --num_val_sample_per_class 5 \
#                --large_feature_map FALSE \
#                --batch_size 32 \
#                --epochs 50 \
#                --lr 0.00225171726 \
#                --lr_decay_frequency 15 \
#                --weight_decay 5.00E-04 \
#                --override_cache FALSE \
#                --workers 4 \
#                --box_v2_metric True \
#                --iou_threshold_list 30 50 70 \
#                --eval_checkpoint_type best \
#                --cutmix_beta 0.04 \
#                --cutmix_prob 0.30