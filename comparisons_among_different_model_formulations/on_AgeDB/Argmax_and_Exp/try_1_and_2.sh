#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
date
CUDA_VISIBLE_DEVICES=0 python -u try_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list orig_cutdown_train.txt --validate_list orig_validation.txt --test_list orig_test.txt --save /home/xjc/test/comparisons_of_different_losses/on_AgeDB/argmax_DEX/resnet18_version --n_epochs 65 --batch_size 32 --lr 0.001 --seed 2
date

date
CUDA_VISIBLE_DEVICES=0 python -u try_2.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list orig_cutdown_train.txt --validate_list orig_validation.txt --test_list orig_test.txt --save /home/xjc/test/comparisons_of_different_losses/on_AgeDB/argmax_DEX/resnet18_version --n_epochs 65 --batch_size 32 --lr 0.01 --seed 2
date
