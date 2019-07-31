#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
date
CUDA_VISIBLE_DEVICES=0 python -u try_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --train_list modified_cutdown_train_1.txt --validate_list modified_validation_1.txt --test_list modified_test_1.txt --save /home/xjc/test/ensemble_learning_age_estimation/idea_try_1/try_on_AgeDB/resnet18_version --n_epochs 65 --batch_size 32 --lr 0.01 --seed 2
date

