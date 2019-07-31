#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#
date
CUDA_VISIBLE_DEVICES=0 python -u test_record_1_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_AgeDB_version_1 --test_list modified_test_1.txt --save /home/xjc/test/ensemble_learning_age_estimation/try_on_AgeDB/resnet101_version/three_group --batch_size 32 --seed 2
date




