#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#

date
CUDA_VISIBLE_DEVICES=0,1 python -u test_record_1.py --data_root /media/xjc/C14D581BDA18EBFA/age_estimation_generic_framework_related\(first_paper\)/Processed_Dataset/processed_Album_2_version_6 --test_list S1_test_probs_1.txt --save /home/xjc/test/ensemble_learning_age_estimation/on_morph2 --batch_size 32 --seed 2
date






